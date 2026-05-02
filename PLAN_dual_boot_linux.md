# Plan: dual-boot Windows + Linux for ML/CUDA work

*Created: 2026-05-02. Context: switching primary dev environment to Linux to
get more stable NVIDIA driver behaviour for sustained K-FAC / CUDA workloads
after a `KMODE_EXCEPTION_NOT_HANDLED` BSOD on Windows during the stability
benchmark overnight run.*

## Why bother

- Windows BSOD'd at 3:09 AM during a K-FAC stability sweep. Bug check `0x1E`
  with `STATUS_ILLEGAL_INSTRUCTION` from a kernel driver, almost certainly the
  NVIDIA display driver under sustained CUDA load. Linux's NVIDIA driver
  crashes much less under the same workload, and when it does crash the
  recovery is graceful (restart the driver) rather than a hard reboot.
- Windows Update can forcibly reboot mid-run; Linux can be configured so
  nothing reboots without explicit consent.
- The `run_benchmark.sh` shell launcher and most of the K-FAC tooling were
  written for Linux first - the `.ps1` adapters are workarounds.
- `nvidia-smi -pl` is fully supported on Linux for power capping. On laptop
  Windows it's often locked by the OEM.
- `tmux + nohup` lets a long training job survive logout / network drops.
  Windows has no clean equivalent.

## Distro choice

**Ubuntu 24.04 LTS** is the recommended pick. NVIDIA's reference distro for
CUDA, PyTorch wheels assume its glibc/CUDA versions, official docs are
written against it, and the Ubuntu NVIDIA driver packaging is well tested.

Alternative: **Pop!_OS 22.04** by System76 (Ubuntu-based) - has NVIDIA
drivers preinstalled and laptop power management baked in. Easier first
install but less mainstream documentation.

Avoid niche distros (Arch, Fedora) for a first Linux setup unless already
comfortable with their package management.

## Total time

About 2 hours including driver setup, plus a few hours of unattended
BitLocker decryption beforehand if applicable.

| Step                          | Effort   |
|-------------------------------|----------|
| Pre-flight (Windows side)     | 30 min   |
| BitLocker decrypt (background)| 1-3 hr (unattended) |
| Resize Windows partition      | 10 min   |
| Create Linux installer USB    | 10 min   |
| Install Ubuntu                | 30 min   |
| NVIDIA driver + CUDA          | 30 min   |
| PyTorch + venv + olssm rebuild| 30 min   |

## Pre-flight checklist (do all, in order)

1. **Back up everything important off C:\.** Partition resize can fail.
   External drive or cloud, doesn't matter, but assume there's a non-zero
   chance the Windows install gets corrupted.

2. **Check free disk space.** From cmd:
   ```cmd
   wmic logicaldisk get size,freespace,caption
   ```
   Target at least 150 GB free on `C:` for a comfortable Linux install.
   200 GB is better. Uninstall stuff first if needed.

3. **Disable BitLocker if enabled** (almost certainly is on a modern Win11
   laptop):
   ```cmd
   manage-bde -status C:
   ```
   If it shows `Protection On`, disable it via Settings - Privacy & Security
   - Device Encryption - Off. This **decrypts** the drive in the background;
   it can take 1-3 hours. Let it finish completely before proceeding.
   Necessary because the Linux installer cannot safely resize an encrypted
   partition.

4. **Disable Windows Fast Startup.** Control Panel - Power Options - "Choose
   what the power buttons do" - uncheck "Turn on fast startup". Fast Startup
   hibernates kernel state across reboot; if Linux mounts the Windows
   partition while it's in this state, the NTFS filesystem can be corrupted.

5. **Note your Secure Boot state.** Settings - Update & Security - Recovery -
   Advanced startup - Restart now - Troubleshoot - Advanced - UEFI Firmware
   Settings. Ubuntu 24.04 supports Secure Boot via signed shim, but the
   NVIDIA proprietary driver requires enrolling a MOK (Machine Owner Key)
   the first boot after install - one extra blue screen prompt to walk
   through, no big deal.

6. **Note current Windows version + edition.** Settings - System - About.
   Useful for restoring Windows if something goes wrong.

## Resize Windows partition

After BitLocker is fully decrypted:

1. Right-click Start - "Disk Management".
2. Right-click the C: partition - "Shrink Volume...".
3. Enter the amount to shrink in MB (e.g., 200000 for 200 GB).
4. Click Shrink. The freed space appears as "Unallocated" - leave it that
   way, the Ubuntu installer will format it.

If shrink fails or won't shrink as much as you want, run in admin cmd:
```cmd
defrag C: /U /X
```
then retry. Persistent files (pagefile, hibernation, system restore points)
sometimes prevent shrinkage; disable them temporarily if needed.

## Create Linux installer USB

1. Download Ubuntu 24.04 LTS desktop ISO from
   https://ubuntu.com/download/desktop (~5 GB).
2. Download Rufus from https://rufus.ie (free, portable Windows tool).
3. Insert a USB stick (>=8 GB; will be wiped).
4. Open Rufus, select the USB and the Ubuntu ISO. Partition scheme: GPT.
   Target: UEFI (non-CSM). File system: FAT32. Click Start.
5. When asked about ISOHybrid mode, choose "Write in ISO Image mode".

## Install Ubuntu alongside Windows

1. Reboot. Press the boot-menu key during POST (varies: F12, F11, ESC, or
   F2 - look at the splash screen). Select the USB.
2. Choose "Try or Install Ubuntu".
3. Walk through language, keyboard, Wi-Fi.
4. **Installation type: choose "Install Ubuntu alongside Windows Boot
   Manager"**. The installer auto-detects the unallocated space.
5. (Optional) Switch to "Something else" if you want to manually create
   partitions:
   - `/boot/efi` - 512 MB FAT32, mounted at `/boot/efi`
   - `/` (root) - 80 GB ext4, mounted at `/`
   - `/home` - rest of the space, ext4, mounted at `/home`
   - swap - skip (Ubuntu uses a swap file by default; saves a partition)
6. Pick username + password.
7. Reboot when prompted, remove USB. The GRUB menu now shows both Ubuntu
   and "Windows Boot Manager" - pick whichever you want.

## NVIDIA driver + CUDA + PyTorch

After first boot into Ubuntu, open a terminal:

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install build essentials and tools
sudo apt install -y build-essential git curl wget tmux htop nvtop

# 3. Install NVIDIA driver (proprietary, latest tested)
sudo ubuntu-drivers autoinstall
# OR pick a specific version:
# sudo apt install -y nvidia-driver-555

# 4. Reboot to load the driver
sudo reboot
```

After reboot, verify:
```bash
nvidia-smi   # should show RTX 3080 Laptop GPU
```

Install Python + venv:
```bash
sudo apt install -y python3.12 python3.12-venv python3-pip

# Create venv mirroring the Windows .venv
mkdir -p ~/OlsVered && cd ~/OlsVered
python3.12 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA 12.8 (matches the Windows install)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install Rust toolchain for olssm
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Install other deps + build olssm
pip install maturin
pip install -r requirements.txt   # once you have the source code synced
maturin develop --release
```

## Mounting the Windows partition (access existing files)

The Ubuntu Files app will show "Windows" in the sidebar. Click to mount.
The path will be something like `/media/<username>/Windows/Users/Admin/OlsVered/`
- you can `cp -r` your project over to the Linux home directory.

Or do the actual sync from a terminal:
```bash
# One-time mount (auto-detects)
sudo mkdir -p /mnt/win
sudo mount /dev/nvme0n1p3 /mnt/win   # adjust device per `lsblk`
cp -r /mnt/win/Users/Admin/OlsVered ~/OlsVered
```

(Skip if already cloned via git.)

## What can go wrong (and recovery)

- **BitLocker not fully decrypted before resize** - installer fails or shows
  the partition as encrypted. Fix: cancel install, finish decryption in
  Windows, restart.
- **Fast Startup left enabled** - first mount of Windows partition from
  Linux corrupts NTFS. Fix: boot Windows, run `chkdsk C: /f`. To prevent:
  do step 4 of pre-flight.
- **Wi-Fi doesn't work post-install** - common on Realtek/Broadcom cards.
  Fix: tether USB from phone, `sudo apt install` the right firmware
  package. Most Intel Wi-Fi cards work out of the box.
- **Windows Update overwrites GRUB later** - reboots into Windows only.
  Fix: boot from the same Ubuntu USB in "Try Ubuntu" mode, install
  `boot-repair`, run it. ~10 min, fully recoverable.
- **NVIDIA driver and Secure Boot fight** - first reboot prompts to enroll
  a MOK key. Walk through the blue prompt screens, set a password, reboot
  again. After that it's permanent.
- **Hybrid graphics (NVIDIA Optimus + Intel iGPU) confusion** - run
  `prime-select query`. If it says `intel` and you want NVIDIA always-on:
  `sudo prime-select nvidia` then reboot. For laptops, `on-demand` mode
  is usually best (iGPU for desktop, dGPU for CUDA workloads).

## Plan B: WSL2 (zero-risk alternative)

If the dual-boot risk feels too high, **WSL2** (Windows Subsystem for Linux)
gives Ubuntu running inside Windows with full CUDA support. No partitioning,
no risk to the Windows install. Trade-offs:

- Pro: no risk to Windows, can keep using Windows for everything else
- Pro: existing files accessible at `/mnt/c/Users/Admin/OlsVered`
- Pro: install in 10 minutes
- Con: ~5-10% performance overhead vs native Linux
- Con: still subject to Windows Update reboots
- Con: still subject to Windows NVIDIA driver bugs (the underlying driver
  is the Windows one, virtualized through to WSL)

To install WSL2 with Ubuntu:
```cmd
wsl --install -d Ubuntu-24.04
```

If the BSOD recurs even after the NVIDIA driver update + power limit, WSL2
won't help - the driver is shared. In that case dual boot is the correct
remedy.

## What I gain after the switch

- The shell launcher (`run_benchmark.sh`) just works as-written - the `.ps1`
  adapters become unnecessary.
- `nvidia-smi -pl` for power capping (no MSI Afterburner required).
- CUDA Graphs (the planned VeredKFAC speedup in
  `TASK_vered_gpu_saturation.md`) is more stable on Linux.
- Long-running benchmarks survive logout via `tmux` / `nohup`. No more
  "left it running and Windows Update rebooted".
- Fewer driver-related crashes under sustained compute. When they do
  happen, recovery is graceful (Ctrl+Alt+F2, restart driver) rather than
  a hard reboot losing all training state.
