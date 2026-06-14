"""
Move stale Classic/Vered MNIST autoencoder JSONs (run with the wrong
hyperparameters before screen 1/2/3) into an archive directory so the
main sweep re-runs them with the tuned defaults.

AdamW results are kept in place (don't depend on K-FAC hyperparams).
WGSO + SINGD results are kept in place (documented as broken on this AE).
"""
from pathlib import Path
import shutil
import time

RESULTS = Path(__file__).resolve().parent / "results"
ARCHIVE = RESULTS / f"archive_ae_pretune_{time.strftime('%Y%m%d_%H%M%S')}"
ARCHIVE.mkdir(exist_ok=True)

moved = 0
# NOTE: this list is rotated each time we re-tune a method.  Comment out the
# patterns you want to KEEP before running.  Currently only bf16 AdamW is
# stale (lr=1e-2 overflowed; re-running at lr=3e-4 — see adamw_bf16_screen).
for pat in [
    # "ae_mnist_fp32_classic_seed*.json",   # keep — final tuned result (~31)
    # "ae_mnist_bf16_classic_seed*.json",   # keep — shows the kappa^2 collapse (~70)
    # "ae_mnist_fp32_vered_seed*.json",     # keep — final tuned result (~31)
    # "ae_mnist_bf16_vered_seed*.json",     # keep — shows the stability claim (~31)
    # "ae_mnist_fp32_adamw_seed*.json",     # keep — final tuned result (~86)
    "ae_mnist_bf16_adamw_seed*.json",        # stale — overflowed at step 264-331
]:
    for p in RESULTS.glob(pat):
        target = ARCHIVE / p.name
        shutil.move(str(p), str(target))
        print(f"  moved {p.name}")
        moved += 1

print(f"\n{moved} files moved to {ARCHIVE.name}")
