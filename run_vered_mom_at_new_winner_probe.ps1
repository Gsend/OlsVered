# =============================================================================
#  run_vered_mom_at_new_winner_probe.ps1
# -----------------------------------------------------------------------------
#  Vered mom-axis re-sweep at the current best (gamma=0.9, lambda=1e-4,
#  clip=300).  Stage B of the Vered grid concluded mom-axis didn't matter,
#  but that conclusion was at the OLD (gamma=0.3, lambda=1e-5) operating
#  point.  Re-verify at the new winner config.
#
#  Sweep: momentum in {0.7, 0.5, 0.0, 0.9}    # 0.3 reused as reference
#
#  Wall time: ~5 hours (4 runs x 75 min).
#  Resume support: any cell with existing JSON is skipped.
#
#  If Vered also benefits from mom=0.7 (like Classic did), Vered's
#  actual best may be 560-590 instead of 618.  The fair Vered-vs-Classic
#  comparison depends on this.
#
#  Usage:
#    .\run_vered_mom_at_new_winner_probe.ps1
# =============================================================================

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Section($msg) { Write-Host "`n== $msg ==" -ForegroundColor Cyan }
function Info($msg)    { Write-Host "[INFO]  $msg" -ForegroundColor Green }
function Fail($msg)    { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

Section "Virtual environment"
$VenvActivate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvActivate)) { Fail ".venv not found at $VenvActivate" }
if (-not $env:VIRTUAL_ENV) {
    Info "Activating $VenvActivate"
    & $VenvActivate
} else {
    Info "Already in venv: $env:VIRTUAL_ENV"
}

Section "GPU check"
$GpuOut = python -c "import torch; n = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'; v = torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0; print(f'{n}|{v:.1f}')"
if ($GpuOut -like "NONE|*") { Fail "CUDA not available." }
$parts = $GpuOut -split '\|'
Info ("GPU: {0}  ({1} GB VRAM)" -f $parts[0], $parts[1])

if (Test-Path "$env:USERPROFILE\.cache\huggingface\datasets\Salesforce___wikitext") {
    $env:HF_DATASETS_OFFLINE = "1"
    $env:HF_HUB_OFFLINE      = "1"
    Info "HF cache present; HF_DATASETS_OFFLINE=1 set"
}

Section "Running Vered mom-axis re-sweep at the new winner config"
& python benchmark\vered_mom_at_new_winner_probe.py
$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -eq 0) {
    Info ("Probe complete.  Results in " +
          "benchmark\results\vered_mom_g0.90_m{0.0,0.5,0.7,0.9}_l1e-04_c300.json")
} else {
    Fail "Probe exited with code $exitCode"
}
