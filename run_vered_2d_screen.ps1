# =============================================================================
#  run_vered_2d_screen.ps1
# -----------------------------------------------------------------------------
#  2D screening grid: momentum x learning rate at the current best operating
#  point (gamma=0.9, lambda=1e-4, clip=300), 1000 steps per cell.
#
#  Comes AFTER the momentum-bug fix in stability_benchmark.py:378.  This is
#  the first valid test of how Vered actually responds to (mom, lr).
#
#  Grid: 4 mom * 8 lr = 32 cells, ~15 min/cell -> ~8 hours total full run.
#  Resume support: any cell with existing JSON is skipped, so re-runs only
#  cost wall time for new cells (currently ~13 unrun cells after the small-LR
#  + 6e-3 extension -> ~3.25 hours).
#
#  Usage:
#    .\run_vered_2d_screen.ps1
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

Section "Running 2D (mom x lr) screening grid at 1000 steps"
& python benchmark\vered_2d_mom_lr_screen.py
$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -eq 0) {
    Info "Screen complete. Results in benchmark\results\vered_2dscreen_*.json"
} else {
    Fail "Screen exited with code $exitCode"
}
