# =============================================================================
#  run_stability.ps1  -  K-FAC numerical stability benchmark launcher
# =============================================================================
#
#  Runs benchmark/stability_benchmark.py on the SmallGPT/WikiText-2 task to
#  measure how high a learning rate each K-FAC variant can tolerate before
#  diverging, then runs full convergence comparisons at those max stable LRs.
#
#  Usage:
#    .\run_stability.ps1                       # both phases
#    .\run_stability.ps1 -Phase 1              # LR sweep only
#    .\run_stability.ps1 -Phase 2              # convergence only (needs phase 1 done)
#    .\run_stability.ps1 -Variants "VeredKFAC,OlsSMKFAC"   # subset
#    .\run_stability.ps1 -ProbeSteps 500       # shorter Phase 1 probes
#
#  Outputs to benchmark/results/:
#    stability_phase1_sweep.json
#    stability_phase2_runs.json
#    stability_summary.csv
#    stability_lr_frontier.png
#    stability_convergence.png
# =============================================================================

[CmdletBinding()]
param(
    [ValidateSet("1","2","all")]
    [string]$Phase = "all",

    [string]$Variants = "ClassicKFAC,OlsSMKFAC,VeredKFAC",

    # Per-probe step budget for Phase 1.  Lower = faster sweep but slow-divergers
    # may get classified as stable.  Default 1000 catches ~all divergence modes.
    [int]$ProbeSteps  = 1000,

    # Steps for the Phase 2 convergence runs (matches gpu_benchmark.py default
    # so results are directly comparable).
    [int]$Phase2Steps = 5000
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Section($msg) { Write-Host "`n== $msg ==" -ForegroundColor Cyan }
function Info($msg)    { Write-Host "[INFO]  $msg" -ForegroundColor Green }
function Fail($msg)    { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

Section "Virtual environment"
$VenvActivate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvActivate)) {
    Fail ".venv not found at $VenvActivate."
}
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

Section "Launching stability benchmark"
$PyArgs = @(
    "benchmark/stability_benchmark.py",
    "--phase",        $Phase,
    "--variants",     $Variants,
    "--probe-steps",  $ProbeSteps,
    "--phase2-steps", $Phase2Steps
)
Info ("Command: python " + ($PyArgs -join " "))
Write-Host ""

& python @PyArgs
$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -eq 0) {
    Info "Stability benchmark finished. See benchmark/results/stability_*"
} else {
    Fail "Stability benchmark exited with code $exitCode."
}
