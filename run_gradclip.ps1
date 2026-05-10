# =============================================================================
#  run_gradclip.ps1  -  Grad-clip frontier benchmark for K-FAC variants
# =============================================================================
#
#  Tests at which grad_clip value each K-FAC variant starts diverging, with
#  fixed LR=8e-3 and momentum=0.9 (the deployment optimum).  Then runs full
#  convergence at each variant's max stable grad_clip.
#
#  Usage:
#    .\run_gradclip.ps1                       # both phases
#    .\run_gradclip.ps1 -Phase 1              # frontier sweep only
#    .\run_gradclip.ps1 -Phase 2              # convergence only (needs phase 1)
#    .\run_gradclip.ps1 -Variants "OlsSMKFAC,VeredKFAC"
#    .\run_gradclip.ps1 -ProbeSteps 500
#
#  Outputs to benchmark/results/:
#    gradclip_phase1_sweep.json
#    gradclip_phase2_runs.json
#    gradclip_summary.csv
#    gradclip_frontier.png
#    gradclip_convergence.png
# =============================================================================

[CmdletBinding()]
param(
    [ValidateSet("1","2","all")]
    [string]$Phase = "all",

    # Vered-only by default after the EMA-blend fix in vered_kfac.py.
    # Classic and OlsSM are unchanged by that fix; Phase 1 results from
    # prior runs for those two stay valid.  Override with
    #   .\run_gradclip.ps1 -Variants "ClassicKFAC,OlsSMKFAC,VeredKFAC"
    # to run all three again (e.g. clean cross-comparison rerun).
    [string]$Variants = "VeredKFAC",

    [int]$ProbeSteps  = 1000,
    [int]$Phase2Steps = 5000,

    # Per-variant clip overrides for Phase 2.  Format:
    #   "ClassicKFAC=60,OlsSMKFAC=60,VeredKFAC=120"
    # Variants not listed fall back to Phase 1 max_stable.
    [string]$Phase2Clips = "",

    # Suffix on Phase 2 output filenames so multiple configs can coexist.
    # E.g. -Phase2Tag "ema_fixed"  ->  gradclip_phase2_runs_ema_fixed.json
    [string]$Phase2Tag = "",

    # default | direction_quality   (low-clip + high-LR regime)
    [ValidateSet("default","direction_quality")]
    [string]$Regime = "default"
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

Section "Launching grad-clip benchmark"
$PyArgs = @(
    "benchmark/gradclip_benchmark.py",
    "--phase",        $Phase,
    "--variants",     $Variants,
    "--probe-steps",  $ProbeSteps,
    "--phase2-steps", $Phase2Steps,
    "--regime",       $Regime
)
if ($Phase2Clips) { $PyArgs += @("--phase2-clips", $Phase2Clips) }
if ($Phase2Tag)   { $PyArgs += @("--phase2-tag",   $Phase2Tag) }
Info ("Command: python " + ($PyArgs -join " "))
Write-Host ""

& python @PyArgs
$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -eq 0) {
    Info "Benchmark finished. See benchmark/results/gradclip_*"
} else {
    Fail "Benchmark exited with code $exitCode."
}
