# =============================================================================
#  run_benchmark.ps1  -  OlsVered GPU Benchmark launcher (Windows / PowerShell)
# =============================================================================
#
#  Mirrors run_benchmark.sh for native Windows. Assumes the .venv is already
#  built (Windows-style: .venv\Scripts\) with torch+cu128, olssm, transformers,
#  datasets and peft already installed. Does NOT run the setup steps.
#
#  Usage from a PyCharm / PowerShell terminal in the repo root:
#    .\run_benchmark.ps1                                 # all tasks, defaults
#    .\run_benchmark.ps1 -Task bert -Fresh
#    .\run_benchmark.ps1 -Task transformer -LrSweepTransformer
#    .\run_benchmark.ps1 -Task all -Skip veredkfac
#
#  If you get "running scripts is disabled on this system", run once:
#    Set-ExecutionPolicy -Scope Process Bypass
#
#  Defaults are tuned for the canonical re-run described in
#  SESSION_CONTEXT_revive-kfac.md:
#    - Task: all 4 tasks on a fresh hardware-consistent pod
#    - Steps: BERT 8000 / Transformer 5000 / MLP 3000 / CIFAR 5000
#    - LR: 0.008 for K-FAC on transformer (best from prior LR sweep)
#    - Wall cap: 0 (no cap) - important so BERT K-FAC can finish
#    - Fresh: enabled - clears stale BERT checkpoints before launching
# =============================================================================

[CmdletBinding()]
param(
    [ValidateSet("mlp","bert","cifar","scaling","transformer","all")]
    [string]$Task = "all",

    [int]$StepsMlp         = 3000,
    [int]$StepsBert        = 8000,
    [int]$StepsCifar       = 5000,
    [int]$StepsTransformer = 5000,
    [int]$StepsScaling     = 300,

    # 0 = no wall-time cap (RECOMMENDED for K-FAC BERT, otherwise it gets
    # truncated mid-training - this is what produced the partial OlsSMKFAC
    # 902-step result in the previous session).
    [double]$MaxWallBert   = 0,

    # Clear stale BERT checkpoints before starting. SESSION_CONTEXT_revive-kfac
    # says to do this; defaults to ON for a clean rerun.
    [switch]$Fresh         = $true,

    # Comma-separated optimizer names to skip.
    # Valid: adam, classickfac, olssmkfac, veredkfac
    [string]$Skip          = "",

    # KFAC LR hierarchy: ClassicKFAC : OlsSMKFAC : VeredKFAC = 1 : 1.5 : 2.25
    # All three tolerate momentum=0.9 differently; ClassicKFAC needs the lowest.
    # OlsSMKFAC 0.008 is the prior LR-sweep optimum (ppl=368).
    [double]$LrOlsTransformer   = 0.008,
    [double]$LrClsTransformer   = 0.0053,    # OlsSM / 1.5
    [double]$LrVeredTransformer = 0.012,     # OlsSM * 1.5

    # BERT VeredKFAC LR (in-code default 4.5e-3 = OlsSMKFAC 3e-3 * 1.5)
    # Override only if you want to deviate from the hierarchy.
    [double]$LrVeredBert        = 0,         # 0 = use in-code default

    [switch]$LrSweepTransformer,

    # Skip the GPU verification step (faster restart). The venv is assumed
    # already built either way; there is no setup phase in the .ps1.
    [switch]$NoCheck
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Section($msg) { Write-Host "`n== $msg ==" -ForegroundColor Cyan }
function Info($msg)    { Write-Host "[INFO]  $msg" -ForegroundColor Green }
function Warn($msg)    { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Fail($msg)    { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# ----- Activate venv ---------------------------------------------------------
Section "Virtual environment"

$VenvActivate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvActivate)) {
    Fail ".venv not found at $VenvActivate. Build it first (maturin develop --release)."
}

if (-not $env:VIRTUAL_ENV) {
    Info "Activating $VenvActivate"
    & $VenvActivate
} else {
    Info "Already in venv: $env:VIRTUAL_ENV"
}

# ----- GPU sanity check ------------------------------------------------------
if (-not $NoCheck) {
    Section "GPU check"

    $GpuOut = python -c @"
import sys, torch
if not torch.cuda.is_available():
    print('NO_CUDA'); sys.exit(1)
n = torch.cuda.get_device_name(0)
v = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'OK|{n}|{v:.1f}')
"@
    if ($LASTEXITCODE -ne 0 -or $GpuOut -notlike "OK|*") {
        Fail "CUDA not available. Got: $GpuOut"
    }
    $parts = $GpuOut -split '\|'
    Info ("GPU: {0}  ({1} GB VRAM)" -f $parts[1], $parts[2])

    if ($Task -eq "bert" -or $Task -eq "all") {
        $vram = [int]([double]$parts[2])
        if ($vram -lt 12) {
            Warn "BERT task wants >= 12 GB VRAM. Detected $vram GB - it may OOM."
        }
        if ($vram -lt 32) {
            Warn "ClassicKFAC BERT peak was 25.7 GB on prior run. Detected $vram GB - watch for OOM."
        }
    }
}

# ----- Build the Python invocation ------------------------------------------
Section "Launching benchmark"

$PyArgs = @(
    "benchmark/gpu_benchmark.py",
    "--task",                  $Task,
    "--max-steps-mlp",         $StepsMlp,
    "--max-steps-bert",        $StepsBert,
    "--max-steps-cifar",       $StepsCifar,
    "--max-steps-scaling",     $StepsScaling,
    "--max-steps-transformer", $StepsTransformer,
    "--max-wall-bert",         $MaxWallBert,
    "--lr-ols-transformer",    $LrOlsTransformer,
    "--lr-cls-transformer",    $LrClsTransformer
)

if ($Skip)                   { $PyArgs += @("--skip", $Skip) }
if ($Fresh)                  { $PyArgs += "--fresh" }
if ($LrSweepTransformer)     { $PyArgs += "--lr-sweep-transformer" }
if ($LrVeredTransformer -gt 0) { $PyArgs += @("--lr-vered-transformer", $LrVeredTransformer) }
if ($LrVeredBert -gt 0)        { $PyArgs += @("--lr-vered-bert",        $LrVeredBert) }

Info ("Command: python " + ($PyArgs -join " "))
Write-Host ""

# ----- Run it ----------------------------------------------------------------
& python @PyArgs
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Info "Benchmark finished. Results in benchmark/results/"
} else {
    Fail "Benchmark exited with code $exitCode. Check the log above."
}
