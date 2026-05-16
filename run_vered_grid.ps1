# =============================================================================
#  run_vered_grid.ps1  -  Three-stage coordinate-descent grid for VeredKFAC
# =============================================================================
#
#  Stage A: gamma sweep    at (mom=0.0, lambda=1e-5, lr=8e-3)   ~3 hours
#  Stage B: momentum sweep at (best gamma, lambda=1e-5)          ~3-5 hours
#  Stage C: damping sweep  at (best gamma, best mom)             ~3-5 hours
#
#  Total wall time: 12-14 hours (depending on legacy reuse).
#  Resume support: skipping any (gamma, mom, lambda) cell whose JSON exists.
#
#  Reference points to beat:
#    Vered post-fix (gamma=0.7, mom=0.9, lambda=1e-5):  908 ppl
#    Vered post-fix (gamma=0.7, mom=0.0, lambda=1e-5):  821 ppl   <-- prior best
#    Classic        (mom=0.9, lambda=1e-4):             776 ppl
#    Vered pre-fix  (buggy EMA, mom=0.9):               756 ppl   <-- target
#
#  Usage:
#    .\run_vered_grid.ps1
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

# Force HF offline on retry-after-failure scenarios; safe to keep on if cache exists.
if (Test-Path "$env:USERPROFILE\.cache\huggingface\datasets\Salesforce___wikitext") {
    $env:HF_DATASETS_OFFLINE = "1"
    $env:HF_HUB_OFFLINE      = "1"
    Info "Hugging Face cache present; HF_DATASETS_OFFLINE=1 set (avoids HF API)"
}

Section "Running full Vered grid (Stages A -> B -> C)"
& python benchmark\vered_full_grid.py
$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -eq 0) {
    Info "Grid complete. Results in benchmark\results\vered_grid_*.json"
} else {
    Fail "Grid exited with code $exitCode"
}
