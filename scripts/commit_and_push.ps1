param(
    [string]$Message = ""
)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Suppress git line-ending warnings (LF/CRLF) - cosmetic only, not errors
$env:GIT_CONFIG_NOSYSTEM = "1"
git config core.autocrlf false 2>&1 | Out-Null

Write-Host ""
Write-Host "=== OlsveredKFAC Git Commit and Push ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Clear stale index.lock
$lockFile = ".git\index.lock"
if (Test-Path $lockFile) {
    Write-Host "[1/4] Removing stale index.lock..." -ForegroundColor Yellow
    Remove-Item $lockFile -Force
    Write-Host "      Removed." -ForegroundColor Green
} else {
    Write-Host "[1/4] No index.lock found - OK." -ForegroundColor Green
}

# Step 2: Stage files
Write-Host ""
Write-Host "[2/4] Staging files..." -ForegroundColor Yellow

$filesToAdd = @(
    # Core code
    "optimizer\hooks.py",
    "optimizer\gram_estimator.py",
    "optimizer\raw_activation_hooks.py",
    "optimizer\sgso.py",
    "optimizer\classic_kfac.py",
    "optimizer\vered_kfac.py",
    "optimizer\bf16_linalg.py",                  # NEW: hand-rolled bf16 linalg
    "optimizer\olssm_kfac.py",
    "optimizer\layer_retrainer.py",
    "optimizer\backend.py",
    "optimizer\__init__.py",
    "optimizer\errors.py",

    # Benchmarks
    "benchmark",                                  # whole directory, .py + RESULTS.md
    "benchmark\results",                          # JSON outputs + PNG figures

    # Tests
    "tests\test_kappa_scaling.py",                # polished kappa-sweep
    "tests\test_bf16_linalg_support.py",          # NEW: PyTorch bf16 probe
    "tests\test_vered_kfac.py",
    "tests\test_hooks.py",

    # Paper
    "paper.md",                                   # main paper draft
    "InvFreeKFAC_paper.docx",                     # rendered docx

    # PS1 launchers
    "run_autoencoder_mnist.ps1",
    "run_autoencoder_mnist_screen.ps1",
    "run_autoencoder_mnist_screen2.ps1",
    "run_autoencoder_mnist_screen3.ps1",
    "run_autoencoder_mnist_adamw_screen.ps1",
    "run_autoencoder_mnist_adamw_screen2.ps1",
    "run_autoencoder_mnist_adamw_bf16_screen.ps1",
    "run_pinn_burgers.ps1",
    "run_pinn_burgers_tune_and_bench.ps1",
    "run_pinn_verify_capture.ps1",
    "run_kappa_sweep_polish.ps1",
    "run_bf16_linalg_probe.ps1",
    "run_bf16_linalg_test.ps1",
    "run_plot_ae_walltime_loss.ps1",
    "run_plot_best_walltime_ppl.ps1",
    "run_adamw_tuning_sweep.ps1",
    "run_kfac_wd_tuning_sweep.ps1",

    # Scripts + meta
    "scripts",
    "requirements.txt",
    "requirements-cpu.txt",
    "pyproject.toml",
    ".gitignore"
)

foreach ($f in $filesToAdd) {
    if (Test-Path $f) {
        git add $f 2>&1 | Where-Object { $_ -notmatch "warning:" } | Out-Null
        Write-Host ("      + " + $f) -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "      Staged:" -ForegroundColor Cyan
git diff --cached --name-only | ForEach-Object { Write-Host ("        " + $_) }

$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host ""
    Write-Host "  Nothing to commit - working tree clean." -ForegroundColor Yellow
    exit 0
}

# Step 3: Commit
Write-Host ""
Write-Host "[3/4] Committing..." -ForegroundColor Yellow

if ($Message -eq "") {
    $changedFiles = git diff --cached --name-only
    $areas = @()
    if ($changedFiles -match "benchmark") { $areas += "benchmark" }
    if ($changedFiles -match "optimizer") { $areas += "optimizer" }
    if ($changedFiles -match "scripts")   { $areas += "scripts" }
    if ($changedFiles -match "run_benchmark") { $areas += "run_benchmark" }
    $areaStr = if ($areas) { $areas -join ", " } else { "misc" }
    $date = Get-Date -Format "yyyy-MM-dd"
    $Message = "Update " + $areaStr + " - " + $date
}

git commit -m $Message
Write-Host ("      Committed: " + $Message) -ForegroundColor Green

# Step 4: Push
Write-Host ""
Write-Host "[4/4] Pushing to origin/main..." -ForegroundColor Yellow
git push origin main

Write-Host ""
Write-Host "=== Done! All changes pushed. ===" -ForegroundColor Green
Write-Host ""
