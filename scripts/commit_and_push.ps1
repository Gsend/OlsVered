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
    "benchmark\gpu_benchmark.py",
    "benchmark\training_benchmark.py",
    "benchmark\economic_analysis.py",
    "benchmark\theoretical_analysis.py",
    "benchmark\results",
    "optimizer\hooks.py",
    "optimizer\olsvered_kfac.py",
    "optimizer\classic_kfac.py",
    "run_benchmark.sh",
    "scripts",
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
