[CmdletBinding()]
param(
    [switch]$Smoke   # screens only, skip the 18-cell multi-seed bench
)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$VenvActivate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
if (-not $env:VIRTUAL_ENV) { & $VenvActivate }
if ($Smoke) {
    & python benchmark\pinn_burgers_tune_and_bench.py --smoke
} else {
    & python benchmark\pinn_burgers_tune_and_bench.py
}
