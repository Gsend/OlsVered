[CmdletBinding()]
param(
    [switch]$Smoke
)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$VenvActivate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
if (-not $env:VIRTUAL_ENV) { & $VenvActivate }
if ($Smoke) {
    & python benchmark\pinn_burgers.py --smoke
} else {
    & python benchmark\pinn_burgers.py
}
