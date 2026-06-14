[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$VenvActivate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
if (-not $env:VIRTUAL_ENV) { & $VenvActivate }
& python benchmark\plot_ae_walltime_loss.py
