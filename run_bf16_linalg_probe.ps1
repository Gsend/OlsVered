[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$VenvActivate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
if (-not $env:VIRTUAL_ENV) { & $VenvActivate }
& python tests\test_bf16_linalg_support.py
