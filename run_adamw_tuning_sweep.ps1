[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$VenvActivate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
if (-not $env:VIRTUAL_ENV) { & $VenvActivate }
if (Test-Path "$env:USERPROFILE\.cache\huggingface\datasets\Salesforce___wikitext") {
    $env:HF_DATASETS_OFFLINE = "1"; $env:HF_HUB_OFFLINE = "1"
}
& python benchmark\adamw_tuning_sweep.py
