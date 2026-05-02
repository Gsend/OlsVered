# =============================================================================
#  diagnose_shutdown.ps1  -  Pull Windows event-log signals for a sudden shutdown
# =============================================================================
#
#  Run from any PowerShell (no admin needed for most queries; some sources
#  require elevation - the script will say so if it can't read them).
#
#  Usage:
#    .\diagnose_shutdown.ps1                # last 12 hours
#    .\diagnose_shutdown.ps1 -Hours 24      # last 24 hours
#    .\diagnose_shutdown.ps1 -Hours 48 -OutFile shutdown.txt
#
#  Reports on:
#    - Shutdown / restart events (1074, 6005, 6006, 6008, 41, 1076)
#    - Critical/Error events in System log
#    - Bug check (BSOD) events (1001)
#    - Thermal events (Kernel-Processor-Power, ACPI thermal)
#    - Memory pressure / OOM events (Resource-Exhaustion-Detector)
#    - Windows Update forced reboots (USO core)
#    - Application crashes (Application Error 1000)
# =============================================================================

[CmdletBinding()]
param(
    [int]$Hours = 12,
    [string]$OutFile = ""
)

$ErrorActionPreference = "Continue"
$start = (Get-Date).AddHours(-$Hours)

function Section($title) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function FormatEvent($e) {
    $msg = $e.Message
    if ($msg -and $msg.Length -gt 400) { $msg = $msg.Substring(0, 400) + "..." }
    [PSCustomObject]@{
        Time     = $e.TimeCreated
        Id       = $e.Id
        Level    = $e.LevelDisplayName
        Provider = $e.ProviderName
        Message  = $msg
    }
}

# Capture all output to a transcript if requested
if ($OutFile) {
    Start-Transcript -Path $OutFile -Force | Out-Null
}

Write-Host "Looking for events since: $start" -ForegroundColor Yellow

# ---------------------------------------------------------------------------
# 1. Shutdown / restart events
# ---------------------------------------------------------------------------
Section "Shutdown / restart events"
# 41   = Kernel-Power: system rebooted without clean shutdown (power loss / hang / BSOD)
# 1074 = User32:        application or user initiated shutdown
# 1076 = User32:        reason for last shutdown
# 6005 = EventLog:      log service started (system boot)
# 6006 = EventLog:      log service stopped (clean shutdown)
# 6008 = EventLog:      previous shutdown was unexpected
$ids = 41, 1074, 1076, 6005, 6006, 6008
try {
    Get-WinEvent -FilterHashtable @{
        LogName   = "System"
        Id        = $ids
        StartTime = $start
    } -ErrorAction Stop |
    Sort-Object TimeCreated |
    ForEach-Object { FormatEvent $_ } |
    Format-Table Time, Id, Level, Provider, Message -Wrap -AutoSize
} catch {
    Write-Host "  None found in window." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# 2. Bug check (BSOD)
# ---------------------------------------------------------------------------
Section "Bug check (BSOD) events  -  EventLog 1001 from BugCheck source"
try {
    Get-WinEvent -FilterHashtable @{
        LogName      = "System"
        ProviderName = "Microsoft-Windows-WER-SystemErrorReporting", "BugCheck"
        Id           = 1001, 1003
        StartTime    = $start
    } -ErrorAction Stop |
    Sort-Object TimeCreated |
    ForEach-Object { FormatEvent $_ } |
    Format-Table Time, Id, Level, Provider, Message -Wrap -AutoSize
} catch {
    Write-Host "  No BSOD events in window." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# 3. Thermal events
# ---------------------------------------------------------------------------
Section "Thermal events  -  Kernel-Processor-Power, ACPI thermal"
try {
    Get-WinEvent -FilterHashtable @{
        LogName      = "System"
        ProviderName = "Microsoft-Windows-Kernel-Processor-Power", "Microsoft-Windows-ACPI"
        StartTime    = $start
    } -ErrorAction SilentlyContinue |
    Where-Object { $_.LevelDisplayName -in "Critical","Error","Warning" -or
                   $_.Message -match "thermal|throttl" } |
    Sort-Object TimeCreated |
    Select-Object -First 20 |
    ForEach-Object { FormatEvent $_ } |
    Format-Table Time, Id, Level, Provider, Message -Wrap -AutoSize
} catch {
    Write-Host "  No thermal events." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# 4. Critical / Error events in System log (broad sweep)
# ---------------------------------------------------------------------------
Section "Other critical / error events in System log (top 30 by time)"
try {
    Get-WinEvent -FilterHashtable @{
        LogName   = "System"
        Level     = 1, 2   # Critical, Error
        StartTime = $start
    } -ErrorAction Stop |
    Sort-Object TimeCreated -Descending |
    Select-Object -First 30 |
    ForEach-Object { FormatEvent $_ } |
    Format-Table Time, Id, Level, Provider, Message -Wrap -AutoSize
} catch {
    Write-Host "  None found." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# 5. Application Error / Hang  -  Python crash, OOM kill
# ---------------------------------------------------------------------------
Section "Application Error / Hang events  -  python.exe in particular"
try {
    Get-WinEvent -FilterHashtable @{
        LogName   = "Application"
        Id        = 1000, 1001, 1002
        StartTime = $start
    } -ErrorAction Stop |
    Where-Object { $_.Message -match "python|cuda|cudnn|torch" -or $_.Id -eq 1002 } |
    Sort-Object TimeCreated |
    ForEach-Object { FormatEvent $_ } |
    Format-Table Time, Id, Level, Provider, Message -Wrap -AutoSize
} catch {
    Write-Host "  None found." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# 6. Resource-Exhaustion (out of RAM / commit limit)
# ---------------------------------------------------------------------------
Section "Resource-Exhaustion / low-memory events"
try {
    Get-WinEvent -FilterHashtable @{
        LogName      = "System"
        ProviderName = "Microsoft-Windows-Resource-Exhaustion-Detector",
                       "Microsoft-Windows-Resource-Exhaustion-Resolver"
        StartTime    = $start
    } -ErrorAction Stop |
    Sort-Object TimeCreated |
    ForEach-Object { FormatEvent $_ } |
    Format-Table Time, Id, Level, Provider, Message -Wrap -AutoSize
} catch {
    Write-Host "  None found." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# 7. Windows Update forced reboots
# ---------------------------------------------------------------------------
Section "Windows Update reboot signals"
try {
    Get-WinEvent -FilterHashtable @{
        LogName      = "System"
        ProviderName = "Microsoft-Windows-WindowsUpdateClient",
                       "Microsoft-Windows-USO Core"
        StartTime    = $start
    } -ErrorAction Stop |
    Where-Object { $_.Message -match "reboot|restart" } |
    Sort-Object TimeCreated |
    ForEach-Object { FormatEvent $_ } |
    Format-Table Time, Id, Level, Provider, Message -Wrap -AutoSize
} catch {
    Write-Host "  None found." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# 8. Battery events (laptop-specific - relevant since you're on RTX 3080 Laptop GPU)
# ---------------------------------------------------------------------------
Section "Battery / power events"
try {
    Get-WinEvent -FilterHashtable @{
        LogName      = "System"
        ProviderName = "Microsoft-Windows-Kernel-Power"
        StartTime    = $start
    } -ErrorAction Stop |
    Where-Object { $_.Message -match "battery|critical|power" -or $_.Id -in 41,42,109 } |
    Sort-Object TimeCreated |
    Select-Object -First 20 |
    ForEach-Object { FormatEvent $_ } |
    Format-Table Time, Id, Level, Provider, Message -Wrap -AutoSize
} catch {
    Write-Host "  None found." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# 9. NVIDIA driver events (display / nvlddmkm)
# ---------------------------------------------------------------------------
Section "NVIDIA / display driver events"
try {
    Get-WinEvent -FilterHashtable @{
        LogName   = "System"
        StartTime = $start
    } -ErrorAction SilentlyContinue |
    Where-Object { $_.ProviderName -match "nvlddmkm|nvidia|Display" -or
                   $_.Message -match "nvlddmkm|nvidia driver" } |
    Sort-Object TimeCreated |
    Select-Object -First 20 |
    ForEach-Object { FormatEvent $_ } |
    Format-Table Time, Id, Level, Provider, Message -Wrap -AutoSize
} catch {
    Write-Host "  None found." -ForegroundColor DarkGray
}

if ($OutFile) {
    Stop-Transcript | Out-Null
    Write-Host ""
    Write-Host "Saved transcript to $OutFile" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done. Most diagnostic value is usually in:" -ForegroundColor Yellow
Write-Host "  - Section 1 (Shutdown/restart): Event 41 = unclean shutdown, 1074 = who initiated" -ForegroundColor Gray
Write-Host "  - Section 2 (BugCheck):         BSOD details with stop-code" -ForegroundColor Gray
Write-Host "  - Section 6 (Resource exh.):    OOM / commit-limit hit" -ForegroundColor Gray
Write-Host "  - Section 8 (Battery):          if on a laptop and unplugged" -ForegroundColor Gray
