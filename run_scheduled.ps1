<#
.SYNOPSIS
    Runs the entropy loop unattended, e.g. from Windows Task Scheduler.

.DESCRIPTION
    Reads the Gemini cookies from a JSON file outside the repository,
    exports them as environment variables, and runs
    gemini_uncached_loop.py with a rotating log file and a persistent
    daily request cap.

    Cookie file format (create it yourself, never commit it):
        {"__Secure-1PSID": "...", "__Secure-1PSIDTS": "..."}

    Example Task Scheduler registration (run once in an elevated prompt):
        schtasks /Create /TN "GeminiEntropy" /SC DAILY /ST 09:00 /TR "powershell -NoProfile -ExecutionPolicy Bypass -File \"C:\Users\sufia\Documents\Code\prompterrrrrrrrr\run_scheduled.ps1\""

.PARAMETER CookieFile
    Path to the JSON cookie file (default: %USERPROFILE%\.gemini-cookies.json).

.PARAMETER Count
    Number of requests this run sends (default: 10).

.PARAMETER Interval
    Seconds between requests (default: 60).

.PARAMETER MaxPerDay
    Persistent daily request cap shared across runs (default: 100).

.PARAMETER Level
    Entropy preset: light, standard, or paranoid (default: standard).

.PARAMETER LogFile
    Rotating JSONL log destination
    (default: %LOCALAPPDATA%\gemini-entropy\requests.jsonl).

.PARAMETER Temporary
    Send each request as a temporary chat (not saved to Gemini history).
#>
param(
    [string]$CookieFile = (Join-Path $env:USERPROFILE ".gemini-cookies.json"),
    [string]$ProjectDir = $PSScriptRoot,
    [int]$Count = 10,
    [double]$Interval = 60,
    [int]$MaxPerDay = 100,
    [ValidateSet("light", "standard", "paranoid")]
    [string]$Level = "standard",
    [string]$LogFile = (Join-Path $env:LOCALAPPDATA "gemini-entropy\requests.jsonl"),
    [string]$Python = "python",
    [switch]$Temporary
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $CookieFile)) {
    Write-Error "Cookie file not found: $CookieFile"
}
$cookies = Get-Content -Raw -LiteralPath $CookieFile | ConvertFrom-Json
if (-not $cookies.'__Secure-1PSID') {
    Write-Error "Cookie file has no __Secure-1PSID entry: $CookieFile"
}
$env:GEMINI_SECURE_1PSID = $cookies.'__Secure-1PSID'
$env:GEMINI_SECURE_1PSIDTS = $cookies.'__Secure-1PSIDTS'

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogFile) | Out-Null

$loopArgs = @(
    (Join-Path $ProjectDir "gemini_uncached_loop.py"),
    "--level", $Level,
    "--count", $Count,
    "--interval", $Interval,
    "--max-per-day", $MaxPerDay,
    "--log-file", $LogFile
)
if ($Temporary) {
    $loopArgs += "--temporary"
}

& $Python @loopArgs
exit $LASTEXITCODE
