# Wait for ONE Kaggle kernel to reach a terminal state, at a deliberately LOW poll rate.
#
# The point is only to WAKE ME when it finishes -- so the interval is measured in minutes and the
# number of API calls is tiny (a 20-min CPU job costs ~4 read calls).  Keeping this cheap matters:
# the 2026-09-23 `SaveKernel` 403 was a write-rate limit, and the user has asked twice for lower
# polling frequency.
#
#   & scripts\watch_kernel.ps1 -Ref ppxl16/rsna-knee-corpus64-verify -MaxMinutes 90
param(
    [Parameter(Mandatory = $true)][string]$Ref,
    [int]$MaxMinutes = 90,
    [int]$PollMinutes = 5,
    [string]$DownloadTo = ""      # NOT a switch: it takes a path.  Declaring it `[switch]` made
                                  # `-DownloadTo <dir>` a parameter-binding error, which is how the
                                  # first run of this watcher died instead of watching anything.
)
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root 'state\watch_kernel.log'

function Say($m) {
    $line = "{0} {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

function StatusOf($r) {
    $o = & kaggle kernels status $r 2>&1 | Out-String
    if ($o -match 'KernelWorkerStatus\.(\w+)') { return $Matches[1] }
    return "UNKNOWN"
}

$deadline = (Get-Date).AddMinutes($MaxMinutes)
Say "watching $Ref (poll every ${PollMinutes} min, max ${MaxMinutes} min)"
$state = "timeout"
while ((Get-Date) -lt $deadline) {
    $s = StatusOf $Ref
    if ($s -in @('COMPLETE', 'ERROR', 'CANCELLED')) { $state = $s; break }
    Start-Sleep -Seconds ($PollMinutes * 60)
}
Say "$Ref -> $state"

if ($state -eq 'COMPLETE' -and $DownloadTo) {
    Say "downloading output to $DownloadTo"
    New-Item -ItemType Directory -Force -Path $DownloadTo | Out-Null
    $o = & kaggle kernels output $Ref -p $DownloadTo 2>&1 | Out-String
    Add-Content -Path $log -Value $o -Encoding UTF8
    Get-ChildItem $DownloadTo -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 12 |
        ForEach-Object { Say ("  {0,-40} {1,10:N1} MB" -f $_.Name, ($_.Length / 1MB)) }
}
Say "WATCH-DONE ($state)"
