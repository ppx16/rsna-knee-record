# Watch one pushed Kaggle kernel until it reaches a terminal status, then pull the log and run the
# acceptance test from RUNBOOK-0943.md section 3.
#
# Why a single long-lived job instead of repeated manual checks: the run takes hours, and the
# acceptance test only becomes meaningful at the end.  Polling is kept at ~10 calls/hour on purpose --
# FACTS A26 records that roughly 100 calls/hour got the read paths 403-banned, so the cadence is part
# of the design, not an afterthought.
#
# Usage:
#   powershell -File scripts\watch_0943.ps1
#   powershell -File scripts\watch_0943.ps1 -Kernel ppxl16/rsna-knee-0943h -Tag 0943h

param(
    [string]$Kernel      = 'ppxl16/rsna-knee-0943',
    [string]$Tag         = '0943',
    [int]$IntervalMin    = 6,
    [int]$MaxHours       = 11
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8        = '1'   # FACTS A77: without this the CLI reads notebooks as gbk and dies
$env:PYTHONIOENCODING  = 'utf-8'
$env:KAGGLE_API_TOKEN  = [Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN', 'User')

$repo = 'D:\kaggle\rsna-knee'
$kg   = 'D:\python\Scripts\kaggle.exe'
$json = Join-Path $repo "state\logs_$Tag.json"
$txt  = Join-Path $repo "state\logs_$Tag.txt"

$deadline = (Get-Date).AddHours($MaxHours)
$status   = 'UNKNOWN'
$polls    = 0

while ((Get-Date) -lt $deadline) {
    $polls++
    $raw = & $kg kernels status $Kernel 2>&1 | Out-String
    $status = 'UNKNOWN'
    if ($raw -match 'KernelWorkerStatus\.(\w+)') { $status = $Matches[1] }
    $stamp = (Get-Date).ToUniversalTime().ToString('MM-dd HH:mm:ss')
    Write-Output ("[{0}Z] poll {1,3}  status={2}" -f $stamp, $polls, $status)
    if ($status -notin @('RUNNING', 'QUEUED', 'UNKNOWN')) { break }
    Start-Sleep -Seconds ($IntervalMin * 60)
}

Write-Output ""
Write-Output ("=== terminal: {0} after {1} poll(s) at {2}Z ===" -f `
    $status, $polls, (Get-Date).ToUniversalTime().ToString('MM-dd HH:mm:ss'))

# ---- pull the log (retry: the link flaps, FACTS A38/A67) ----------------------
$got = $false
foreach ($try in 1..5) {
    & $kg kernels logs $Kernel 2>&1 | Out-File -Encoding utf8 $json
    if ((Test-Path $json) -and (Get-Item $json).Length -gt 200) { $got = $true; break }
    Write-Output "log fetch attempt $try produced nothing; retrying"
    Start-Sleep -Seconds 20
}
if (-not $got) { Write-Output '!! could not fetch logs'; exit 2 }

python (Join-Path $repo 'scripts\klog.py') $json 2>&1 | Out-File -Encoding utf8 $txt

# ---- acceptance test (RUNBOOK-0943.md section 3) ------------------------------
$log = Get-Content $txt -Raw
$fp    = ([regex]::Matches($log, 'fingerprint matches within')).Count
$gate  = ([regex]::Matches($log, 'DINOv2 fail-closed gate PASS')).Count
$bad   = ([regex]::Matches($log, 'no fallback permitted|WeightsError|Traceback')).Count
$drop  = ([regex]::Matches($log, 'dropped')).Count
$done  = ([regex]::Matches($log, 'rsna_phase.*a5.*COMPLETE|final DINO member count')).Count

Write-Output ""
Write-Output "----------- acceptance -----------"
Write-Output ("fingerprint matches within : {0}   (want >= 20)" -f $fp)
Write-Output ("fail-closed gate PASS      : {0}   (want 1)"    -f $gate)
Write-Output ("errors / no-fallback       : {0}   (want 0)"    -f $bad)
Write-Output ("dropped                    : {0}   (want 0)"    -f $drop)
Write-Output ("a5-complete markers        : {0}"               -f $done)
$ok = ($fp -ge 20) -and ($gate -eq 1) -and ($bad -eq 0) -and ($drop -eq 0)
Write-Output ("VERDICT: {0}" -f $(if ($ok) { 'ACCEPTANCE PASS - safe to submit' } else { 'ACCEPTANCE FAIL - do NOT submit; read the log' }))
Write-Output ("log: {0}" -f $txt)
