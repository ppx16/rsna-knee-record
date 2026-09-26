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
    [int]$MaxHours       = 11,
    # Re-evaluate the log already on disk, without touching Kaggle.  This exists so the gate can be
    # shown to PASS on a known-good run -- the positive control FACTS A79 says is mandatory, and
    # which this gate did not have the first two times it was used (it reported FAIL on two runs
    # that had in fact finished with 20/20 fingerprints).
    [switch]$EvalOnly
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

if ($EvalOnly) {
    Write-Output "=== -EvalOnly: re-evaluating $txt (no Kaggle calls) ==="
    # Re-decode FIRST (from the raw json), then check existence -- the other order makes this mode
    # useless for exactly the case it was built for: a fresh, not-yet-converted log.
    if (Test-Path $json) { python (Join-Path $repo 'scripts\klog.py') $json 2>&1 | Out-File -Encoding utf8 $txt }
    if (-not (Test-Path $txt)) { Write-Output "!! neither $txt nor a decodable $json is present"; exit 2 }
} else {

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

}   # end of the -EvalOnly guard: everything above needs Kaggle; everything below needs only the log

# ---- acceptance test (RUNBOOK-0943.md section 3) ------------------------------
# Two corrections after this gate lied twice on 2026-09-26 (FACTS A79):
#   * `dropped` is NOT a usable gate.  The notebook prints the explanatory sentence "a dropped
#     study is not a wrong label -- it is no label", so a perfect run scores 1.  It cost us two
#     false FAILs; keep it as an observation only.
#   * The decoder must be sanity-checked BEFORE its output is believed.  When klog.py could not
#     parse the log it emitted 5 bytes, which looks exactly like "the pipeline never ran".
#     A few-byte text file means the DECODER failed, not the run.
$log = Get-Content $txt -Raw
$bytes = (Get-Item $txt).Length
$fp    = ([regex]::Matches($log, 'fingerprint matches within')).Count
$gate  = ([regex]::Matches($log, 'DINOv2 fail-closed gate PASS')).Count
$bad   = ([regex]::Matches($log, 'no fallback permitted|WeightsError|Traceback')).Count
$sub   = ([regex]::Matches($log, 'COMPLETE: /kaggle/working/submission.csv')).Count
$drop  = ([regex]::Matches($log, 'dropped')).Count
$preset = ([regex]::Match($log, '\[config\] preset=\S+ a5_w=\S+ rad_alpha=\S+ k_eval=\S+')).Value

Write-Output ""
Write-Output "----------- acceptance -----------"
Write-Output ("decoded log bytes          : {0}   (healthy ~16000; a few bytes = DECODER FAILURE)" -f $bytes)
Write-Output ("effective config           : {0}" -f $preset)
Write-Output ("fingerprint matches within : {0}   (want >= 20)" -f $fp)
Write-Output ("fail-closed gate PASS      : {0}   (want 1)"    -f $gate)
Write-Output ("errors / no-fallback       : {0}   (want 0)"    -f $bad)
Write-Output ("submission.csv COMPLETE    : {0}   (want 1)"    -f $sub)
Write-Output ("dropped (OBSERVE ONLY)     : {0}   (1 is normal -- explanatory prose)" -f $drop)

if ($bytes -lt 2000) {
    Write-Output "VERDICT: DECODER FAILURE - the log did not convert; this says NOTHING about the run"
    Write-Output ("log: {0}" -f $txt)
    exit 3
}
$ok = ($fp -ge 20) -and ($gate -eq 1) -and ($bad -eq 0) -and ($sub -eq 1)
Write-Output ("VERDICT: {0}" -f $(if ($ok) { 'ACCEPTANCE PASS - safe to submit' } else { 'ACCEPTANCE FAIL - do NOT submit; read the log' }))
Write-Output ("log: {0}" -f $txt)
