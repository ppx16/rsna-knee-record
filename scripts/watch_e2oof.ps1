# Watch the E2-OOF kernel and report ITS acceptance criteria -- deliberately not the 0943 gate.
#
# The 0943 gate asks for "20/20 fingerprints + submission.csv", which is the wrong question here:
# this kernel's job is to (a) select a fold of TRAINING studies, (b) finish the E2 stage on them,
# and (c) leave per-member predictions on disk.  Judging it with the submission gate would print a
# confident FAIL for a run that did exactly what it was asked (the same class of mistake as A79).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\watch_e2oof.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\watch_e2oof.ps1 -Kernel ppxl16/rsna-knee-e2oof -Tag e2oof

param(
    [string]$Kernel   = 'ppxl16/rsna-knee-e2oof-smoke',
    [string]$Tag      = 'e2oof',
    [int]$IntervalMin = 2,
    [int]$MaxHours    = 4
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8       = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:KAGGLE_API_TOKEN = [Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN', 'User')

$repo = 'D:\kaggle\rsna-knee'
$kg   = 'D:\python\Scripts\kaggle.exe'
$json = Join-Path $repo "state\logs_$Tag.json"
$txt  = Join-Path $repo "state\logs_$Tag.txt"

$deadline = (Get-Date).AddHours($MaxHours)
$status = 'UNKNOWN'
$polls = 0
while ((Get-Date) -lt $deadline) {
    $polls++
    $raw = & $kg kernels status $Kernel 2>&1 | Out-String
    $status = 'UNKNOWN'
    if ($raw -match 'KernelWorkerStatus\.(\w+)') { $status = $Matches[1] }
    Write-Output ("[{0}Z] poll {1,3}  status={2}" -f (Get-Date).ToUniversalTime().ToString('MM-dd HH:mm:ss'), $polls, $status)
    if ($status -notin @('RUNNING', 'QUEUED', 'UNKNOWN')) { break }
    Start-Sleep -Seconds ($IntervalMin * 60)
}
Write-Output ""
Write-Output ("=== terminal: {0} after {1} poll(s) ===" -f $status, $polls)

$got = $false
foreach ($try in 1..5) {
    & $kg kernels logs $Kernel 2>&1 | Out-File -Encoding utf8 $json
    if ((Test-Path $json) -and (Get-Item $json).Length -gt 200) { $got = $true; break }
    Start-Sleep -Seconds 15
}
if (-not $got) { Write-Output '!! could not fetch logs'; exit 2 }
python (Join-Path $repo 'scripts\klog.py') $json 2>&1 | Out-File -Encoding utf8 $txt

$log = Get-Content $txt -Raw
$bytes = (Get-Item $txt).Length
Write-Output ""
Write-Output "----------- E2-OOF acceptance -----------"
Write-Output ("decoded log bytes        : {0}  (a few bytes = decoder failure, says nothing about the run)" -f $bytes)
foreach ($pat in @("OOF: fold", "OOF: .* series over", "OOF: E2 stage covered",
                   "DINOv2 fail-closed gate PASS", "fingerprint matches within",
                   "no fallback permitted", "WeightsError", "Traceback",
                   "FileNotFoundError", "no train_series rows", "missed .* of the requested")) {
    $n = ([regex]::Matches($log, $pat)).Count
    Write-Output ("  {0,-34} {1}" -f $pat, $n)
}
Write-Output ""
Write-Output "--- the OOF selection lines, verbatim ---"
Select-String -Path $txt -Pattern 'OOF:' | ForEach-Object { '  ' + $_.Line.Trim() }
Write-Output ""
$sel   = ([regex]::Matches($log, 'OOF: fold')).Count
$cover = ([regex]::Matches($log, 'OOF: E2 stage covered')).Count
$bad   = ([regex]::Matches($log, 'Traceback|no fallback permitted|WeightsError|FileNotFoundError')).Count
if ($bytes -lt 2000) { Write-Output 'VERDICT: DECODER FAILURE - the log did not convert'; exit 3 }
Write-Output ("VERDICT: {0}" -f $(if ($sel -ge 1 -and $cover -ge 1 -and $bad -eq 0) {
        'E2 OOF RUN OK - check /kaggle/working/diagnostics for dino_*.npz' }
    else { 'E2 OOF RUN PROBLEM - read the log' }))
