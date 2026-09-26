# Keep the (limited!) CPU session slots filled with corpus shards.
#
# MEASURED 2026-09-24: Kaggle allows at most **5 concurrent CPU sessions** --
#   `Maximum batch CPU session count of 5 reached.`
# -- so pushing all 8 shards at once started 5 and rejected 3.  This script polls at a low rate
# (10 min) and pushes the next pending shard whenever a slot frees, so the build runs at the
# concurrency the platform allows instead of waiting for a human to notice.
#
# NOTE this is a LOCAL script hitting the Kaggle read API a few times per hour -- not a tight loop.
#
#   & scripts\corpus_shard_queue.ps1            # shards s0..s7
param(
    [int]$Shards = 8,
    [int]$Slots = 5,
    [int]$PollMinutes = 10,
    [int]$MaxHours = 40
)
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root 'state\corpus_shard_queue.log'

function Say($m) {
    $line = "{0} {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}
function StatusOf($ref) {
    $o = & kaggle kernels status $ref 2>&1 | Out-String
    if ($o -match 'KernelWorkerStatus\.(\w+)') { return $Matches[1] }
    return "ABSENT"
}
function Pending() {
    $p = @()
    for ($i = 0; $i -lt $Shards; $i++) {
        if ((StatusOf "ppxl16/rsna-knee-corpus64-s$i") -eq 'ABSENT') { $p += $i }
    }
    return $p
}

$deadline = (Get-Date).AddHours($MaxHours)
Say "shard queue start: $Shards shards, $Slots concurrent CPU slots"
while ((Get-Date) -lt $deadline) {
    $states = @()
    $active = 0
    for ($i = 0; $i -lt $Shards; $i++) {
        $s = StatusOf "ppxl16/rsna-knee-corpus64-s$i"
        $states += "s${i}:$s"
        if ($s -in @('RUNNING', 'QUEUED')) { $active++ }
    }
    $pend = Pending
    $done = ($states | Where-Object { $_ -match ':COMPLETE$' }).Count
    Say ("active=$active done=$done pending=[{0}]" -f ($pend -join ','))
    if ($pend.Count -eq 0 -and $active -eq 0) { Say "all shards terminal"; break }

    while ($active -lt $Slots -and $pend.Count -gt 0) {
        $i = $pend[0]; $pend = $pend[1..($pend.Count - 1)]
        # kernels moved under kernels/ on 2026-09-26; fall back to the old flat layout if absent
        $dir = Join-Path $root "kernels\kernel-corpus64-s$i"
        if (-not (Test-Path $dir)) { $dir = Join-Path $root "kernel-corpus64-s$i" }
        Say "pushing s$i"
        $o = & kaggle kernels push -p $dir 2>&1 | Out-String
        Add-Content -Path $log -Value $o -Encoding UTF8
        if ($o -match 'successfully pushed') { $active++ }
        else {
            Say "  push s$i failed: $($o.Trim() -split "`n" | Select-Object -Last 1)"
            break
        }
    }
    Start-Sleep -Seconds ($PollMinutes * 60)
}
$final = @()
for ($i = 0; $i -lt $Shards; $i++) { $final += "s${i}:$(StatusOf "ppxl16/rsna-knee-corpus64-s$i")" }
Say ("queue done -> " + ($final -join ' '))
Say "SHARD-QUEUE-DONE"
