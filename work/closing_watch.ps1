# Hourly monitor for the closing day: log the board and the pending ARC draws, do not act.
#
# Why: on the last day the only thing that changes is the board, and the record should show its closing
# trajectory rather than whichever single reading I happened to take. It only reads -- no submissions,
# no kernel pushes (A193: Create*/Submit* are actions, not reads).
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$log = 'D:\kaggle\rsna-knee\state\closing_watch.txt'
$deadline = [datetime]::SpecifyKind([datetime]'2026-10-22T23:59:00', [DateTimeKind]::Utc)

function Say($m) {
    $line = '{0}  {1}' -f ([datetime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')), $m
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
    Write-Output $line
}

Say 'CLOSING WATCH START (hourly, read-only)'

while ($true) {
    $now = [datetime]::UtcNow
    $left = ($deadline - $now).TotalHours
    if ($left -lt 0) {
        Say ('DEADLINE PASSED {0:N1}h ago -- monitor is now only useful for the ARC' -f (-$left))
    }

    # RSNA board, from the CLI (the same source the daily check uses)
    try {
        $o = & python 'D:\kaggle\rsna-knee\work\daily_rsna.py' 2>&1
        $keep = $o | Select-String -Pattern 'us      :|bronze  :|silver  :|our band:|none above|TICK-|VERDICT'
        foreach ($l in $keep) { Say ('RSNA ' + $l.Line.Trim()) }
    } catch {
        Say ('RSNA board read failed: ' + $_.Exception.Message)
    }

    # ARC: has the pending draw settled?
    try {
        $a = & python 'D:\kaggle\arc\work\recon.py' 2>&1
        $last = $a | Select-String -Pattern 'COMPLETE|PENDING|best public score'
        foreach ($l in $last) { Say ('ARC  ' + $l.Line.Trim()) }
    } catch {
        Say ('ARC recon failed: ' + $_.Exception.Message)
    }

    Say ('-- hours to deadline: {0:N2}' -f $left)
    Start-Sleep -Seconds 3600
}
