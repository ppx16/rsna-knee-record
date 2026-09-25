# Keep the single Kaggle TPU slot FILLED, without polling every few minutes.
#
# WHY THIS IS NOT A POLL LOOP: the user watches the run and will ping me, and each status call is a
# round trip.  A 77-min fold needs ~2 checks, not ~26.  So this SLEEPS until the expected finish
# (measured fold time + margin), and only then starts checking -- at a 4-minute cadence, which for
# the last few minutes is the only window where the answer can change.
#
# The refill itself still has to be automatic: A28/A33 -- a queued kernel already occupies the only
# TPU slot, so a slot left idle between folds is wasted quota, and TPU quota is the scarce resource.
#
#   & scripts\tpu_chain.ps1 -Folds 2,3 -WaitMinutes 40
param(
    [int[]]$Folds = @(2, 3),
    [string]$Current = "ppxl16/rsna-knee-tpu-nx336f1",
    [int]$WaitMinutes = 40,          # sleep before the FIRST check (expected remaining fold time)
    [int]$PollSeconds = 240,         # cadence once we are inside the finish window
    [int]$NeedMinutesPerFold = 80,
    [int]$MarginMinutes = 20
)
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root 'state\tpu_chain.log'

function Say($m) {
    $line = "{0} {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}
function StatusOf($ref) {
    $o = & kaggle kernels status $ref 2>&1 | Out-String
    if ($o -match 'KernelWorkerStatus\.(\w+)') { return $Matches[1] }
    return "UNKNOWN"
}
function TpuHoursLeft() {
    $o = & kaggle quota 2>&1 | Out-String
    foreach ($line in ($o -split "`n")) {
        if ($line -match '^TPU\s+\S+\s+([0-9.]+)h') { return [double]$Matches[1] }
    }
    return -1
}
function WaitTerminal($ref, $firstSleepMin) {
    if ($firstSleepMin -gt 0) {
        Say "sleeping $firstSleepMin min before the first status check on $ref (a fold is ~80 min; checking sooner cannot change the answer)"
        Start-Sleep -Seconds ($firstSleepMin * 60)
    }
    while ($true) {
        $s = StatusOf $ref
        if ($s -in @('COMPLETE', 'ERROR', 'CANCELLED')) { Say "$ref -> $s"; return $s }
        Say "$ref -> $s"
        Start-Sleep -Seconds $PollSeconds
    }
}

Say "=== tpu_chain start (quiet mode); current = $Current ==="
$s = WaitTerminal $Current $WaitMinutes
if ($s -ne 'COMPLETE') { Say "!! $Current ended as $s -- stopping so the failure is visible"; Say "TPU-CHAIN-STOPPED"; exit 1 }

foreach ($f in $Folds) {
    $left = TpuHoursLeft
    Say ("TPU quota remaining: {0}h (a fold needs ~{1}h + {2}min margin)" -f $left, [math]::Round($NeedMinutesPerFold / 60.0, 2), $MarginMinutes)
    if ($left -lt 0) { Say "!! quota unreadable; stopping"; break }
    if (($left * 60) -lt ($NeedMinutesPerFold + $MarginMinutes)) { Say "== not enough TPU quota for fold $f -- stopping cleanly"; break }

    $name = "kernel-tpu-nx336f$f"; $tag = "nx336$f"
    Say "=== building + pushing fold $f ($tag / $name) ==="
    $out = & python (Join-Path $PSScriptRoot 'build_tpu_cv.py') --folds 5 --fold $f --tag $tag `
        --kernel-name $name --arch convnext_base.fb_in22k_ft_in1k --res 336 --epochs 12 --push 2>&1 | Out-String
    Add-Content -Path $log -Value $out -Encoding UTF8
    if ($out -notmatch 'successfully pushed') { Say "!! push for fold $f FAILED -- stopping`n$out"; break }

    $ref = "ppxl16/rsna-knee-tpu-nx336f$f"
    $s = WaitTerminal $ref $NeedMinutesPerFold
    if ($s -ne 'COMPLETE') { Say "!! fold $f ended as $s; stopping for inspection"; break }
    Say "fold $f COMPLETE"
}
Say ("=== tpu_chain done; TPU quota remaining: {0}h" -f (TpuHoursLeft))
Say "TPU-CHAIN-DONE"
