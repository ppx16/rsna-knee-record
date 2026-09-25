# Download all 8 shards of the 64-slot corpus locally, then publish them as ONE Kaggle dataset.
#
# WHY: kernel-output URLs (www.kaggleusercontent.com) are the wrong transport for a rented box --
# measured 16 parallel streams = 0.89 MB/s from AutoDL versus 8.60 MB/s from this machine.  A Kaggle
# DATASET is fetched from storage.googleapis.com, which the rented box reached at 14.9 MB/s for the
# 44-slot corpus.  So: pull once here, publish once, and every future machine (and every Kaggle
# kernel) gets it over the fast path.
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root 'corpus64\shards'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$log = Join-Path $root 'state\corpus64_download.log'

function Say($m) {
    $line = "{0} {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

Say "=== downloading 8 shards -> $dest ==="
foreach ($i in 0..7) {
    $tsv = Join-Path $root (".testtmp\shard$i.tsv")
    if (-not (Test-Path $tsv)) { Say "s${i}: no url file, SKIP"; continue }
    foreach ($line in Get-Content $tsv) {
        $parts = $line -split "`t"
        $name = $parts[0]; $url = $parts[1]
        if ($name -like '*.pt') { continue }
        $out = Join-Path $dest $name
        # No skip logic here on purpose: kaggle_parallel.py keeps its own `<file>.parts` sidecar and
        # resumes from it.  Guessing a "already complete" test from the outside is how you either
        # re-download 4 GB or silently keep a truncated file.
        Say "s${i} $name"
        # 16 streams + reopen every 8 MB: measured 8.60 MB/s aggregate on this machine
        & python (Join-Path $root 'scripts\kaggle_parallel.py') --url $url --out $out `
            --streams 16 --rotate-mb 8 --rotate-sec 12 2>&1 | Select-Object -Last 2 | ForEach-Object { Say "    $_" }
    }
}
Say "=== sizes ==="
Get-ChildItem $dest -File | Where-Object { $_.Name -notlike '*.sidecar' } |
    Sort-Object Name | ForEach-Object { Say ("  {0,-20} {1,10:N1} MB" -f $_.Name, ($_.Length / 1MB)) }
$tot = (Get-ChildItem $dest -File | Where-Object { $_.Name -notlike '*.sidecar' } |
        Measure-Object -Property Length -Sum).Sum
Say ("TOTAL {0:N2} GB" -f ($tot / 1GB))
Say "CORPUS64-DOWNLOAD-DONE"
