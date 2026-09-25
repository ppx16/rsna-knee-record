# Publish the 64-slot dense corpus as ONE Kaggle dataset.
#
# WHY (2026-09-24, user): "大概率我会用64槽全面代替44槽" -- if 64 slots replaces 44 slots wholesale,
# the dense corpus has to become a first-class asset rather than eight kernel outputs.  A dataset is
# fetched from storage.googleapis.com, which the rented box reached at 14.9 MB/s for the 44-slot
# corpus; kernel outputs (www.kaggleusercontent.com) gave it 0.89 MB/s.  So this is what makes the new
# primary corpus usable on rented GPUs during a quota drought -- and mountable in one line in any
# Kaggle kernel.
#
# LAYOUT MATTERS: per-shard SUBDIRECTORIES (rsna-knee-corpus64-s0/vols_s0.npy), because that is the
# shape the loader's SHARD_GLOBS already matches.  A flat layout would silently find nothing -- and
# with SHARD_REQUIRED set, "silently" is the part that is already handled: it fails loudly instead.
#
# Files are HARDLINKED into the staging tree, so this does not double the 31.8 GB on disk.
#
#   powershell -File scripts\publish_corpus64_dataset.ps1            # stage + report
#   powershell -File scripts\publish_corpus64_dataset.ps1 -Upload    # stage + create the dataset
param(
    [switch]$Upload,
    [string]$Slug = 'rsna-knee-corpus64-dense',
    [string]$Title = 'RSNA knee corpus64 dense'
)
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root 'corpus64\shards'
$stage = Join-Path $root 'corpus64\ds'

if (-not (Test-Path $src)) { Write-Error "no shards at $src"; exit 1 }

New-Item -ItemType Directory -Force -Path $stage | Out-Null
# FLAT staging on purpose: `kaggle datasets create --dir-mode` only offers skip/zip/tar for
# directories, so nested per-shard folders are not something to bet on.  The loader handles this flat
# shape as a first-class case (SHARD_FLAT_GLOBS), verified by build_dense_pilot.py --selftest.
$missing = @()
$staged = 0
# Sizes are EXACT and derivable, so check them instead of trusting the downloader's .parts sidecar:
# that sidecar is a progress ledger and can lag the final chunk (observed on s7: 100% displayed but
# done < total), which would have made this guard refuse a perfectly complete corpus.
$rowsFor = { param($i) if ($i -lt 7) { 551 } else { 550 } }
$expSize = {
    param($kind, $i)
    $r = & $rowsFor $i
    switch ($kind) {
        'vols'  { $r * 64 * 336 * 336 + 128 }
        'masks' { $r * 64 + 128 }
        'ids'   { $r * 256 + 128 }
    }
}
foreach ($i in 0..7) {
    foreach ($kind in 'vols', 'masks', 'ids') {
        $name = "{0}_s{1}.npy" -f $kind, $i
        $from = Join-Path $src $name
        $to = Join-Path $stage $name
        if (-not (Test-Path $from)) { $missing += "$name (absent)"; continue }
        $want = & $expSize $kind $i
        $got = (Get-Item $from).Length
        if ($got -ne $want) { $missing += ("$name ({0} of {1} bytes)" -f $got, $want); continue }
        if (Test-Path $to) { Remove-Item $to -Force }
        New-Item -ItemType HardLink -Path $to -Target $from | Out-Null
        $staged++
    }
}
Write-Host "staged files: $staged (expect 24)"
if ($missing.Count) {
    Write-Host "MISSING / INCOMPLETE -- refusing to publish:"
    $missing | ForEach-Object { Write-Host "   $_" }
    exit 1
}

$meta = @{
    title = $Title
    id    = "ppxl16/$Slug"
    # derived from competition training data with a public recipe; not our original content
    licenses = @(@{ name = 'other' })
} | ConvertTo-Json -Depth 5
# ⚠️ NOT `Set-Content -Encoding utf8`: Windows PowerShell's utf8 writer prepends a BOM (EF BB BF), and
# a BOM makes the Kaggle CLI's json.load fail with exactly `Expecting value: line 1 column 1 (char 0)`
# -- measured 2026-09-24, which is what made the first upload attempt die in 0.0 min.
[System.IO.File]::WriteAllText((Join-Path $stage 'dataset-metadata.json'), $meta,
                               (New-Object System.Text.UTF8Encoding($false)))
$first = [System.IO.File]::ReadAllBytes((Join-Path $stage 'dataset-metadata.json'))[0..2]
if ($first[0] -eq 0xEF) { Write-Error "metadata still has a BOM"; exit 1 }

$tot = (Get-ChildItem $stage -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("staged {0:N2} GB in {1} files" -f ($tot / 1GB), (Get-ChildItem $stage -Recurse -File).Count)
Write-Host "layout check (loader reads these flat via SHARD_FLAT_GLOBS):"
Get-ChildItem $stage -File | Select-Object -First 6 | ForEach-Object { Write-Host "   $($_.Name)" }

if ($Upload) {
    Write-Host "=== uploading (this is the long part) ==="
    $t0 = Get-Date
    $ulog = Join-Path $root 'state\corpus64_upload.log'
    # STREAM the output (Tee) instead of buffering it with Out-String.  Buffered, the first attempt
    # looked identical whether it was uploading or hung -- and it was in fact hung for 14 minutes on a
    # stalled request with zero CPU and zero IO.  Streamed, progress and stalls are both visible.
    & kaggle datasets create -p $stage -r skip 2>&1 | Tee-Object -FilePath $ulog -Append
    $dt = ((Get-Date) - $t0).TotalMinutes
    Write-Host ("upload call returned after {0:N1} min => {1:N2} MB/s" -f $dt, ($tot / 1MB / [Math]::Max($dt * 60, 0.01)))
} else {
    Write-Host "(staged only; re-run with -Upload to create the dataset)"
}
Write-Host "PUBLISH-DONE"
