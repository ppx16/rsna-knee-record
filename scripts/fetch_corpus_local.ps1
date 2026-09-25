# Local (Windows) side of the migration kit: stage the whole training corpus under
# D:\kaggle\corpus, using the resume-safe downloader instead of `kaggle datasets download`.
#
# WHY NOT `kaggle datasets download`: on 2026-09-24 it truncated a 7.30 GB partial back to
# 6.80 GB and restarted -- ~40 minutes of transfer lost, because a Range request answered with
# 200 makes the CLI reopen the file in "wb".  `kaggle_resume.py` only ever appends, and retries
# the flaky redirect to storage.googleapis.com (SSLEOFError) with backoff.
#
#   pwsh -File scripts\fetch_corpus_local.ps1            # resume-safe, re-runnable
#   pwsh -File scripts\fetch_corpus_local.ps1 -Info      # just report what is on disk
param([switch]$Info)

$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$here = $PSScriptRoot
$repo = Split-Path -Parent $here
$stage = Join-Path (Split-Path -Parent $repo) 'corpus'
$zips = Join-Path $stage 'zips'
New-Item -ItemType Directory -Force $zips | Out-Null

function Show {
    Get-ChildItem $zips -Force -ErrorAction SilentlyContinue | ForEach-Object {
        "{0,-42} {1,10:N3} GB" -f $_.Name, ($_.Length / 1GB)
    }
}

if ($Info) { Show; exit 0 }

# Whole-dataset zips use --bundle (that is `kaggle datasets download -d ref`: same request
# without file_name, output "<slug>.zip").  Single-file datasets are fetched by name.
$jobs = @(
    @{ ref = 'dreaddevelopment/knee-raptor-corpus';       bundle = $true },
    @{ ref = 'dreaddevelopment/knee-raptor-corpus-ext';   bundle = $true },
    @{ ref = 'flight0234/rsna-knee-hybrid-report-labels'; file = 'report_labels_v4hybrid.csv' },
    @{ ref = 'ppxl16/rsna-knee-cv-folds';                 file = 'fold_file.json' }
)
foreach ($j in $jobs) {
    Write-Host "`n=== $($j.ref)" -ForegroundColor Cyan
    if ($j.bundle) {
        & python (Join-Path $here 'kaggle_resume.py') --ref $j.ref --bundle --outdir $zips
    } else {
        & python (Join-Path $here 'kaggle_resume.py') --ref $j.ref --file $j.file --out (Join-Path $zips $j.file)
    }
}

# train.csv: only competition file we need (study list + gold-58 truth). Its signed URL is the
# same flaky storage.googleapis.com hop, so retry the CLI a few times.
if (-not (Test-Path (Join-Path $zips 'train.csv'))) {
    for ($i = 1; $i -le 8; $i++) {
        Write-Host "`n=== competition train.csv (attempt $i)" -ForegroundColor Cyan
        & kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p $zips
        if (Test-Path (Join-Path $zips 'train.csv')) { break }
        Start-Sleep -Seconds (5 * $i)
    }
}

Show
Write-Host "`nnext: python scripts\corpus_manifest.py build   (then verify)"
