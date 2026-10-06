"""Download the public reader and read its interface before wiring anything.

Step 1 of the plan in docs/STRATEGY-2026-10-06-two-new-levers.md. Zero GPU. What must be established
BEFORE building an arm:

  * the exact inference entry point and its argument order (their notebook calls
    `infer.py <data_dir> test <out.csv> <ckpt...>`);
  * what `data_dir` must contain (they pass a dir holding `test_series.csv`);
  * how long it takes, so the combined runtime can be checked against the 9 h cap -- their stack
    already needs 6.5-8 h, and overshooting the cap loses the whole submission;
  * whether it needs anything that is not mounted (pip wheels are shipped IN the dataset, so it should
    be self-contained with internet off).
"""
import pathlib

from kaggle.api.kaggle_api_extended import KaggleApi

REF = "goodpjw2008/rsna-knee-2-5d-convnext-reader"
OUT = pathlib.Path("work/reader")
OUT.mkdir(parents=True, exist_ok=True)

api = KaggleApi()
api.authenticate()
print("downloading %s ..." % REF)
api.dataset_download_files(REF, path=str(OUT), unzip=True, quiet=False)
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print("   %9.1f KB  %s" % (p.stat().st_size / 1e3, p.relative_to(OUT)))
