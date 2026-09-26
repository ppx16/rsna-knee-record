"""Dump scanner / site / acquisition fingerprints per training SERIES, to test a fold leak locally.

WHY
---
The annotated version of the pipeline we replicate says its EDA exposes "scanner and site
fingerprints, **with the fold-grouping leak they imply**".  A fold split that scatters one site's
studies across folds lets a model learn site-specific shortcuts, so the OOF is optimistic -- and
unlike ordinary sampling noise, that bias does NOT shrink as folds are added.  FACTS A100 measured
the report-group axis (clean: 2 of 49 shared-report groups split, 4 studies) but the scanner/site
axis needs DICOM headers, which are not on our machine.

WHY A CPU KERNEL
----------------
CPU sessions consume neither GPU nor TPU quota (FACTS A62), so this costs nothing that competes
with submissions.  It reads only headers -- `stop_before_pixels=True` -- so it never touches pixels.

WHAT IT WRITES
--------------
`/kaggle/working/site_probe.csv`, one row per series, plus a JSON summary.  The analysis (does a
site span folds?) happens LOCALLY against our own fold file, where it can be inspected and re-run.

DELIBERATELY NOT DERIVED FROM THE PIPELINE NOTEBOOK: this is a standalone script kernel.  Deriving
it from the 51-cell notebook would inherit that notebook's cohort globals, its fail-closed gates and
its cache machinery for no benefit -- the four smoke iterations on the A5 kernel were each spent
fighting exactly that inheritance.
"""
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pydicom

TAGS = [
    "Manufacturer", "ManufacturerModelName", "StationName", "SoftwareVersions",
    "MagneticFieldStrength", "DeviceSerialNumber", "InstitutionName",
    "InstitutionalDepartmentName", "BodyPartExamined", "SeriesDescription",
    "ScanOptions", "SliceThickness", "Rows", "Columns", "PixelSpacing",
    "RepetitionTime", "EchoTime", "ScanningSequence", "SequenceName",
]
OUT = "/kaggle/working/site_probe.csv"
SUMMARY = "/kaggle/working/site_probe_summary.json"
THREADS = 32
# A smoke mode so the header path can be proven on a few studies before reading all 24k series.
LIMIT_STUDIES = int(os.environ.get("SITE_PROBE_STUDIES", "0") or 0)


def find_root():
    cands = [
        "/kaggle/input/rsna-knee-abnormality-detection",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    ]
    cands += sorted(
        os.path.join("/kaggle/input", d)
        for d in os.listdir("/kaggle/input")
        if os.path.isdir(os.path.join("/kaggle/input", d))
    )
    cands += sorted(
        os.path.join("/kaggle/input", a, b)
        for a in os.listdir("/kaggle/input")
        if os.path.isdir(os.path.join("/kaggle/input", a))
        for b in os.listdir(os.path.join("/kaggle/input", a))
        if os.path.isdir(os.path.join("/kaggle/input", a, b))
    )
    seen = []
    for c in cands:
        if not os.path.isdir(c):
            continue
        seen.append(c)
        if os.path.isdir(os.path.join(c, "train_series")):
            print("competition root: %s" % c, flush=True)
            return c
    print("no train_series under any of: %s" % seen[:12], flush=True)
    return None


def series_dirs(root):
    base = os.path.join(root, "train_series")
    for study in sorted(os.listdir(base)):
        sd = os.path.join(base, study)
        if not os.path.isdir(sd):
            continue
        for series in sorted(os.listdir(sd)):
            pd = os.path.join(sd, series)
            if os.path.isdir(pd):
                yield study, series, pd


def first_file(sdir):
    try:
        names = sorted(n for n in os.listdir(sdir) if n.lower().endswith(".dcm"))
    except OSError:
        return None
    return os.path.join(sdir, names[0]) if names else None


def read_one(job):
    study, series, sdir = job
    f = first_file(sdir)
    if f is None:
        return None
    try:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
    except Exception as exc:
        return {"StudyInstanceUID": study, "SeriesInstanceUID": series,
                "_error": "%s: %s" % (type(exc).__name__, exc)}
    row = {"StudyInstanceUID": study, "SeriesInstanceUID": series,
           "n_files": len([n for n in os.listdir(sdir) if n.lower().endswith(".dcm")])}
    for t in TAGS:
        v = getattr(ds, t, None)
        if isinstance(v, (list, tuple)):
            v = "|".join(str(x) for x in v)
        row[t] = "" if v is None else str(v)
    return row


def main():
    t0 = time.time()
    root = find_root()
    if root is None:
        raise SystemExit("competition data not attached")
    jobs = list(series_dirs(root))
    if LIMIT_STUDIES:
        keep = sorted(set(j[0] for j in jobs))[:LIMIT_STUDIES]
        jobs = [j for j in jobs if j[0] in set(keep)]
    print("series to probe: %d (%.1f s to enumerate)" % (len(jobs), time.time() - t0), flush=True)

    rows, done, errors = [], 0, 0
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        for row in pool.map(read_one, jobs, chunksize=64):
            done += 1
            if row is None:
                continue
            if "_error" in row:
                errors += 1
            rows.append(row)
            if done % 2000 == 0:
                el = time.time() - t0
                print("  %d/%d  %.1f min  (%.1f series/s)"
                      % (done, len(jobs), el / 60, done / max(el, 1e-9)), flush=True)

    cols = ["StudyInstanceUID", "SeriesInstanceUID", "n_files", "_error"] + TAGS
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    studies = sorted(set(r["StudyInstanceUID"] for r in rows))
    def uniq(tag):
        return sorted(set(r.get(tag, "") for r in rows if r.get(tag, "") != ""))
    summary = {
        "root": root,
        "series": len(rows),
        "studies": len(studies),
        "read_errors": errors,
        "elapsed_seconds": round(time.time() - t0, 1),
        "distinct": {t: len(uniq(t)) for t in TAGS},
        "top_manufacturers": sorted(
            ((v, sum(1 for r in rows if r.get("Manufacturer") == v))
             for v in uniq("Manufacturer")), key=lambda x: -x[1])[:10],
        "top_stations": sorted(
            ((v, sum(1 for r in rows if r.get("StationName") == v))
             for v in uniq("StationName")), key=lambda x: -x[1])[:10],
        "top_models": sorted(
            ((v, sum(1 for r in rows if r.get("ManufacturerModelName") == v))
             for v in uniq("ManufacturerModelName")), key=lambda x: -x[1])[:10],
    }
    with open(SUMMARY, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2)[:3000], flush=True)
    print("wrote %s and %s" % (OUT, SUMMARY), flush=True)


if __name__ == "__main__":
    main()
