# -*- coding: utf-8 -*-
"""Zip the CODE (+ fold evidence) of this repo into one file worth carrying to a rented box.

The corpus moves as Kaggle downloads (see `corpus_fetch.py`); this is the other half of a
migration -- the instrument itself.  `scripts/**` is under 1 MB, so there is no reason to
carry `outputs/` (12.7 GB of kernel by-products) or `downloads/` (9.4 GB of caches).  What
IS worth carrying is the evidence: every `raptor_oof_*.npz` / `raptor_gold_*.npz` and the
`raptor_ft_*.json` headers, because that is what makes a rerun comparable to our CV.

    python scripts/make_pack.py                 # -> ../corpus/packs/rsna-knee-code-<date>.zip
    python scripts/make_pack.py --no-evidence   # code only, ~6 MB
"""
import argparse
import datetime as dt
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT.parent / "corpus" / "packs"
MAX_FILE = 32 << 20          # nothing in the instrument is this big; bigger = a by-product
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".testtmp", "outputs", "downloads",
             "ktpu_out", "ktpu_out3", "data", "weights"}
EVIDENCE_GLOBS = ["downloads/**/raptor_oof_*.npz", "downloads/**/raptor_gold_*.npz",
                  "downloads/**/raptor_ft_*.json", "downloads/**/labels.parquet"]


def collect(with_evidence: bool):
    keep, skipped = [], []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        parts = set(rel.parts)
        if parts & SKIP_DIRS:
            continue
        if p.suffix in {".zip", ".pt", ".npy", ".log"}:
            skipped.append((rel, p.stat().st_size))
            continue
        if p.stat().st_size > MAX_FILE:
            skipped.append((rel, p.stat().st_size))
            continue
        keep.append(rel)
    if with_evidence:
        for g in EVIDENCE_GLOBS:
            for p in sorted(ROOT.glob(g)):
                rel = p.relative_to(ROOT)
                if rel not in keep:
                    keep.append(rel)
    return sorted(set(keep)), skipped


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--no-evidence", action="store_true")
    a = ap.parse_args()
    keep, skipped = collect(not a.no_evidence)
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / ("rsna-knee-code-%s.zip" % dt.datetime.now().strftime("%Y%m%d"))
    total = 0
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in keep:
            z.write(ROOT / rel, arcname=str(rel).replace("\\", "/"))
            total += (ROOT / rel).stat().st_size
    print("== %s" % dst)
    print("   %d files, %.1f MB raw -> %.1f MB zipped" % (len(keep), total / 1e6, dst.stat().st_size / 1e6))
    print("\n   on the rented box:  unzip -q %s && python3 scripts/corpus_manifest.py verify --root <corpus>" % dst.name)
    print("\n   left behind (by design, not needed to rerun a fold):")
    for rel, n in sorted(skipped, key=lambda x: -x[1])[:12]:
        print("     %-58s %8.2f MB" % (rel, n / 1e6))
    if len(skipped) > 12:
        print("     ... and %d more" % (len(skipped) - 12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
