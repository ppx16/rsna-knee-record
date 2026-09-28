"""Harvest `rsna_optimization` ledgers from every notebook we have pulled.

WHY (FACTS A167 / HANDOFF 223)
    mattiaangeli's notebooks carry an `rsna_optimization` block in the NOTEBOOK METADATA: one entry
    per version naming the single predictive change.  Reading it told us his current public revision
    is v38-radcal-off-credit-trim whose only_predictive_change is "Rad 88-feature calibration blend
    0.40 to 0.00" -- and that his public build STILL ships the _RAD_EXCLUDE bug we fixed.

    Forks inherit notebook metadata, so the same block may exist in dozens of pulled notebooks and
    amount to a crowd-sourced changelog of what actually moved the score.  That is the cheapest
    possible route to the 0.946 the frontier's team is reported to hold.

    ⚠️ `kernels_pull` is NOT version-pinned (A167): two pulls of the same notebook minutes apart gave
    141,800 B / v38-... and 173,364 B / v66-....  So every ledger printed here is stamped with the
    file size and mtime it came from; a ledger without that stamp is not evidence.

Usage:
    python scripts/harvest_ledgers.py
    python scripts/harvest_ledgers.py --glob '.testtmp/**/*.ipynb'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_GLOB = "downloads/**/*.ipynb"
EXTRA_GLOB = ".testtmp/**/*.ipynb"
KEYS = ("revision", "only_predictive_change", "official_source_score", "raptor_arm_weights",
        "inner_raptor_coat_weights", "inner_rerank", "new_models", "parent_kernel_version",
        "source", "release_guard", "prediction_identical_cleanup")


def ledgers(path):
    try:
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    return d.get("metadata", {}).get("rsna_optimization")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="")
    ap.add_argument("--all", action="store_true", help="search downloads/ and .testtmp/ together")
    a = ap.parse_args(argv)
    pats = [a.glob] if a.glob else ([DEFAULT_GLOB, EXTRA_GLOB] if a.all
                                    else [DEFAULT_GLOB, EXTRA_GLOB])
    files = []
    for p in pats:
        files += glob.glob(str(ROOT / p), recursive=True)
    files = sorted(set(files))
    found = 0
    for f in files:
        opt = ledgers(f)
        if not opt:
            continue
        found += 1
        st = os.stat(f)
        print("=" * 96)
        print("%s   [%d B, mtime %s]" % (os.path.relpath(f, ROOT), st.st_size,
                                         __import__("datetime").datetime.fromtimestamp(
                                             st.st_mtime).strftime("%Y-%m-%d %H:%M")))
        for k in KEYS:
            if k in opt:
                v = opt[k]
                print("    %-28s %s" % (k, (json.dumps(v, ensure_ascii=False)
                                            if not isinstance(v, str) else v)))
        other = [k for k in opt if k not in KEYS]
        if other:
            print("    %-28s %s" % ("(other keys)", other))
    print("\nscanned %d notebooks, %d carry an rsna_optimization ledger" % (len(files), found))
    if not found:
        print("⚠️ none found -- the forge may have stripped metadata, or nothing is pulled yet")
    print("⚠️ every ledger above is stamped with its file size and mtime because `kernels_pull` is")
    print("   NOT version-pinned (A167); an unstamped ledger is not evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
