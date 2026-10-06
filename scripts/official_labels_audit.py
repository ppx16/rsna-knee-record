"""How many OFFICIALLY labelled studies does the competition actually give us?

WHY
    The raptor training script defines gold as the train.csv rows where ALL TWELVE label
    columns are non-null, and found 58 of them (HANDOFF 82.1).  But that is a definition of
    "fully labelled", not of "labelled at all".  If train.csv also carries PARTIAL official
    labels -- rows with some findings filled in and others blank -- then there is real
    official-label signal that neither our pipeline nor (apparently) the public notebooks use,
    because every one of them trains on report-derived soft labels for everything.

    This is a 30-second local question with a potentially large answer, so it gets a script
    rather than a here-string (HANDOFF 74.4 #7 ate the quotes on the first attempt).

Usage:
    python scripts/official_labels_audit.py
"""
from __future__ import annotations


# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded
# `print("... \u26a0 ...")` raises UnicodeEncodeError at that line and kills the script MID-REPORT, while
# everything printed earlier is already on screen -- so the run looks complete. Self-contained on purpose:
# it imports `sys` itself and depends on nothing else in this tree.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream: leave it alone, prints still degrade
    pass
# ----------------------------------------------------------------------------------------------------
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "raw" / "train.csv"
FOLDS = ROOT / "dataset-cv-fold" / "fold_file.json"

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main() -> int:
    tr = pd.read_csv(TRAIN)
    print("train.csv shape: %s" % (tr.shape,))
    print("columns: %s" % list(tr.columns))
    present = [c for c in LAB if c in tr.columns]
    print("\nlabel columns found: %d / 12" % len(present))
    if not present:
        print("!! none of the expected label names are columns -- the schema is different")
        return 1
    missing = [c for c in LAB if c not in tr.columns]
    if missing:
        print("missing: %s" % missing)

    nn = tr[present].notna().sum()
    print("\n== per-label non-null counts (of %d rows) ==" % len(tr))
    for c in present:
        print("  %-18s %6d" % (c, int(nn[c])))

    k = tr[present].notna().sum(axis=1)
    print("\n== rows by number of labels present ==")
    vc = k.value_counts().sort_index()
    for kk, v in vc.items():
        print("  %2d labels : %6d rows" % (int(kk), int(v)))
    print("\n  rows with >=1 label : %d" % int((k >= 1).sum()))
    print("  rows with all 12    : %d   <- this is the 'gold-58' the training script uses"
          % int((k == 12).sum()))

    # how much of the corpus does the partial-label set cover?
    if FOLDS.exists():
        fmap = json.load(open(FOLDS))["folds"]
        ids = set(fmap)
        print("\n== overlap with the CV fold map (%d ids) ==" % len(ids))
        col = "StudyInstanceUID" if "StudyInstanceUID" in tr.columns else tr.columns[0]
        uid = tr[col].astype(str)
        print("  fold-map ids present in train.csv      : %d" % int(uid.isin(ids).sum()))
        part = uid[k >= 1]
        alll = uid[k == 12]
        print("  >=1-label rows also in the fold map    : %d" % int(part.isin(ids).sum()))
        print("  all-12 rows also in the fold map       : %d" % int(alll.isin(ids).sum()))
        print("  ⚠️ if the all-12 rows are IN the fold map, they could leak into a fold's")
        print("     training set -- the training script excludes them explicitly, but any new")
        print("     partial-label scheme must do the same.")

    print("\n== verdict ==")
    n_partial = int(((k >= 1) & (k < 12)).sum())
    if n_partial > 50:
        print("  There ARE %d partially labelled rows." % n_partial)
        print("  => A masked-loss (partially supervised) run could use REAL official labels on")
        print("     those findings, on top of the report-derived soft labels for everything else.")
        print("     This is the only wholly-unexplored label source found so far.")
    else:
        print("  Only %d partially labelled rows -> no meaningful extra signal; the official"
              % n_partial)
        print("     labels really are just the 58 fully-labelled studies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
