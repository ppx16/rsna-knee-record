"""Forensics on `report_labels_v5.csv`, which scores a PERFECT 1.0000 on gold-58.

WHY
    `scripts/label_source_audit.py` scored seven local label files against the 58 officially
    labelled studies.  Six landed between 0.835 and 0.899 (v4hybrid's known 0.8991 was the best).
    The seventh, `report_labels_v5.csv`, scored **1.0000 on all twelve labels**.

    A report-derived extractor does not get 58 studies x 12 findings perfectly right.  A perfect
    score means the file CONTAINS the official labels for those rows.  Two very different
    explanations, with opposite consequences:

      (a) v5 = v4hybrid everywhere, but with the official labels written in for the 58 gold rows.
          Then v5 is a strictly better TRAINING target (real labels where they exist) and it is
          useless as an EVALUATION baseline on gold (it is the answer key there).
      (b) v5 differs from v4hybrid on the other 4349 rows too.
          Then the perfect gold score says nothing about v5's honesty on those rows, and the
          1.0000 has to be explained some other way before anything is trained on it.

    This script decides between (a) and (b) -- and it is a script rather than a here-string
    because the FIRST attempt at it died exactly the way HANDOFF 74.4 #7 records: the shell ate
    the quotes in "Baker's" and produced a SyntaxError in generated source.

Usage:
    python scripts/label_v5_forensics.py
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
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main() -> int:
    base = ROOT / "downloads" / "labels"
    v5p = next(base.rglob("report_labels_v5.csv"), None)
    v4p = next(base.rglob("report_labels_v4hybrid.csv"), None)
    if v5p is None or v4p is None:
        print("need both report_labels_v5.csv and report_labels_v4hybrid.csv under %s" % base)
        return 1
    print("v5: %s" % v5p.relative_to(ROOT))
    print("v4: %s" % v4p.relative_to(ROOT))

    v5 = pd.read_csv(v5p).set_index("StudyInstanceUID")
    v4 = pd.read_csv(v4p).set_index("StudyInstanceUID")
    print("\nv5 shape %s   v4 shape %s" % (v5.shape, v4.shape))

    common = v5.index.intersection(v4.index)
    print("common study ids: %d" % len(common))
    A = v5.loc[common, LABELS].to_numpy(float)
    B = v4.loc[common, LABELS].to_numpy(float)

    same_cell = np.isclose(A, B, atol=1e-9)
    same_row = same_cell.all(1)
    print("cells identical : %.4f" % same_cell.mean())
    print("rows identical  : %d / %d" % (int(same_row.sum()), len(common)))

    tr = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    gold = set(tr.loc[tr[LABELS].notna().all(axis=1), "StudyInstanceUID"].astype(str))
    isgold = np.array([str(i) in gold for i in common])
    print("\ngold rows in common: %d" % int(isgold.sum()))
    print("  identical among gold    : %d / %d" % (int(same_row[isgold].sum()), int(isgold.sum())))
    print("  identical among non-gold: %d / %d"
          % (int(same_row[~isgold].sum()), int((~isgold).sum())))

    nong = ~isgold
    if nong.sum():
        diff_rows = nong & ~same_row
        print("  non-gold rows that DIFFER: %d" % int(diff_rows.sum()))
        if diff_rows.sum():
            d = np.abs(A[diff_rows] - B[diff_rows])
            print("  mean |diff| there: %.4f   max %.3f" % (d.mean(), d.max()))

    print("\nv5 value set on gold rows : %s" % np.unique(A[isgold])[:10])
    print("v4 value set on gold rows : %s" % np.unique(B[isgold])[:10])
    print("v5 value set on non-gold  : %s ... (%d distinct)"
          % (np.unique(A[nong])[:6], len(np.unique(A[nong]))))

    # does v5 reproduce train.csv's official values exactly on the gold rows?
    g = tr.set_index("StudyInstanceUID").loc[common[isgold], LABELS].to_numpy(float)
    off = np.isclose(A[isgold], g, atol=1e-9)
    print("\nv5 == train.csv official on gold rows: %.4f of cells (%d/%d rows exact)"
          % (off.mean(), int(off.all(1).sum()), int(isgold.sum())))

    print("\n== VERDICT ==")
    if same_row[nong].all() if nong.sum() else True:
        print("  (a) v5 differs from v4hybrid ONLY on the 58 gold rows, and it reproduces the")
        print("      official labels there exactly.")
        print("      => v5 is 'v4hybrid + the answer key for 58 studies'.")
        print("      => usable as a TRAINING target (58 real labels instead of derived ones),")
        print("         but it must NEVER be used to score itself, and any gold-58 comparison")
        print("         against v5 is meaningless.")
    else:
        print("  (b) v5 ALSO differs from v4hybrid on non-gold rows.")
        print("      => the perfect gold-58 score does not tell us v5 is honest elsewhere;")
        print("         do NOT train on it until the difference on the other 4349 rows is")
        print("         explained.  Report this rather than acting on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
