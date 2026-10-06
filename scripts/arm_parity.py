# -*- coding: utf-8 -*-
"""Is our training line actually behind, or was it judged against the wrong reference?

WHY THIS EXISTS
    A106 closed the training line: "our best arm 0.8731 against A5's 0.9143 on the 181 studies
    where both are genuinely held out -- a gap whose CI excludes zero".  That comparison is real,
    but A5 is the STRONGEST single arm in the frontier's stack (a DINOv3 model).  The frontier's
    twenty published DINOv2 members -- the family our own arms are the same kind of thing as --
    may be nowhere near 0.914.  If they are not, then "we are 0.04 behind" is a statement about
    A5, not about our training, and the closure means something different.

    That distinction decides whether training is worth GPU hours: behind A5 by 0.04 is hopeless
    (A5 is one model we cannot match cheaply), while at parity with the DINOv2 members means the
    frontier's edge lives in the OTHER families and their fusion -- and that our own fold-trained
    members can serve as an honest instrument at the right operating point, which is the thing
    every fusion knob has been blocked on.

    Everything here is scored against ONE label set (the report-derived labels used elsewhere in
    the repo) on the studies each family has in common, so the columns are comparable.

Usage:
    python scripts/arm_parity.py
    python scripts/arm_parity.py --fold0-only     # restrict every family to fold 0, like A112
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
import argparse
import csv
import glob
import os
import pathlib
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from calibrator_probe import rank_columns, read_labels  # noqa: E402
from cv_score import LABELS  # noqa: E402

OOF = ROOT / "downloads" / "oof"
A5 = ROOT / "downloads" / "research" / "a5oof_full" / "speed_a5_raw.npz"
FOLDS = ROOT / "downloads" / "research" / "oof" / "v52_oof.csv"
LABELS_CSV = ROOT / "downloads" / "research" / "oof" / "report_labels_v2.csv"


def macro(Y, M, P):
    vals = []
    for j in range(Y.shape[1]):
        m = M[:, j]
        y = Y[m, j].astype(int)
        if len(np.unique(y)) < 2:
            continue
        vals.append(roc_auc_score(y, P[m, j]))
    return float(np.mean(vals)) if vals else float("nan")


def load_ours():
    """Base arms only (no _epN/_swa/_epens variants), keyed by 'arm@fold'."""
    out = {}
    pats = [str(ROOT / "downloads" / "cv" / "**" / "raptor_oof_*.npz"),
            str(ROOT / "downloads" / "rent" / "**" / "raptor_oof_*.npz")]
    for pat in pats:
        for p in glob.glob(pat, recursive=True):
            m = re.search(r"raptor_oof_([A-Za-z0-9]+?)(?:_(ep\d+|epens|swa))?_fold(\d+)\.npz$",
                          os.path.basename(p))
            if not m or m.group(2):
                continue
            d = np.load(p, allow_pickle=True)
            out["%s@f%s" % (m.group(1), m.group(3))] = ([str(u) for u in d["ids"]],
                                                        np.asarray(d["pred"], dtype=np.float64))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold0-only", action="store_true",
                    help="restrict every family to fold-0 studies (A112's setting)")
    a = ap.parse_args(argv)

    lab_ids, Y, M = read_labels(str(LABELS_CSV))
    pos_lab = {u: i for i, u in enumerate(lab_ids)}
    fold = {}
    with open(FOLDS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fold[r["StudyInstanceUID"]] = int(r["fold"])

    fams = {}
    for k, (ids, P) in load_ours().items():
        fams["ours " + k] = (ids, P)

    p20_ids = [str(u) for u in np.load(OOF / "train_ids.npy", allow_pickle=True)]
    p20 = np.load(OOF / "oof_arm_prob.npy").astype(np.float64)
    for j in range(p20.shape[1]):
        fams["public20 seed%d" % j] = (p20_ids, p20[:, j, :])

    d = np.load(A5, allow_pickle=True)
    a5_ids = [str(u) for u in d["study_uids"]]
    raw = np.asarray(d["raw_probabilities"], dtype=np.float64)
    a5 = np.full((len(a5_ids), 12), np.nan)
    for i, u in enumerate(a5_ids):
        k = fold.get(u)
        if k is not None and k < raw.shape[0]:
            a5[i] = raw[k][i]
    fams["A5 (DINOv3, fold-correct)"] = (a5_ids, a5)

    rad = {}
    for r in csv.DictReader(open(ROOT / "downloads/research/oof/v52_oof.csv", newline="",
                                 encoding="utf-8")):
        rad[r["StudyInstanceUID"]] = np.array([float(r[t]) for t in LABELS])
    fams["Rad heads (v52)"] = (list(rad), np.array([rad[u] for u in rad]))

    print("families loaded: %d   report labels cover %.1f%% of cells"
          % (len(fams), 100.0 * M.mean()))
    print("\n%-30s %6s %6s  %s" % ("family", "n", "folds", "macro-AUC (report labels)"))
    rows = []
    for name, (ids, P) in fams.items():
        keep = [i for i, u in enumerate(ids)
                if u in pos_lab and np.isfinite(P[i]).all()
                and (not a.fold0_only or fold.get(u) == 0)]
        if len(keep) < 50:
            continue
        Ys = np.array([Y[pos_lab[ids[i]]] for i in keep])
        Ms = np.array([M[pos_lab[ids[i]]] for i in keep])
        fs = sorted({fold.get(ids[i], -1) for i in keep})
        rows.append((macro(Ys, Ms, np.array([P[i] for i in keep])), name, len(keep), fs))
    for s, name, n, fs in sorted(rows, reverse=True):
        print("%-30s %6d %6s  %.4f" % (name, n, ",".join(str(x) for x in fs), s))

    print("\n=> read the ours/public20 block against the A5 and Rad rows.  If our arms sit with the")
    print("   public20 seeds, then A106's 0.04 gap is a statement about A5 -- the single strongest")
    print("   arm in the stack -- not about our training, and the closure needs re-reading.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
