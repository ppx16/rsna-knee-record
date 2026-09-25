# -*- coding: utf-8 -*-
"""Assemble the CV folds' OOF predictions and score them against LB ground truth.

WHAT IT READS
    Each `kernel-cv-foldK` run writes, via the training script's own CV hook,

        raptor_oof_<tag>_fold<K>.npz   with  pred (N,12), truth (N,12), ids (N,), fold, nfolds

    where the N studies are exactly the ones held out for that fold.  Concatenated over
    K = 0..nfolds-1 they cover every non-gold training study exactly once.

WHAT IT IS FOR  (HANDOFF 75.2 / 76.4)
    A local CV is only worth using if it TRACKS the public leaderboard.  The project has a
    recorded list of known LB differences - `state/lb_ground_truth.csv` - and the CV has to
    reproduce them.  Gate 1 is the cheap one:

        our own trained members scored 0.842 / 0.874 / 0.854 on the LB,
        the 0.941 ensemble scores 0.941  ->  a ~0.08 gap the CV must show too.

    Gate 2 (the routing dose-response 0.939 / 0.940 / 0.941) is NOT reachable from here:
    routing acts in the final fusion layer, which needs every arm including the 64-slot
    ones that the preprocessed corpus cannot feed.  That limitation is known and recorded.

LABEL CHOICE
    Score against `flight0234/v4hybrid`, not the 0.349-mean weak labels: HANDOFF 75.3
    measured local AUC 0.8505 (v4hybrid) vs 0.8497 (weak) on the same predictions, and
    v4hybrid is the better substrate on gold-58 (0.8608 vs 0.8433).

Usage:
    python scripts/cv_assemble.py                      # auto-find landed fold outputs
    python scripts/cv_assemble.py --dir outputs/_landing
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
V4HYBRID = ROOT / "downloads" / "labels" / "rsna-knee-hybrid-report-labels" / "report_labels_v4hybrid.csv"
TRAIN = ROOT / "data" / "raw" / "train.csv"


def load_folds(root: pathlib.Path, arm: str | None = None):
    """All raptor_oof npz files under `root`, newest per (tag, fold) wins.

    ARM FILTERING MATTERS (HANDOFF 78 / 80).  Three tag families live on disk:
        A : tag 'cv0', 'cv1' (the two folds pushed before the naming change) and 'cva2..4'
        G : tag 'cvg0..4'   - the 1-epoch large-effect control
        B : tag 'cvb0..4'   - the rule-label arm
    Without a filter an assembly would silently mix arms and the OOF table would be
    meaningless.  The regexes below are deliberately explicit rather than a prefix test,
    because 'cv0' does NOT start with 'cva'.
    """
    import re as _re
    pat = {
        None: _re.compile(r"^cv\w*$"),
        "A": _re.compile(r"^cv(\d+|a\d+)$"),
        "G": _re.compile(r"^cvg\d+$"),
        "B": _re.compile(r"^cvb\d+$"),
    }[arm]
    best = {}
    for p in root.rglob("raptor_oof_*_fold*.npz"):
        m = _re.search(r"raptor_oof_(.+)_fold(\d+)\.npz$", p.name)
        if not m:
            continue
        tag = m.group(1)
        if not pat.match(tag):
            continue
        # per-epoch files (ep<k>_fold) are the same fold measured at another epoch - skip
        # them here; the assembler wants the best-weight / ensemble file per fold.
        if _re.search(r"_ep\d+$", tag) or tag.endswith("_epens"):
            continue
        # Key by FOLD ONLY, newest file wins.  The A arm exists under two tag families -
        # 'cv0'/'cv1' were pushed before the naming change and 'cva0..4' after - and both
        # would otherwise contribute the same held-out studies to the same fold, doubling
        # those ids.  The later-written file is the one from the current kernels.
        key = int(m.group(2))
        if key not in best or p.stat().st_mtime > best[key][0].stat().st_mtime:
            best[key] = (p, tag)
    return best


def macro_auc(Y, P, mask=None):
    m = np.ones(len(Y), bool) if mask is None else mask
    vals = []
    for j in range(len(LAB)):
        yy = (Y[m, j] >= 0.5).astype(int)
        if len(set(yy.tolist())) > 1:
            vals.append(roc_auc_score(yy, P[m, j]))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "outputs" / "_landing"))
    ap.add_argument("--arm", default=None, choices=["A", "G", "B"],
                    help="assemble ONE arm only. Never mix arms - see load_folds().")
    args = ap.parse_args()

    folds = load_folds(pathlib.Path(args.dir), args.arm)
    if not folds:
        print("no raptor_oof_*_fold*.npz%s under %s yet - the fold kernels may still be running"
              % ("" if args.arm is None else " for arm %s" % args.arm, args.dir))
        return 2

    ids, preds = [], []
    for k in sorted(folds):
        p, tag = folds[k]
        z = np.load(p, allow_pickle=True)
        ids.extend([str(x) for x in z["ids"]])
        preds.append(np.asarray(z["pred"], dtype=np.float32))
        print("  fold %-2d tag=%-6s n=%-5d  %s" % (k, tag, len(z["ids"]), p.name))
    ids = np.array(ids)
    P = np.concatenate(preds, 0)

    dup = len(ids) - len(set(ids.tolist()))
    print("\nstudies: %d   duplicate ids: %d   (0 duplicates = a clean partition)" % (len(ids), dup))

    # ---- labels: v4hybrid is the scoring substrate (HANDOFF 75.3) --------------
    order = {u: i for i, u in enumerate(ids)}
    Y = np.full((len(ids), len(LAB)), np.nan, np.float32)
    for r in csv.DictReader(open(V4HYBRID, encoding="utf-8-sig", newline="")):
        i = order.get(r["StudyInstanceUID"])
        if i is None:
            continue
        for j, l in enumerate(LAB):
            Y[i, j] = float(r[l])

    out = ROOT / "outputs" / ("cv_oof%s.npz" % ("" if args.arm is None else "_" + args.arm))
    np.savez(out, pred=P, ids=ids, truth=Y)
    print("wrote", out, P.shape)

    ok = ~np.isnan(Y).any(1)
    print("\n=== CV score (v4hybrid labels)%s ===" % ("" if args.arm is None else ", arm " + args.arm))
    print("  folds present            : %s" % sorted(k for _, k in folds))
    print("  non-gold studies covered : %d / %d%s"
          % (ok.sum(), len(ids), "" if ok.all() else "   <- NOT a complete partition yet"))
    print("  macro-AUC                : %.4f" % macro_auc(Y, P, ok))

    # ---- gate 1 (HANDOFF 80.2): the 1-epoch control must score clearly LOWER ----
    print("\n=== GATE 1 (HANDOFF 80.2) ===")
    print("  assemble arm G and arm A separately, then compare:")
    print("    python scripts/cv_assemble.py --arm G      # 1 epoch")
    print("    python scripts/cv_assemble.py --arm A      # 12 epochs")
    print("  G must be CLEARLY lower than A (gold-58 reference: 0.7272 vs 0.8440 = 0.117).")
    print("  Pass -> the CV resolves large effects and is usable.")
    print("  Fail -> the CV cannot even separate a 0.117 gap: DISCARD it, do not iterate on it.")
    print("\n  NOTE: the 0.941 ensemble cannot be reproduced here (routing needs the 64-slot")
    print("        arms, which the preprocessed corpus cannot feed) - gate 2 is out of reach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
