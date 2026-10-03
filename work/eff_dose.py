"""Honest DOSE curve for the EffB0-MIL leg — is w=0.10 leaving gain on the table?

WHY
    The leg is applied at a deliberately conservative **w=0.10**. The adoption curve (in-sample on
    our OOF) peaked higher, at 0.20-0.30. The honest per-fold test showed +0.00403 at w=0.10 with
    4/4 folds positive — but it only tested that ONE weight, so it cannot say whether 0.10 is
    under-dosing or correctly cautious.

    This walks a small weight grid and scores each weight on the SAME honest split: the leg's own
    four out-of-fold folds (which are independent of ours — 19.2% agreement = chance). For each
    weight it reports the mean gain over the four folds and how many folds agree in sign. A weight
    whose gain is positive in 4/4 folds is a candidate; one that straddles zero is not.

    This is NOT "tuning blend weights" (closed item #1). That closure is about re-fitting the
    PUBLISHED fusion weights on the PUBLIC split. This is asking whether a NEW member, which has
    its own independent out-of-fold predictions, is being under-dosed — and it is answered on a
    split the member never trained on.

Usage:  python work/eff_dose.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eff_honest import macro_auc, rankpct  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OOF = ROOT / "downloads" / "oof"
EFF = ROOT / "work" / "eff_assets"
GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def main() -> int:
    ours = np.load(OOF / "oof_equal_prob.npy")
    y = np.load(OOF / "train_y.npy")
    ids = np.asarray([str(x) for x in np.load(OOF / "train_ids.npy", allow_pickle=True).ravel()])
    idx = {u: i for i, u in enumerate(ids)}

    leg = np.full_like(ours, np.nan, dtype=np.float64)
    fold_of = np.full(len(ids), -1, dtype=int)
    for k in range(4):
        p = np.load(EFF / ("oof_effb0mil_sl16_tv7w_f%d.npy" % k))
        pi = pd.read_csv(EFF / ("oof_effb0mil_sl16_tv7w_f%d_ids.csv" % k)).iloc[:, 0].astype(str)
        for r, u in enumerate(pi):
            j = idx.get(u)
            if j is not None:
                leg[j] = p[r]
                fold_of[j] = k

    br = rankpct(ours)
    lr = rankpct(leg)
    lr = np.where(np.isnan(lr), np.nanmedian(lr, axis=0), lr)
    fold_of[fold_of < 0] = 0          # the 58 uncovered studies ride along; they are 1.3% of n

    # base AUC per fold, computed once
    a0 = {k: macro_auc(y[fold_of == k], br[fold_of == k]) for k in range(4)}

    print("%-7s %s   %-9s %-6s" % ("w", "per-fold gain".ljust(44), "mean", "pos/4"))
    rows = []
    for w in GRID:
        gs = []
        for k in range(4):
            te = fold_of == k
            aw = macro_auc(y[te], (1.0 - w) * br[te] + w * lr[te])
            gs.append(aw - a0[k])
        gs = np.array(gs)
        rows.append(dict(w=w, mean=gs.mean(), pos=int((gs > 0).sum()),
                         lo=gs.min(), hi=gs.max()))
        print("%-7.2f %-44s %+9.5f %d/4"
              % (w, " ".join("%+.5f" % g for g in gs), gs.mean(), int((gs > 0).sum())))

    t = pd.DataFrame(rows)
    best = t.loc[t["mean"].idxmax()]
    print("\nbest mean gain: w=%.2f  %+.5f  (positive in %d/4 folds)"
          % (best.w, best["mean"], best.pos))
    print("w=0.10 reference: %+.5f" % t.loc[t.w == 0.10, "mean"].iloc[0])
    print("\n(positive-in-4/4 weights: %s)"
          % t.loc[t.pos == 4, "w"].tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
