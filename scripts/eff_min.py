"""Minimal honest check: per-fold gain of the EffB0 leg at w=0.10, using the leg's OWN folds.

Deliberately tiny -- 8 metric calls instead of the full 200+, because the full sweep was
hanging and the question does not need it:

    Q: is the adopted +0.0063 at w=0.10 uniform across the leg's four folds, or does it
       straddle zero (i.e. it was read off whichever split flattered it)?

The leg's folds are its own out-of-fold predictions, and its fold split is NOT ours (19.2%
agreement = chance), so this is a genuine outer split. If all four folds move the same way by a
similar amount, the leg is real. If they straddle zero, treat it as noise and do not select it.

Usage:  python work/eff_min.py
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
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eff_honest import macro_auc, rankpct  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OOF = ROOT / "downloads" / "oof"
EFF = ROOT / "work" / "eff_assets"


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
    cov = fold_of >= 0
    print("leg covers %d / %d studies" % (cov.sum(), len(ids)))

    # rank-align both sides, then fill uncovered studies with the leg's median rank so the
    # blend is defined everywhere (w=0 must reproduce the parent exactly)
    br = rankpct(ours)
    lr = rankpct(leg)
    lr = np.where(np.isnan(lr), np.nanmedian(lr, axis=0), lr)

    print("\n%-6s %8s %9s %9s %9s %9s" % ("fold", "n", "w=0", "w=0.10", "gain", "gain/w"))
    gains, gains_oof = [], []
    for k in range(4):
        te = fold_of == k
        a0 = macro_auc(y[te], br[te])
        a1 = macro_auc(y[te], 0.9 * br[te] + 0.1 * lr[te])
        gains.append(a1 - a0)
        # parent scored on the whole labelled set, leg only where it exists -> apples to apples
        print("%-6d %8d %9.5f %9.5f %+9.5f %+9.5f"
              % (k, te.sum(), a0, a1, a1 - a0, (a1 - a0)))
    g = np.array(gains)
    # each fold is ~1/4 of the population, so the population gain is the mean
    print("\nper-fold gains: %s" % np.round(g, 5).tolist())
    print("mean gain over the covered population @w=0.10 = %+.5f" % g.mean())
    print("folds positive: %d/4" % (g > 0).sum())
    # paired bootstrap ACROSS folds (the four folds are independent samples by construction)
    rng = np.random.default_rng(0)
    bs = rng.choice(g, size=(20000, 4), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("bootstrap over folds: mean %+.5f  CI [%+.5f, %+.5f]  P(>0)=%.3f"
          % (g.mean(), lo, hi, (bs > 0).mean()))

    print("\n---- whole-population, w=0.10, for reference ----")
    a0 = macro_auc(y, br)
    a1 = macro_auc(y, 0.9 * br + 0.1 * lr)
    print("full-set w=0 %.5f -> w=0.10 %.5f  (%+.5f)   <-- in-sample, what adoption quoted"
          % (a0, a1, a1 - a0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
