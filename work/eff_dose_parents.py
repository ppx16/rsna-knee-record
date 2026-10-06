"""Robustness of the EffB0 dose conclusion across different PARENTS.

WHY
    `work/eff_dose.py` established that w=0.15 dominates w=0.10 at equal unanimity (4/4 folds
    positive), using `oof_equal_prob.npy` as the parent. But that file is the **stage-1 public20**
    surrogate, and a single parent is a single instrument. The decision it drives -- what weight the
    next EffB0 kernel runs at -- should not rest on one parent.

    `downloads/oof/` ships several parent variants. This walks the same honest per-fold grid against
    each, and reports the largest weight that stays 4/4 positive for every parent.

    The rule this tests: **pick the dose that is unanimous across folds under EVERY parent, not the
    dose that maximises the mean under one.** A dose that is only good under one parent is a fit to
    that parent.

Usage:  python work/eff_dose_parents.py
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
GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
PARENTS = ["oof_equal_prob.npy", "oof_selected.npy", "oof_arm_prob.npy",
           "oof_arm_logit_pool.npy"]


def main() -> int:
    y = np.load(OOF / "train_y.npy")
    ids = np.asarray([str(x) for x in np.load(OOF / "train_ids.npy", allow_pickle=True).ravel()])
    idx = {u: i for i, u in enumerate(ids)}

    leg = np.full(y.shape, np.nan, dtype=np.float64)
    fold_of = np.full(len(ids), -1, dtype=int)
    for k in range(4):
        p = np.load(EFF / ("oof_effb0mil_sl16_tv7w_f%d.npy" % k))
        pi = pd.read_csv(EFF / ("oof_effb0mil_sl16_tv7w_f%d_ids.csv" % k)).iloc[:, 0].astype(str)
        for r, u in enumerate(pi):
            j = idx.get(u)
            if j is not None:
                leg[j] = p[r]
                fold_of[j] = k
    fold_of[fold_of < 0] = 0
    lr = rankpct(leg)
    lr = np.where(np.isnan(lr), np.nanmedian(lr, axis=0), lr)

    summary = {}
    # ⚠️ `oof_equal_prob.npy` and `oof_selected.npy` are BYTE-IDENTICAL (same sha256), so they are
    # NOT two parents. The genuine alternatives are the per-arm arrays, which must be aggregated
    # first -- otherwise this whole test silently reduces to a single instrument, which is exactly
    # the "looks like more evidence than it is" failure this repo keeps hitting.
    parents = {}
    flat = OOF / "oof_equal_prob.npy"
    if flat.exists():
        parents["equal_prob (stage-1)"] = np.load(flat)
    ap = OOF / "oof_arm_prob.npy"
    if ap.exists() and np.load(ap).ndim == 3:
        A = np.load(ap).astype(np.float64)
        parents["mean-of-4-arms"] = np.nanmean(A, axis=1)
        parents["rankmean-of-4-arms"] = np.nanmean(rankpct(A.reshape(-1, 12)).reshape(A.shape), axis=1)
    lp = OOF / "oof_arm_logit_pool.npy"
    if lp.exists() and np.load(lp).ndim == 3:
        L = np.load(lp).astype(np.float64)
        parents["mean-logitpool-arms"] = np.nanmean(L, axis=1)

    for pname, arr in parents.items():
        if arr.shape != y.shape:
            print("== %-26s shape %s != y %s -- skipped" % (pname, arr.shape, y.shape))
            continue
        br = rankpct(arr)
        a0 = {k: macro_auc(y[fold_of == k], br[fold_of == k]) for k in range(4)}
        print("== %-26s base per-fold %s" % (pname, " ".join("%.5f" % a0[k] for k in range(4))))
        line = []
        for w in GRID:
            gs = []
            for k in range(4):
                te = fold_of == k
                gs.append(macro_auc(y[te], (1.0 - w) * br[te] + w * lr[te]) - a0[k])
            gs = np.array(gs)
            summary.setdefault(w, []).append((int((gs > 0).sum()), gs.mean()))
            line.append("w=%.2f %+.5f %d/4" % (w, gs.mean(), int((gs > 0).sum())))
        for i in range(0, len(line), 2):
            print("     " + "   |   ".join(line[i:i + 2]))

    print("\n==== unanimous (4/4) under EVERY parent? ====")
    safe = []
    for w in GRID:
        rows = summary.get(w, [])
        if rows and all(pos == 4 for pos, _ in rows):
            safe.append(w)
        print("  w=%.2f  %s" % (w, "  ".join("%d/4 (%+.5f)" % (pos, m) for pos, m in rows)))
    print("\nlargest dose unanimous under every parent: %s"
          % (("w=%.2f" % max(safe)) if safe else "NONE"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
