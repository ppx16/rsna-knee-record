# -*- coding: utf-8 -*-
"""Honest fusion audit on the full 4,407-study OOF table.

Input is `denisrukavishnikov/rnsa-knee-pred-public20-oof-audit`, the only artifact found
that carries per-arm out-of-fold predictions for every training study rather than for the
58 gold ones. That distinction is the whole point: the competition's own community has
already documented (yosukeinada, 22 LB-verified submissions) that the 58-study gold set
cannot referee a +0.004 decision, and that a Gold58-based forecast of 0.941 delivered
0.937 - a -0.004 error. Fitting on 4,407 studies instead of 58 is the fix.

⚠️ READ THIS BEFORE QUOTING A NUMBER FROM HERE
The labels in this table are the WEAK LLM labels (7 distinct values, mean 0.349), not
ground truth. An AUC measured against them is capped well below the leaderboard: this
Stage-1 stack scores 0.8494 here and 0.899 on the public LB. So:
    * OOF numbers are only comparable to OTHER OOF numbers from this same table;
    * never present an OOF delta as a predicted LB delta - that is the exact error the
      community already made once.

Measured 2026-09-11 (scripts/oof_audit.py):
    equal arm mean        0.8497     four seeds of one model, averaged
    global_weight_fitted  0.8497     the shipped global weight vector [0.3,0.2,0.3,0.2]
    target_weight_fitted  0.8502     +0.0005 over equal - the only fitted gain
    oof_equal_prob        0.8494     what the author actually shipped
    rank-mean arms        0.8495
    per-target ridge CV   0.8480     WORSE out-of-fold than doing nothing

The conclusion this supports: the four Stage-1 arms are near-identical (solo macro-AUC
0.8412-0.8424), so there is almost no disagreement for a fusion to exploit, and a fitted
per-target stack only adds variance. This independently reproduces the community finding
that per-label member selection shows +0.011 in-sample and +0.0003 out-of-fold. Ensemble
value comes from DISAGREEMENT, not from better weighting of agreement.

Usage:
    python scripts/oof_audit.py                 # the table above
    python scripts/oof_audit.py --dir downloads/oof
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
import pathlib

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

DEFAULT_DIR = pathlib.Path(__file__).resolve().parent.parent / "downloads" / "oof"
N_LABELS = 12


def macro_auc(pred, y_bin):
    return float(np.mean([roc_auc_score(y_bin[:, i], pred[:, i])
                          for i in range(N_LABELS)]))


def per_label(pred, y_bin):
    return [roc_auc_score(y_bin[:, i], pred[:, i]) for i in range(N_LABELS)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    if not (d / "oof_arm_prob.npy").is_file():
        print("missing %s - download denisrukavishnikov/rnsa-knee-pred-public20-oof-audit"
              % d)
        return 2

    load = lambda n: np.load(d / n, allow_pickle=True)          # noqa: E731
    y = load("train_y.npy")
    y_bin = (y >= 0.5).astype(int)
    targets = load("targets.npy").tolist()
    arm = load("oof_arm_prob.npy")                 # (4407, 4 arms, 12 labels)
    tw = load("target_weight_full.npy")            # (4 arms, 12 labels)
    gw = load("global_weight_full.npy")            # (4 arms,)
    fold = load("train_fold.npy")

    print("rows=%d  arms=%d  labels=%d  label mean=%.4f"
          % (arm.shape[0], arm.shape[1], arm.shape[2], float(y.mean())))
    print("arms: %s" % ", ".join(load("arm_labels.npy").tolist()))
    for k, name in enumerate(load("arm_labels.npy").tolist()):
        print("  arm %-10s solo macro-AUC = %.4f" % (name, macro_auc(arm[:, k, :], y_bin)))
    print()

    rows = [
        ("equal arm mean", macro_auc(arm.mean(1), y_bin)),
        ("rank-mean arms", macro_auc(
            np.stack([np.apply_along_axis(rankdata, 0, arm[:, k, :]) for k in range(arm.shape[1])],
                     1).mean(1), y_bin)),
        ("global_weight_fitted", macro_auc((arm * gw[None, :, None]).sum(1), y_bin)),
        ("target_weight_fitted", macro_auc((arm * tw[None, :, :]).sum(1), y_bin)),
        ("oof_equal_prob (shipped)", macro_auc(load("oof_equal_prob.npy"), y_bin)),
    ]

    # Per-target ridge stacking, cross-validated on the SAME fold assignment used to make
    # the OOF in the first place. This is the honest version of "fit the weights".
    stack = np.zeros((arm.shape[0], N_LABELS))
    for i in range(N_LABELS):
        for f in np.unique(fold):
            tr, va = fold != f, fold == f
            X, Xv = arm[tr, :, i], arm[va, :, i]
            w = np.linalg.lstsq(np.c_[X, np.ones(len(X))], y_bin[tr, i].astype(float),
                                rcond=None)[0]
            stack[va, i] = np.c_[Xv, np.ones(len(Xv))] @ w
    rows.append(("per-target ridge CV", macro_auc(stack, y_bin)))

    print("%-26s macro-AUC" % "fusion")
    for name, val in rows:
        print("%-26s %.4f" % (name, val))

    print("\nper-label, equal vs shipped target weights:")
    pe, pt = per_label(arm.mean(1), y_bin), per_label((arm * tw[None, :, :]).sum(1), y_bin)
    for t, a, b in zip(targets, pe, pt):
        print("   %-18s equal=%.4f  tw=%.4f  delta=%+.4f" % (t, a, b, b - a))

    best = max(v for _, v in rows)
    gain = best - rows[0][1]
    print("\nbest fusion beats the equal-weight mean by %+.4f" % gain)
    if gain <= 0.001:
        print("=> No measurable headroom in re-weighting these arms. Their errors are too "
              "correlated; spend submissions on a NEW arm instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
