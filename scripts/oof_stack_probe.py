"""Does a CROSS-LABEL stacking layer beat the plain weighted rank-mean?

WHY (and what is already closed)
    On the 4407-study public-pipeline OOF we have already measured, out-of-fold:
      * per-LABEL arm weights (diagonal stacking)      -> -0.0003  (HANDOFF 86.3)  CLOSED
      * window pooling / arm weight axes               -> ~0            (86.2, 88)  CLOSED
    What has NOT been tested is using the OTHER labels' arm outputs as features for a given
    label.  The ensemble's own calibration comment says the three OA compartments "move
    together", and findings that co-occur are found together -- i.e. there is known structure
    ACROSS labels that a per-label blend cannot use.

    This fits, per label, a regularised logistic regression on all 4 arms x 12 labels (48
    features), strictly out-of-fold, and compares with the baseline blend by paired bootstrap
    on the same studies.

    Falsifiable and cheap: 4407 studies, zero GPU, seconds to run.

Usage:
    python scripts/oof_stack_probe.py                 # default: rank features, C=0.1
    python scripts/oof_stack_probe.py --C 1.0 --raw   # raw probabilities instead of ranks
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
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
OOF = ROOT / "downloads" / "oof"
sys.path.insert(0, str(ROOT / "scripts"))
from cv_score import LABELS, paired_bootstrap  # noqa: E402


def macro(Y, P):
    vals = []
    for j in range(Y.shape[1]):
        yy = (Y[:, j] >= 0.5).astype(int)
        if len(set(yy.tolist())) > 1:
            vals.append(roc_auc_score(yy, P[:, j]))
    return float(np.mean(vals))


def rank_pct(a):
    out = np.empty_like(a, dtype=np.float64)
    for j in range(a.shape[1]):
        out[:, j] = rankdata(a[:, j]) / len(a)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--raw", action="store_true", help="use raw probabilities, not rank percentiles")
    args = ap.parse_args(argv)

    arms = np.load(OOF / "oof_arm_prob.npy").astype(np.float64)      # (N,4,12)
    Y = np.load(OOF / "train_y.npy").astype(np.float64)
    valid = np.load(OOF / "valid_mask.npy")
    folds = np.load(OOF / "train_fold.npy")
    N = Y.shape[0]
    print("OOF %d studies, %d arms, %d labels, %d folds"
          % (N, arms.shape[1], arms.shape[2], len(set(folds.tolist()))))

    # ---- baseline: the production-style weighted rank-mean of the arms ----
    R = rank_pct(arms.reshape(-1, arms.shape[2])).reshape(arms.shape)
    base_w = np.array([0.55, 0.10, 0.15, 0.20])
    base_w = base_w / base_w.sum()
    P_base = np.tensordot(R, base_w, axes=([1], [0]))
    print("baseline (rank-mean, canonical arm weights) macro-AUC = %.4f" % macro(Y[valid], P_base[valid]))

    # ---- features ----
    if args.raw:
        F = arms.reshape(N, -1)
        print("features: raw probabilities, %d wide" % F.shape[1])
    else:
        F = R.reshape(N, -1)
        print("features: rank percentiles, %d wide (ALL labels for every label)" % F.shape[1])

    # ---- out-of-fold stacking ----
    P_stack = np.zeros((N, 12))
    fold_ids = sorted(set(folds.tolist()))
    for f in fold_ids:
        te = (folds == f)
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        for j in range(12):
            yj_tr = (Y[tr, j] >= 0.5).astype(int)
            yj_te = (Y[te, j] >= 0.5).astype(int)
            if len(set(yj_tr.tolist())) < 2:
                P_stack[te, j] = P_base[te, j]      # degenerate: keep the baseline
                continue
            lr = LogisticRegression(C=args.C, max_iter=2000)
            lr.fit(F[tr], yj_tr)
            P_stack[te, j] = lr.predict_proba(F[te])[:, 1]
        print("  fitted fold %d (train %d / test %d)" % (f, int(tr.sum()), int(te.sum())))

    m_base = macro(Y[valid], P_base[valid])
    m_stack = macro(Y[valid], P_stack[valid])
    print("\n== out-of-fold macro-AUC ==")
    print("  baseline rank-mean : %.4f" % m_base)
    print("  cross-label stack  : %.4f  (delta %+.4f)" % (m_stack, m_stack - m_base))
    _, _, ci, win = paired_bootstrap(Y[valid], P_stack[valid], P_base[valid],
                                     n_boot=600, seed=7)
    print("  paired bootstrap   : delta %+.4f  95%% CI [%+.4f, %+.4f]  A wins %.0f%%"
          % (m_stack - m_base, ci[0], ci[1], win))
    verdict = "REAL" if (ci[0] > 0 or ci[1] < 0) else "NOT RESOLVED"
    print("  verdict: %s" % verdict)

    # per-label view, so a single big label cannot hide a broad regression
    print("\n  per-label delta (stack - baseline):")
    for j, nm in enumerate(LABELS):
        yj = (Y[valid, j] >= 0.5).astype(int)
        if len(set(yj.tolist())) < 2:
            continue
        d = roc_auc_score(yj, P_stack[valid, j]) - roc_auc_score(yj, P_base[valid, j])
        print("    %-18s %+.4f" % (nm, d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
