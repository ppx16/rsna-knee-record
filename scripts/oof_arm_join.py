"""Does OUR trained arm add anything to the author's 4-arm OOF?

WHY (HANDOFF 85.4 parked this for want of a fair comparison)
    We trained an arm (raptor, v4hybrid labels) whose gold-58 is 0.8470 -- above the public
    members' 0.8401 -- but it is NOT in the ensemble, and §85.4 parked "insert it" because no
    fair comparison seemed available: the author's arms had seen fold 0's studies, and so had
    our production arm, so a fold-0 comparison would only measure who memorised better.

    That objection dissolves once BOTH sides are out-of-fold:

        downloads/oof/oof_arm_prob   (4407, 4, 12)   author's 4 arms, each study predicted by
                                                     a model that did NOT train on it
        downloads/cv/.../cva0                        our arm's fold-0 OOF (870 studies our arm
                                                     did NOT train on)

    Both are honest on the studies they share.  So the marginal-contribution question is
    answerable offline, at zero GPU cost, BEFORE spending a day of integration work.

METHOD
    Rank-mean blending, because that is what the production fusion does
    ("submission.csv = weighted rank mean of N member(s)") and because rank normalisation
    removes calibration differences between our model and the author's.
    Paired bootstrap over the shared studies gives the error bar.

Usage:
    python scripts/oof_arm_join.py --ours downloads/cv/cva0
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
import glob
import os
import pathlib
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cv_score import LABELS, paired_bootstrap  # noqa: E402


def rank_pct(P):
    """Per-label rank percentile in [0,1] -- column-wise, NaNs impossible here."""
    out = np.empty_like(P, dtype=np.float64)
    for j in range(P.shape[1]):
        out[:, j] = rankdata(P[:, j]) / len(P)
    return out


def macro(Y, P):
    vals = []
    for j in range(Y.shape[1]):
        yy = (Y[:, j] >= 0.5).astype(int)
        if len(set(yy.tolist())) > 1:
            vals.append(roc_auc_score(yy, P[:, j]))
    return float(np.mean(vals)) if vals else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=os.path.join("downloads", "cv", "cva0"),
                    help="directory holding our raptor_oof_<tag>_fold<N>.npz")
    ap.add_argument("--old", default=os.path.join("downloads", "oof"))
    ap.add_argument("--boot", type=int, default=600)
    a = ap.parse_args(argv)

    import os as _os  # noqa: F401  (kept for symmetry; os is imported at module level)
    old = pathlib.Path(a.old)
    ids = np.load(old / "train_ids.npy").astype(str)
    arms = np.load(old / "oof_arm_prob.npy").astype(np.float64)
    Y = np.load(old / "train_y.npy").astype(np.float64)
    gold = np.load(old / "gold_mask.npy")
    names = [str(x) for x in np.load(old / "arm_labels.npy", allow_pickle=True)]
    print("author OOF: %d studies x %d arms %s" % (len(ids), arms.shape[1], names))

    hits = sorted(glob.glob(os.path.join(a.ours, "**", "raptor_oof_*.npz"), recursive=True))
    ours_files = [h for h in hits if "_ep" not in os.path.basename(h)
                  and "_swa" not in os.path.basename(h)]
    if not ours_files:
        print("no OUR oof npz under %s -- run cv_score.py on the landed fold first" % a.ours)
        return 1
    d = np.load(ours_files[0])
    o_ids = np.asarray(d["ids"]).astype(str)
    o_P = d["pred"].astype(np.float64)
    print("ours      : %s  %d studies" % (os.path.basename(ours_files[0]), len(o_ids)))

    old_pos = {v: i for i, v in enumerate(ids)}
    common = np.array([v for v in o_ids if v in old_pos])
    if common.size == 0:
        print("NO shared study ids -- the two id spaces do not overlap")
        return 1
    oi = np.array([list(o_ids).index(v) for v in common])
    ai = np.array([old_pos[v] for v in common])
    print("shared    : %d studies (%d of them gold)" % (common.size, int(gold[ai].sum())))

    Yc = Y[ai]
    R_old = rank_pct(arms[ai].reshape(-1, arms.shape[2])).reshape(len(ai), arms.shape[1], -1)
    R_our = rank_pct(o_P[oi])
    R_old_flat = arms[ai].mean(1)          # for the correlation table below

    print("\n== solo (rank-AUC on the shared studies) ==")
    for k, nm in enumerate(names):
        print("  author arm %-10s %.4f" % (nm, macro(Yc, R_old[:, k, :])))
    print("  OURS                  %.4f" % macro(Yc, R_our))

    print("\n== how correlated is our arm with each author arm? (mean per-label Spearman) ==")
    for k, nm in enumerate(names):
        rhos = [np.corrcoef(rankdata(R_our[:, j]), rankdata(R_old[:, k, j]))[0, 1]
                for j in range(Yc.shape[1])]
        print("  vs %-10s mean rho %.3f" % (nm, float(np.mean(rhos))))

    base_w = np.array([0.55, 0.10, 0.15, 0.20])
    base_w = base_w / base_w.sum()
    P_base = np.tensordot(R_old, base_w, axes=([1], [0]))
    P_eq = R_old.mean(1)
    print("\n== blends ==")
    print("  author 4-arm, canonical weights  %.4f" % macro(Yc, P_base))
    print("  author 4-arm, equal weights      %.4f" % macro(Yc, P_eq))

    print("\n== adding OUR arm (rank-mean, alpha = weight of ours) ==")
    for alpha in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        P = (1 - alpha) * P_eq + alpha * R_our
        _, _, ci, win = paired_bootstrap(Yc, P, P_eq, n_boot=a.boot, seed=11)
        d0 = macro(Yc, P) - macro(Yc, P_eq)
        print("  alpha=%.2f  %.4f   delta %+.4f  CI [%+.4f, %+.4f]  %s"
              % (alpha, macro(Yc, P), d0, ci[0], ci[1],
                 "REAL" if (ci[0] > 0 or ci[1] < 0) else "not resolved"))

    print("\n== and the same on gold-58 only (the official-rubric slice) ==")
    # Our gold predictions come from raptor_gold_<tag>.npz, whose `truth` is the COMPETITION's
    # own labels: the training script defines gold as the train.csv rows where all 12 label
    # columns are non-null, and holds them out of training entirely.  So this is the one
    # comparison in the whole project that is scored against official labels.
    gfiles = sorted(glob.glob(os.path.join(a.ours, "**", "raptor_gold_*.npz"), recursive=True))
    if not gfiles:
        print("  no raptor_gold_*.npz under %s" % a.ours)
        return 0
    gd = np.load(gfiles[0])
    g_ids = np.asarray(gd["ids"]).astype(str)
    g_P = gd["pred"].astype(np.float64)
    g_Y = gd["truth"].astype(np.float64)
    keep = np.array([v in old_pos for v in g_ids])
    if keep.sum() < 20:
        print("  only %d of our %d gold studies are in the author OOF -- skipping"
              % (int(keep.sum()), len(g_ids)))
        return 0
    gi = np.array([old_pos[v] for v in g_ids[keep]])
    Yg = g_Y[keep]
    Og = rank_pct(g_P[keep])
    Rg = rank_pct(arms[gi].reshape(-1, arms.shape[2])).reshape(len(gi), arms.shape[1], -1)
    Ag = Rg.mean(1)
    print("  shared gold studies: %d" % len(gi))
    print("  author 4-arm equal %.4f   OURS %.4f" % (macro(Yg, Ag), macro(Yg, Og)))
    for k, nm in enumerate(names):
        print("    author arm %-10s %.4f" % (nm, macro(Yg, Rg[:, k, :])))
    for alpha in (0.10, 0.20, 0.30):
        P = (1 - alpha) * Ag + alpha * Og
        _, _, ci, _ = paired_bootstrap(Yg, P, Ag, n_boot=a.boot, seed=12)
        print("  alpha=%.2f  %.4f  delta %+.4f  CI [%+.4f, %+.4f]  %s"
              % (alpha, macro(Yg, P), macro(Yg, P) - macro(Yg, Ag), ci[0], ci[1],
                 "REAL" if (ci[0] > 0 or ci[1] < 0) else "not resolved"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
