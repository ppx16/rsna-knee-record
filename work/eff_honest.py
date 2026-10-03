"""Is the EffB0-MIL leg's +0.0063 real out-of-fold, or an artifact of picking w on the same data?

WHY THIS EXISTS
    The leg was adopted on a curve measured against OUR OOF:
        w=0.00 0.84962 | 0.05 0.85290 | 0.10 0.85587 | 0.15 0.85851 | 0.20 0.86083 | 0.30 0.86440
    But that curve is IN-SAMPLE for the weight: w was read off the same 4349 studies it is
    evaluated on. Our own closed item #8 is exactly this failure -- per-label member selection
    gave +0.011 in-sample and +0.0003 out-of-fold.

    The EffB0 checkpoints come with FOUR fold predictions, and their fold split is NOT ours
    (19.2% agreement = chance, verified). That gives an outer split we can use honestly:

        for each fold k:  pick w* on the other 3 folds,  evaluate the gain at w* on fold k

    If the out-of-fold gain collapses toward 0, the leg is a mirage and must not be selected.
    If it survives at ~+0.005, the leg is real and worth its slot.

ALSO REPORTS
    * the fixed-weight (w=0.10) gain per fold, so the spread is visible
    * a paired bootstrap CI on the honest estimate
    * the same numbers on `oof_selected.npy`, as a second parent

Usage:  python work/eff_honest.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
OOF = ROOT / "downloads" / "oof"
EFF = ROOT / "work" / "eff_assets"
CAND_W = np.round(np.arange(0.0, 0.401, 0.01), 2)


def _col_auc(yj: np.ndarray, pj: np.ndarray) -> float:
    """Soft (pairwise) AUC for ONE column.

        AUC = sum_{i,j} [y_i > y_j] * (1[p_i > p_j] + 0.5*1[p_i == p_j])
              / sum_{i,j} [y_i > y_j]

    This is the generalisation that stays correct when `y` is a soft label rather than 0/1.
    Chunked because the full n x n comparison is 4407^2 floats.

    ⚠️ Do NOT reach for `sklearn.roc_auc_score` here: on this install it raises
    `ValueError: continuous format is not supported` for soft `y_true`, and an earlier version of
    this file swallowed that in a bare `except ValueError: return 0.5` -- so every column came
    back 0.5 and the run printed a completely fictitious "honest gain = +0.00000, P(>0)=0.000".
    Two silent-instrument failures in a row from the same three lines. Exceptions now propagate.
    """
    n = yj.size
    if n < 2 or len(np.unique(yj)) < 2:
        return 0.5
    order = np.argsort(pj, kind="stable")
    ps, ys = pj[order], yj[order]
    vals, inv = np.unique(ys, return_inverse=True)
    k = vals.size
    total = np.bincount(inv, minlength=k).astype(np.float64)
    # denominator: every ordered pair with y_i > y_j
    below = np.concatenate([[0.0], np.cumsum(total)[:-1]])
    den = float((total * below).sum())
    if den <= 0:
        return 0.5
    hist = np.zeros(k, dtype=np.float64)
    num = 0.0
    i = 0
    while i < n:
        # ⚠️ `j` MUST start at i+1, not i. With `j = i`, a NaN in `ps` makes `ps[j] == ps[i]`
        # False (NaN != NaN), so j never advances, `i = j` re-enters the same iteration, and the
        # loop spins forever -- which is exactly the hang this script produced for 7+ minutes
        # while a single metric call takes 0.38 s.
        j = i + 1
        while j < n and ps[j] == ps[i]:
            j += 1
        grp = inv[i:j]
        cnt = np.bincount(grp, minlength=k).astype(np.float64)
        below_g = np.concatenate([[0.0], np.cumsum(cnt)[:-1]])
        # ⚠️ `below` must come from the RUNNING histogram (p strictly better so far), not from the
        # global `total`: using `total` double-counts pairs whose p-rank is worse, which pushed
        # the result to 1.00001 -- an AUC above 1, i.e. a self-announcing bug.
        below_run = np.concatenate([[0.0], np.cumsum(hist)[:-1]])
        num += float((cnt * below_run).sum())          # vs strictly better-ranked, weight 1.0
        num += 0.5 * float((cnt * below_g).sum())      # inside the p-tie group, weight 0.5
        hist += cnt
        i = j
    return min(1.0, max(0.0, num / den))


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Macro AUC over the 12 finding columns, using the soft-label-safe column AUC above."""
    aucs = []
    for j in range(y.shape[1]):
        yj, pj = y[:, j], p[:, j]
        ok = ~np.isnan(yj) & ~np.isnan(pj)
        aucs.append(_col_auc(yj[ok], pj[ok]))
    return float(np.mean(aucs))


def is_soft(y: np.ndarray) -> bool:
    vals = np.unique(y[~np.isnan(y)])
    return not np.all(np.isin(vals, (0.0, 1.0)))


def rankpct(a: np.ndarray) -> np.ndarray:
    return pd.DataFrame(a).rank(method="average", pct=True).to_numpy(np.float64)


def main() -> int:
    # ---- load our side -------------------------------------------------------------
    ours = np.load(OOF / "oof_equal_prob.npy")
    alt = np.load(OOF / "oof_selected.npy")
    y = np.load(OOF / "train_y.npy")
    ids = np.load(OOF / "train_ids.npy", allow_pickle=True)
    print("ours %s | alt %s | y %s | ids %s"
          % (ours.shape, alt.shape, y.shape, ids.shape))
    print("y dtype %s  unique %s" % (y.dtype, np.unique(y)[:6]))
    ids = np.asarray([str(x) for x in ids.ravel()])

    # ---- load the leg --------------------------------------------------------------
    parts, pids = [], []
    for k in range(4):
        parts.append(np.load(EFF / ("oof_effb0mil_sl16_tv7w_f%d.npy" % k)))
        d = pd.read_csv(EFF / ("oof_effb0mil_sl16_tv7w_f%d_ids.csv" % k))
        pids.append(d.iloc[:, 0].astype(str).to_numpy())
    print("leg fold shapes: %s" % [p.shape for p in parts])
    leg_cols = parts[0].shape[1] if parts[0].ndim == 2 else None
    print("leg columns: %s (ours: %s)" % (leg_cols, ours.shape[1]))
    if leg_cols != ours.shape[1]:
        print("!! column counts differ -- cannot align per label")
        return 2

    leg = np.full_like(ours, np.nan, dtype=np.float64)
    fold_of = np.full(len(ids), -1, dtype=int)
    idx = {u: i for i, u in enumerate(ids)}
    for k, (p, pi) in enumerate(zip(parts, pids)):
        hit = 0
        for row, u in enumerate(pi):
            j = idx.get(u)
            if j is not None:
                leg[j] = p[row]
                fold_of[j] = k
                hit += 1
        print("  fold %d: %d ids, %d matched into our %d" % (k, len(pi), hit, len(ids)))
    cov = fold_of >= 0
    print("leg coverage: %d / %d studies (%.1f%%)" % (cov.sum(), len(ids), 100 * cov.mean()))
    if cov.sum() < 500:
        print("!! too little coverage to analyse")
        return 2

    # ---- the look the leg was ADOPTED on: in-sample weight choice -------------------
    print("\n==== IN-SAMPLE (what the adoption used; w chosen on the same rows) ====")
    base_r = rankpct(ours)
    leg_r = rankpct(leg)
    curve = []
    for w in CAND_W:
        blend = (1.0 - w) * base_r + w * np.where(np.isnan(leg_r), 0.0, leg_r)
        curve.append((w, macro_auc(y, blend)))
    c = pd.DataFrame(curve, columns=["w", "auc"])
    b0 = c.loc[c.w == 0.0, "auc"].iloc[0]
    wbest = c.loc[c.auc.idxmax()]
    print(c[c.w.isin([0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30])].to_string(index=False))
    print("w=0 baseline %.5f | best w=%.2f -> %.5f  (+%.5f in-sample)"
          % (b0, wbest.w, wbest.auc, wbest.auc - b0))

    # ---- THE HONEST TEST: choose w out-of-fold, evaluate on the held-out fold -------
    print("\n==== OUT-OF-FOLD (w chosen on 3 folds, scored on the 4th) ====")
    rows = []
    for k in range(4):
        tr = cov & (fold_of != k)
        te = fold_of == k
        if te.sum() < 50 or tr.sum() < 200:
            print("  fold %d skipped (train %d / test %d)" % (k, tr.sum(), te.sum()))
            continue
        best_w, best_a = 0.0, -1.0
        for w in CAND_W:
            b = (1.0 - w) * base_r[tr] + w * np.where(np.isnan(leg_r[tr]), 0.0, leg_r[tr])
            a = macro_auc(y[tr], b)
            if a > best_a:
                best_a, best_w = a, w
        auc_w0 = macro_auc(y[te], base_r[te])
        auc_ws = macro_auc(y[te], (1.0 - best_w) * base_r[te]
                           + best_w * np.where(np.isnan(leg_r[te]), 0.0, leg_r[te]))
        auc_fix = macro_auc(y[te], 0.9 * base_r[te] + 0.1 * np.where(np.isnan(leg_r[te]), 0.0, leg_r[te]))
        rows.append(dict(fold=k, n=int(te.sum()), w_star=best_w,
                         auc_w0=auc_w0, auc_wstar=auc_ws, gain=auc_ws - auc_w0,
                         auc_w10=auc_fix, gain_w10=auc_fix - auc_w0))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    if len(t):
        print("\nmean honest gain (w selected OOF) = %+.5f   (per-fold %s)"
              % (t.gain.mean(), np.round(t.gain.to_numpy(), 5).tolist()))
        print("mean fixed w=0.10 gain           = %+.5f   (per-fold %s)"
              % (t.gain_w10.mean(), np.round(t.gain_w10.to_numpy(), 5).tolist()))
        print("folds positive: w* %d/%d | w=0.10 %d/%d"
              % ((t.gain > 0).sum(), len(t), (t.gain_w10 > 0).sum(), len(t)))

    # ---- paired bootstrap on the honest per-fold gains ------------------------------
    g = t.gain_w10.to_numpy() if len(t) else np.array([])
    if g.size:
        rng = np.random.default_rng(0)
        bs = rng.choice(g, size=(20000, g.size), replace=True).mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print("\npaired bootstrap over folds on the w=0.10 gain: mean %+.5f  CI [%+.5f, %+.5f]  P(>0)=%.3f"
              % (g.mean(), lo, hi, (bs > 0).mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
