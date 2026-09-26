# -*- coding: utf-8 -*-
"""Would one of OUR arms add anything to a parent-quality blend?

WHY THIS GATES THE TRAINING DECISION
    scripts/arm_parity.py measured our best arm at 0.8769 against the frontier's own DINOv2
    seed-members at 0.8676-0.8697 -- so our training is not behind that family (A106's 0.04 gap
    is about A5, the one DINOv3 model).  That removes the reason the training line was closed,
    but it does not by itself make training worth 27 GPU-hours: what matters is whether a member
    like ours IMPROVES a strong blend, which is D14's question, not A106's.

    So this asks it directly, out-of-fold, on the best honest instrument we have (the 0.9151
    Rad-free surrogate from oof_surrogate.py).  Our arm covers fold 0 only, so the base is
    restricted to fold-0 studies and the blend weight for our arm is chosen on one half and
    scored on the other, in both directions -- a weight fitted on the half it is scored on would
    buy roughly +0.001 of nothing (HANDOFF 86.2).

    Two numbers decide it:
      * the DELTA -- does adding our arm help at all;
      * the RANK CORRELATION against the base, which is the diversity diagnostic this repo uses
        (A5 vs Rad share 0.8061 residual correlation, and that is the bar a new arm must beat).

Usage:
    python scripts/member_add_test.py
    python scripts/member_add_test.py --arm cn0@f0 --boot 400
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import pathlib
import re
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from calibrator_probe import read_labels  # noqa: E402

OOF = ROOT / "downloads" / "oof"
A5 = ROOT / "downloads" / "research" / "a5oof_full" / "speed_a5_raw.npz"
FOLDS = ROOT / "downloads" / "research" / "oof" / "v52_oof.csv"
LABELS_CSV = ROOT / "downloads" / "research" / "oof" / "report_labels_v2.csv"
A5_W = 0.70          # the fold-wise-honest weight inside the surrogate (oof_surrogate.py)


def rank01(a):
    out = np.empty_like(a, dtype=np.float64)
    for j in range(a.shape[1]):
        out[:, j] = (rankdata(a[:, j]) - 1) / max(len(a) - 1, 1)
    return out


def macro(Y, M, P):
    from sklearn.metrics import roc_auc_score
    vals = []
    for j in range(Y.shape[1]):
        m = M[:, j]
        y = Y[m, j].astype(int)
        if len(np.unique(y)) < 2:
            continue
        vals.append(roc_auc_score(y, P[m, j]))
    return float(np.mean(vals))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="cn0@f0")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260926)
    a = ap.parse_args(argv)

    lab_ids, Y, M = read_labels(str(LABELS_CSV))
    pos_lab = {u: i for i, u in enumerate(lab_ids)}
    fold = {}
    with open(FOLDS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fold[r["StudyInstanceUID"]] = int(r["fold"])

    # our arm
    ours = None
    for pat in (str(ROOT / "downloads/cv/**/raptor_oof_*.npz"),
                str(ROOT / "downloads/rent/**/raptor_oof_*.npz")):
        for p in glob.glob(pat, recursive=True):
            m = re.search(r"raptor_oof_([A-Za-z0-9]+?)(?:_(ep\d+|epens|swa))?_fold(\d+)\.npz$",
                          os.path.basename(p))
            if not m or m.group(2):
                continue
            if "%s@f%s" % (m.group(1), m.group(3)) == a.arm:
                d = np.load(p, allow_pickle=True)
                ours = ([str(u) for u in d["ids"]], np.asarray(d["pred"], dtype=np.float64))
    if ours is None:
        raise SystemExit("!! arm %r not found among the fold OOF npz files" % a.arm)
    o_ids, o_P = ours

    # base components: A5 (fold-correct) and the public20 seed arms
    d = np.load(A5, allow_pickle=True)
    a5_ids = [str(u) for u in d["study_uids"]]
    raw = np.asarray(d["raw_probabilities"], dtype=np.float64)
    a5 = np.full((len(a5_ids), 12), np.nan)
    for i, u in enumerate(a5_ids):
        k = fold.get(u)
        if k is not None and k < raw.shape[0]:
            a5[i] = raw[k][i]
    pos_a5 = {u: i for i, u in enumerate(a5_ids)}

    p20_ids = [str(u) for u in np.load(OOF / "train_ids.npy", allow_pickle=True)]
    p20 = np.load(OOF / "oof_arm_prob.npy").astype(np.float64)
    pos_p20 = {u: i for i, u in enumerate(p20_ids)}

    # ⚠️ Do NOT require the two fold systems to agree.  Their fold maps are different partitions
    # (ours comes from rsna-knee-cv-folds, A5/public20 from v52_oof.csv), and intersecting them
    # leaves 181 studies -- which is how A106 came to compare us to A5 on 181 studies.  Honesty is
    # PER FAMILY: our arm's file is `..._fold0.npz`, so every study in it was held out by OUR
    # fold-0 model; A5's `raw[k]` is honest because m_f{k} never trained on it under v52's split;
    # public20's OOF is honest under its own split.  Mixing three honest predictions is honest.
    ids = [u for u in o_ids if u in pos_lab and u in pos_a5 and u in pos_p20
           and np.isfinite(a5[pos_a5[u]]).all()]
    print("arm %s: %d studies, each held out by its own family's fold discipline" % (a.arm, len(ids)))
    Ys = np.array([Y[pos_lab[u]] for u in ids])
    Ms = np.array([M[pos_lab[u]] for u in ids])
    A = rank01(np.array([a5[pos_a5[u]] for u in ids]))
    idx = [pos_p20[u] for u in ids]
    P20 = np.mean([rank01(p20[idx, j, :]) for j in range(p20.shape[1])], axis=0)
    O = rank01(np.array([o_P[list(o_ids).index(u)] for u in ids]))

    base = A5_W * A + (1 - A5_W) * P20
    print("base (A5 %.2f + public20 %.2f): %.4f" % (A5_W, 1 - A5_W, macro(Ys, Ms, base)))
    print("our arm alone:                  %.4f" % macro(Ys, Ms, O))

    # diversity diagnostic: per-label Spearman of our arm against the base
    cors = [spearmanr(O[:, j], base[:, j]).statistic for j in range(12)]
    print("rank correlation vs base: mean %.4f (min %.4f, max %.4f)"
          % (float(np.mean(cors)), float(np.min(cors)), float(np.max(cors))))
    print("  (A5 vs Rad share a 0.8061 residual correlation -- the bar a new arm has to beat)")

    # honest weight: choose on one half, score on the other, both directions
    grid = np.arange(0.0, 0.501, 0.05)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(ids))
    halves = [perm[:len(ids) // 2], perm[len(ids) // 2:]]
    P_out = np.zeros_like(base)
    chosen = []
    for fit_ix, ev_ix in ((halves[0], halves[1]), (halves[1], halves[0])):
        sc = [macro(Ys[fit_ix], Ms[fit_ix], (1 - w) * base[fit_ix] + w * O[fit_ix]) for w in grid]
        w_best = float(grid[int(np.nanargmax(sc))])
        chosen.append(w_best)
        P_out[ev_ix] = (1 - w_best) * base[ev_ix] + w_best * O[ev_ix]
    m_base = macro(Ys, Ms, base)
    m_add = macro(Ys, Ms, P_out)
    print("\nhonest member-addition test (weight chosen on the other half)")
    print("  chosen weights: %s" % ", ".join("%.2f" % w for w in chosen))
    print("  base %.4f  ->  base + our arm %.4f   delta %+.4f" % (m_base, m_add, m_add - m_base))

    b = rng.integers(0, len(ids), (a.boot, len(ids)))
    dd = []
    for ix in b:
        x, y = macro(Ys[ix], Ms[ix], P_out[ix]), macro(Ys[ix], Ms[ix], base[ix])
        if not (np.isnan(x) or np.isnan(y)):
            dd.append(x - y)
    dd = np.array(dd)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    print("  95%% CI [%+.4f, %+.4f]  P(delta>0)=%.2f" % (lo, hi, float((dd > 0).mean())))
    print("\n=> if the delta's CI covers 0, training more of this family buys the ensemble nothing")
    print("   even though the arm beats the frontier's own members solo -- that is D14's point.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
