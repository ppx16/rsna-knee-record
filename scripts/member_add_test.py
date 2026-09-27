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
import itertools
import os
import pathlib
import re
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from calibrator_probe import read_labels, TARGETS  # noqa: E402

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
    ap.add_argument("--arm", action="append", default=None,
                    help="repeatable. ONE arm answers 'does it help?'; TWO OR MORE answer the "
                         "question that actually matters before integrating both -- 'is the second "
                         "one additive GIVEN the first?', because the gate is single-member and two "
                         "arms of the same family are usually not additive.")
    ap.add_argument("--strong", action="store_true",
                    help="add the two published Rad heads to the base and choose every blend "
                         "weight on the fit half -- a parent-quality instrument, and the honest "
                         "upper bound on what this arm can buy")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--curve", action="store_true",
                    help="also print the descriptive dose-response on the fixed base (shape only; "
                         "the in-sample argmax is NOT a decision)")
    ap.add_argument("--seed", type=int, default=20260926)
    a = ap.parse_args(argv)
    arms_wanted = a.arm or ["cn0@f0"]

    lab_ids, Y, M = read_labels(str(LABELS_CSV))
    pos_lab = {u: i for i, u in enumerate(lab_ids)}
    fold = {}
    with open(FOLDS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fold[r["StudyInstanceUID"]] = int(r["fold"])

    # our arms (one per --arm; the gate is only meaningful for one, the JOINT question needs N)
    found = {}
    for pat in (str(ROOT / "downloads/cv/**/raptor_oof_*.npz"),
                str(ROOT / "downloads/rent/**/raptor_oof_*.npz"),
                # the A-lite paired arms land here rather than under cv/ (they are written by
                # kernels/kernel-a5lite*, not by the cvtrain pipeline).  NOTE these use a DIFFERENT
                # fold system (ppxl16/rsna-knee-cv-folds, 881 studies) from the base's v52_oof.csv,
                # which is fine -- the id intersection handles it, and honesty is per-family.
                str(ROOT / "downloads/a5lite_*/raptor_oof_*.npz")):
        for p in glob.glob(pat, recursive=True):
            m = re.search(r"raptor_oof_([A-Za-z0-9]+?)(?:_(ep\d+|epens|swa))?_fold(\d+)\.npz$",
                          os.path.basename(p))
            if not m or m.group(2):
                continue
            found["%s@f%s" % (m.group(1), m.group(3))] = p
    ours = []
    for want in arms_wanted:
        p = found.get(want)
        if p is None:
            raise SystemExit("!! arm %r not found among the fold OOF npz files (have: %s)"
                             % (want, ", ".join(sorted(found))))
        d = np.load(p, allow_pickle=True)
        ours.append((want, [str(u) for u in d["ids"]], np.asarray(d["pred"], dtype=np.float64)))
    o_ids = [u for u in ours[0][1]]

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

    # the two published Rad heads, for the strong-base variant below
    rads = []
    for name in ("v52_oof.csv", "v52_e11_oof.csv"):
        rr = {}
        for r in csv.DictReader(open(ROOT / "downloads/research/oof" / name, newline="",
                                     encoding="utf-8")):
            rr[r["StudyInstanceUID"]] = np.array([float(r[t]) for t in TARGETS])
        rads.append(rr)

    # ⚠️ Do NOT require the two fold systems to agree.  Their fold maps are different partitions
    # (ours comes from rsna-knee-cv-folds, A5/public20 from v52_oof.csv), and intersecting them
    # leaves 181 studies -- which is how A106 came to compare us to A5 on 181 studies.  Honesty is
    # PER FAMILY: our arm's file is `..._fold0.npz`, so every study in it was held out by OUR
    # fold-0 model; A5's `raw[k]` is honest because m_f{k} never trained on it under v52's split;
    # public20's OOF is honest under its own split.  Mixing three honest predictions is honest.
    ids = [u for u in o_ids if all(u in s for s in [set(o[1]) for o in ours])
           and u in pos_lab and u in pos_a5 and u in pos_p20
           and np.isfinite(a5[pos_a5[u]]).all()]
    print("arms %s: %d studies, each held out by its own family's fold discipline"
          % (", ".join(o[0] for o in ours), len(ids)))
    if len(ours) > 1:
        print("  (intersection across all requested arms -- a joint number on a smaller n)")
    Ys = np.array([Y[pos_lab[u]] for u in ids])
    Ms = np.array([M[pos_lab[u]] for u in ids])
    A = rank01(np.array([a5[pos_a5[u]] for u in ids]))
    idx = [pos_p20[u] for u in ids]
    P20 = np.mean([rank01(p20[idx, j, :]) for j in range(p20.shape[1])], axis=0)
    Os = []
    for _n, a_ids, a_P in ours:
        pos = {u: i for i, u in enumerate(a_ids)}
        Os.append(rank01(np.array([a_P[pos[u]] for u in ids])))
    O = Os[0]

    base = A5_W * A + (1 - A5_W) * P20
    print("base (A5 %.2f + public20 %.2f): %.4f" % (A5_W, 1 - A5_W, macro(Ys, Ms, base)))
    for (nm_, _i, _P), Oi in zip(ours, Os):
        print("arm %-14s alone:            %.4f" % (nm_, macro(Ys, Ms, Oi)))

    if a.curve:
        # DESCRIPTIVE dose-response on the fixed 0.917 base: what is still on the table between a
        # small member dose and the optimum?  No fitting happens here, so this is a shape, not a
        # choice -- the CHOICE is validated separately by the two-half protocol below.
        print("\ndose-response on the fixed base (descriptive; no fitting):")
        print("  %-6s %s" % ("w", "macro"))
        best_w, best_v = 0.0, macro(Ys, Ms, base)
        for w in np.arange(0.0, 0.501, 0.025):
            v = macro(Ys, Ms, (1 - w) * base + w * O)
            if v > best_v:
                best_w, best_v = float(w), v
            print("  %-6.3f %.4f  %s" % (w, v, "+" * int(round((v - 0.90) * 2000))))
        print("  argmax w=%.3f -> %.4f  (gain %+.4f)"
              % (best_w, best_v, best_v - macro(Ys, Ms, base)))
        print("  ⚠️ this argmax is IN-SAMPLE and must not be used as a decision; it only shows the")
        print("     SHAPE.  The honest choice is the two-half number below.")

    if a.strong:
        # ---- the strong-base variant -------------------------------------------------------
        # The fixed base above is A5 + public20 only (0.917).  The parent is ~0.94 and also
        # carries the Rad heads, so a member-add gain measured at 0.917 is an UPPER bound on
        # what the same arm would buy in the real stack -- exactly the A109/A112 pattern.
        # Here the base gets the two published Rad heads as well AND its weights are chosen
        # on the fit half (a well-set base is the A127 lesson: never measure an addition on a
        # mis-set blend), and the member weight is chosen jointly with them.  The honest
        # comparison is "best base found on the fit half" vs "best base + member found on the
        # same half", both scored on the other half.
        Rk = rank01(np.array([rads[0][u] for u in ids]))
        Ek = rank01(np.array([rads[1][u] for u in ids]))
        nm = len(Os)
        # one member keeps the shipped 0.05 grid (so the cn0 control reproduces bit-for-bit);
        # a joint test coarsens to 0.1, because the grid is multiplicative in the member count.
        mvals = np.arange(0.0, 0.501, 0.05 if nm == 1 else 0.1)
        grid = []
        for wa in np.arange(0.3, 0.91, 0.1):
            for wr in np.arange(0.0, 0.31, 0.1):
                for we in np.arange(0.0, 0.21, 0.1):
                    for wms in itertools.product(mvals, repeat=nm):
                        rest = 1.0 - wa - wr - we - float(sum(wms))
                        if rest < -1e-9:
                            continue
                        grid.append((float(wa), float(wr), float(we))
                                    + tuple(float(x) for x in wms) + (max(0.0, rest),))
        nomem = [p for p in grid if not any(p[3:3 + nm])]
        print("\nstrong base: %d candidate weightings over {A5, Rad-v52, Rad-e11, public20, %s}"
              % (len(grid), ", ".join(o[0] for o in ours)))
        print("  grid step for member weights: %.2f ; %d weightings use NO member"
              % (0.05 if nm == 1 else 0.1, len(nomem)))

        def blend(p, sl):
            out = p[0] * A[sl] + p[1] * Rk[sl] + p[2] * Ek[sl] + p[3 + nm] * P20[sl]
            for w, Oi in zip(p[3:3 + nm], Os):
                out = out + w * Oi[sl]
            return out

        rng2 = np.random.default_rng(a.seed)
        perm2 = rng2.permutation(len(ids))
        h = [perm2[:len(ids) // 2], perm2[len(ids) // 2:]]
        P_b = np.zeros_like(base)
        P_m = np.zeros_like(base)
        chosen = []
        for fit_ix, ev_ix in ((h[0], h[1]), (h[1], h[0])):
            sc_base = [macro(Ys[fit_ix], Ms[fit_ix], blend(p, fit_ix)) for p in nomem]
            gb = nomem
            pb = gb[int(np.nanargmax(sc_base))]
            sc_mem = [macro(Ys[fit_ix], Ms[fit_ix], blend(p, fit_ix)) for p in grid]
            pm = grid[int(np.nanargmax(sc_mem))]
            chosen.append((pb, pm))
            P_b[ev_ix] = blend(pb, ev_ix)
            P_m[ev_ix] = blend(pm, ev_ix)
        mb = macro(Ys, Ms, P_b)
        mm = macro(Ys, Ms, P_m)
        print("  chosen on half A: base %s | base+member %s"
              % (tuple(round(x, 2) for x in chosen[0][0]), tuple(round(x, 2) for x in chosen[0][1])))
        print("  chosen on half B: base %s | base+member %s"
              % (tuple(round(x, 2) for x in chosen[1][0]), tuple(round(x, 2) for x in chosen[1][1])))
        print("  honest strong base        : %.4f" % mb)
        print("  honest strong base+member : %.4f   delta %+.4f" % (mm, mm - mb))
        dd2 = []
        for ix in rng2.integers(0, len(ids), (a.boot, len(ids))):
            x, y = macro(Ys[ix], Ms[ix], P_m[ix]), macro(Ys[ix], Ms[ix], P_b[ix])
            if not (np.isnan(x) or np.isnan(y)):
                dd2.append(x - y)
        dd2 = np.array(dd2)
        lo2, hi2 = np.percentile(dd2, [2.5, 97.5])
        print("  95%% CI [%+.4f, %+.4f]  P(delta>0)=%.2f"
              % (lo2, hi2, float((dd2 > 0).mean())))
        return 0

    # diversity diagnostic: per-label Spearman of our arm against the base
    if len(Os) > 1:
        raise SystemExit("!! the non-strong path is single-member by construction; use --strong "
                         "for a joint test (it is the only variant with joint weight fitting)")
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
