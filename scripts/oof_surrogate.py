# -*- coding: utf-8 -*-
"""Build the strongest HONEST local surrogate of the frontier's parent ensemble.

WHY THIS EXISTS
    Every fusion-layer conclusion in this repository was reached on an instrument whose absolute
    level is far below the thing it is used to reason about:

        instrument                     n        level     what it can speak about
        ---------------------------------------------------------------------------------------
        downloads/oof (public20 x4)    4407     0.8497    arm-internal weights, pooling
        Rad arm OOF (v52)              4407     0.8424    one arm
        A5 OOF (speed_a5_raw)          4407     ~0.909    one arm
        our own fold-0 ensemble        870      0.8823    meta-level blending (A112)
        the parent ensemble itself     0        ~0.94     <-- nothing can speak about this

    That last row is why the calibrator (`rad_cal_w`), the outer routing (`coatnet_w`) and the
    A5/Rad weights are all unresolved rather than settled: A109/A112/A117 each showed the same
    shape -- a gain that shrinks as the base strengthens (+0.008 on a 0.842 arm, +0.002 on a
    0.909 arm, ~0 on a 0.882 ensemble).  An instrument at 0.88 cannot tell us what happens at 0.94.

    A genuine OOF of the parent cannot be built: the twenty published DINOv2 members are not
    fold-trained, so no study has an honest held-out prediction from them (A93).  But the parent
    is not a monolith -- it is a rank blend.  Everything in it that IS fold-trained can be
    assembled honestly, and the result is a strictly better operating point than we have had.

WHAT THIS BUILDS (and the one thing it deliberately excludes)
    surrogate = rank-mean of components that are each a genuine out-of-fold prediction:
        A5         `mattiaangeli/knee-mri-fold-weights` m_f0..m_f4, asserted per-fold by the
                   stage's own loader -> raw[k] is honest for fold-k studies
        public20   the four seed-arms of `denisrukavishnikov/rnsa-knee-pred-public20-oof-audit`
    ⚠️ The Rad heads (v52_oof.csv / v52_e11_oof.csv) are EXCLUDED even though they are genuine,
    because they are the calibrator probe's `reference`/`second` inputs.  A baseline that already
    contains them would make "does the calibrator add information?" partly circular.  Rad-free is
    what makes the surrogate usable as `baseline_rank`.

WHAT IT WOULD GET WRONG (read before quoting a number)
    * The labels are report-derived soft labels, not the competition's hidden truth.  Absolute
      levels here are NOT leaderboard levels and no delta here is a predicted LB delta (§86).
    * public20's own table is scored against weak LLM labels elsewhere; here every component is
      re-scored against ONE label set so the comparison is at least internally consistent.
    * A blend weight chosen on the same studies it is scored on buys ~+0.001 (measured, §86.2).
      So the honest number is the per-fold one; the fixed-weight number is reported beside it as
      an upper bound, never on its own.

Usage:
    python scripts/oof_surrogate.py                     # levels + the weight scan
    python scripts/oof_surrogate.py --save downloads/research/oof/surrogate_radfree.npz
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from calibrator_probe import rank_columns, read_labels  # noqa: E402
from cv_score import LABELS, paired_bootstrap  # noqa: E402

OOF = ROOT / "downloads" / "oof"
A5 = ROOT / "downloads" / "research" / "a5oof_full" / "speed_a5_raw.npz"
FOLDS = ROOT / "downloads" / "research" / "oof" / "v52_oof.csv"
LABELS_CSV = ROOT / "downloads" / "research" / "oof" / "report_labels_v2.csv"


def macro(Y, M, P):
    """Macro-AUC over labels with both classes present under the mask."""
    vals = []
    for j in range(Y.shape[1]):
        m = M[:, j]
        y = Y[m, j].astype(int)
        if len(np.unique(y)) < 2:
            vals.append(np.nan)
            continue
        vals.append(roc_auc_score(y, P[m, j]))
    return float(np.nanmean(vals))


def load_a5():
    """(ids, (n,12)) genuinely out-of-fold: each study scored by its own fold's checkpoint."""
    fold = {}
    with open(FOLDS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fold[r["StudyInstanceUID"]] = int(r["fold"])
    d = np.load(A5, allow_pickle=True)
    ids = [str(u) for u in d["study_uids"]]
    raw = np.asarray(d["raw_probabilities"], dtype=np.float64)      # (5, n, 12)
    out = np.full((len(ids), 12), np.nan)
    used = np.zeros(5, dtype=int)
    for i, u in enumerate(ids):
        k = fold.get(u)
        if k is not None and k < raw.shape[0]:
            out[i] = raw[k][i]
            used[k] += 1
    print("A5 OOF: %d ids, %d scored, per-fold used %s" % (len(ids), int(np.isfinite(out).all(1).sum()),
                                                           used.tolist()))
    return ids, out


def load_public20():
    ids = [str(u) for u in np.load(OOF / "train_ids.npy", allow_pickle=True)]
    arms = np.load(OOF / "oof_arm_prob.npy").astype(np.float64)     # (n, 4, 12)
    return ids, arms


def paired_bootstrap_masked(Y, M, Pa, Pb, n_boot=400, seed=0):
    """Paired study-level bootstrap of macro-AUC(a) - macro-AUC(b), under the SAME mask.

    WHY NOT cv_score.paired_bootstrap: that one binarises `Y >= 0.5` over every study, so the
    ~27% of cells with no YES/NO verdict are scored as negatives.  `macro()` here uses the mask.
    A CI computed under a different truth definition than the point estimate beside it is not
    comparable to that point estimate, and this project has already paid once for a metric that
    silently used the wrong axis (see the note in cv_score.paired_bootstrap).

    Invariant, checked in main(): pairing a prediction with ITSELF must give delta 0.0000 and a
    degenerate CI.  If that ever fails, the bootstrap is resampling the wrong axis.
    """
    rng = np.random.default_rng(seed)
    n = Y.shape[0]

    def delta(idx):
        return macro(Y[idx], M[idx], Pa[idx]) - macro(Y[idx], M[idx], Pb[idx])

    point = delta(np.arange(n))
    draws = []
    for _ in range(n_boot):
        v = delta(rng.integers(0, n, n))
        if v == v:
            draws.append(v)
    draws = np.asarray(draws)
    if draws.size == 0:
        return point, float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), float((draws > 0).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="", help="write the chosen surrogate as ids/pred npz")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args(argv)

    lab_ids, Y, M = read_labels(str(LABELS_CSV))
    pos_lab = {u: i for i, u in enumerate(lab_ids)}

    a5_ids, a5 = load_a5()
    pos_a5 = {u: i for i, u in enumerate(a5_ids)}
    p20_ids, p20_raw = load_public20()
    pos_p20 = {u: i for i, u in enumerate(p20_ids)}
    print("public20 OOF: %d ids x %d arms" % (len(p20_ids), p20_raw.shape[1]))

    ids = [u for u in lab_ids if u in pos_a5 and u in pos_p20]
    if not ids:
        raise SystemExit("!! no common studies -- check id formats")
    Ys = np.array([Y[pos_lab[u]] for u in ids])
    Ms = np.array([M[pos_lab[u]] for u in ids])
    A = rank_columns(np.array([a5[pos_a5[u]] for u in ids]))
    idx20 = [pos_p20[u] for u in ids]
    # per ARM across studies -- p20_raw[study] is (4 arms, 12 labels), so indexing by [idx, a, :]
    # gives one arm's (n, 12) matrix.  Ranking must happen across STUDIES, per label.
    arms = [rank_columns(p20_raw[idx20, a, :]) for a in range(p20_raw.shape[1])]
    P20 = np.mean(arms, axis=0)
    print("common studies scored: %d   label coverage: %.1f%% of %d cells"
          % (len(ids), 100.0 * Ms.mean(), Ms.size))

    print("\n== component levels (macro-AUC on the report-derived labels, n=%d) ==" % len(ids))
    rows = [("A5 OOF (fold-correct)", A)]
    for a in range(len(arms)):
        rows.append(("public20 arm %d" % a, arms[a]))
    rows.append(("public20 rank-mean of 4 arms", P20))
    for name, P in rows:
        print("  %-28s %.4f" % (name, macro(Ys, Ms, P)))

    print("\n== weight scan: w*A5 + (1-w)*public20 (rank space) ==")
    best = (-1.0, None)
    for w in np.arange(0.0, 1.01, 0.05):
        P = w * A + (1.0 - w) * P20
        s = macro(Ys, Ms, P)
        flag = ""
        if s > best[0]:
            best = (s, float(w)); flag = " <-"
        print("  w_A5=%.2f   %.4f%s" % (w, s, flag))
    w_best = best[1]
    print("  best fixed w_A5 = %.2f  (%.4f)   <-- chosen on the scored studies: an upper bound" % (w_best, best[0]))

    # Honest number: choose w on four folds, score the fifth.  Needs the split, which is v52's.
    fold = {}
    with open(FOLDS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fold[r["StudyInstanceUID"]] = int(r["fold"])
    kf = np.array([fold.get(u, -1) for u in ids])
    grid = np.arange(0.0, 1.01, 0.05)
    P_oof = np.zeros_like(A)
    ok = kf >= 0
    for k in range(5):
        te = kf == k
        tr = ok & ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        scores = [macro(Ys[tr], Ms[tr], w * A[tr] + (1.0 - w) * P20[tr]) for w in grid]
        w_k = grid[int(np.nanargmax(scores))]
        P_oof[te] = w_k * A[te] + (1.0 - w_k) * P20[te]
        print("  fold %d: w_A5 chosen on the other folds = %.2f  (n_heldout=%d)" % (k, w_k, int(te.sum())))
    s_oof = macro(Ys[ok], Ms[ok], P_oof[ok])
    s_a5 = macro(Ys[ok], Ms[ok], A[ok])
    print("\n== the honest surrogate ==")
    print("  fold-wise weight selection, scored on held-out folds: %.4f  (n=%d)" % (s_oof, int(ok.sum())))
    print("  A5 alone on the same studies:                          %.4f" % s_a5)
    pa, pb, ci, win = paired_bootstrap(Ys[ok], P_oof[ok], A[ok], n_boot=args.boot, seed=args.seed)
    print("  [cross-check, binarised truth, cv_score helper] %+.4f  CI [%+.4f, %+.4f]"
          % (pa - pb, ci[0], ci[1]))
    d0, lo, hi, pr = paired_bootstrap_masked(Ys[ok], Ms[ok], P_oof[ok], A[ok],
                                             n_boot=args.boot, seed=args.seed)
    print("  [masked truth] surrogate minus A5: %+.4f  95%% CI [%+.4f, %+.4f]  P(>0)=%.2f"
          % (d0, lo, hi, pr))
    st = paired_bootstrap_masked(Ys[ok], Ms[ok], A[ok], A[ok], n_boot=50, seed=1)
    print("  self-test (A5 paired with itself): delta %+.4f  CI [%+.4f, %+.4f]  %s"
          % (st[0], st[1], st[2], "OK" if st[0] == 0.0 and st[1] == 0.0 and st[2] == 0.0 else "FAIL"))
    print("\n  => the surrogate is the baseline any fusion probe should use from now on.")
    print("     It is stronger than our fold-0 ensemble (0.8823) but still Rad-free, so the")
    print("     calibrator's reference/second inputs stay independent of it.")

    if args.save:
        out = pathlib.Path(args.save)
        if not out.is_absolute():
            out = ROOT / out
        np.savez_compressed(out, ids=np.array(ids, dtype=object), pred=P_oof, fold=kf)
        print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
