"""How much does per-label weight tuning on one half of the data cost you on the other half?

WHY THIS EXISTS
---------------
The user's standing instruction: reading only the public leaderboard overfits the public leaderboard,
so local cross-validation must carry the generalisation argument.  The public board is a *subsample*
of the hidden set (~400 studies, B11) while the medal is decided on the private portion, so tuning on
the displayed value optimises for a sample that will not be used -- and with board noise at about
0.008 against effects of 0.001-0.003 (B12), any selection between candidates that differ by 0.001 is
selection on noise.

For the ENSEMBLE configuration we cannot do this locally at all: the routing and calibrator knobs are
blends against stage 1, and stage 1 is the arm measured to be leaky (A93), so no legitimate OOF
exists for them.  That is precisely why the hedged arm exists.

What we CAN do is measure the MECHANISM, using two arms that do have genuine out-of-fold predictions
(A5 and Rad).  Simulate the public/private partition by splitting the cohort in half:

  * fit the blend weight on half A, evaluate on half B   -> the honest number
  * fit and evaluate on half B                            -> what "tuning on the same split" reports
  * the difference is the optimism that public-board tuning buys

Three fits are compared, because the policy question is whether PER-LABEL tuning is worse than one
shared weight:

  1. one global weight, fitted on the fitting half
  2. twelve per-label weights, fitted on the fitting half
  3. (reference) the best possible per-label weight, fitted on the evaluation half itself

If (2) transfers worse than (1), per-label tuning is the overfitting surface, and a hedge that
softens or flattens the per-label map is justified by measurement rather than by intuition.  That is
the argument C57 made from the leaderboard; this replaces it with a number.

Usage:
    python scripts/tuning_transfer.py --reps 20
"""
import argparse
import csv
import sys

import numpy as np

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def avg_ranks(x):
    x = np.asarray(x, dtype=np.float64)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    sp = x[o]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def rank01(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 1:
        return np.full_like(x, 0.5)
    return (avg_ranks(x) - 1.0) / (len(x) - 1.0)


def auc(y, p):
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = avg_ranks(p)
    return float((r[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="downloads/research/oof/v52_oof.csv")
    ap.add_argument("--labels", default="downloads/research/oof/report_labels_v2.csv")
    ap.add_argument("--a5", default="downloads/research/a5oof_full/speed_a5_raw.npz")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--eval-size", type=int, default=1000,
                    help="held-out studies per split; held CONSTANT across the sweep so that the "
                         "evaluation noise does not change as the fitting set grows")
    ap.add_argument("--fit-sizes", default="100,200,400,870,1600")
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    fold, rad = {}, {}
    with open(args.folds, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            u = r["StudyInstanceUID"]
            fold[u] = int(r["fold"])
            rad[u] = np.array([float(r[t]) for t in TARGETS])
    rows = list(csv.DictReader(open(args.labels, newline="", encoding="utf-8")))
    ids_l = [r["StudyInstanceUID"] for r in rows]
    Y = np.zeros((len(rows), 12)); M = np.zeros((len(rows), 12), dtype=bool)
    for i, r in enumerate(rows):
        for j, t in enumerate(TARGETS):
            v = (r.get(t + "__verdict") or "").strip().upper()
            if v == "YES":
                Y[i, j], M[i, j] = 1.0, True
            elif v == "NO":
                Y[i, j], M[i, j] = 0.0, True
    d = np.load(args.a5, allow_pickle=True)
    uids = [str(u) for u in d["study_uids"]]
    raw = np.asarray(d["raw_probabilities"], dtype=np.float64)
    pos_a = {u: i for i, u in enumerate(uids)}

    pos_l = {u: i for i, u in enumerate(ids_l)}
    sel = [u for u in ids_l if u in rad and u in pos_a]
    A_raw = np.array([raw[fold[u]][pos_a[u]] for u in sel])
    R_raw = np.array([rad[u] for u in sel])
    Ys = np.array([Y[pos_l[u]] for u in sel])
    Ms = np.array([M[pos_l[u]] for u in sel])
    print("studies: %d  (A5 genuine OOF vs Rad genuine OOF, report labels)" % len(sel))

    rng = np.random.default_rng(args.seed)
    n = len(sel)
    fit_sizes = [int(x) for x in str(args.fit_sizes).split(",")]
    print("\nThis is the question that matters: the OPTIMISM of tuning a per-label weight depends on")
    print("how many studies the tuning set has -- and the public board is only ~400 studies (B11),")
    print("while the frontier tuned on 870 OOF / 58 gold.  The evaluation set is held at %d studies."
          % args.eval_size)
    print()
    print("  %-9s %-11s %-11s %-11s %-11s" % ("fit n", "honest", "in-sample", "optimism", "per-label"))
    print("  %-9s %-11s %-11s %-11s %-11s" % ("", "", "", "", "vs shared"))
    for nf in fit_sizes:
        if nf + args.eval_size > n:
            continue
        honest, insample, shared = [], [], []
        for rep in range(args.reps):
            perm = rng.permutation(n)
            A, B = perm[:nf], perm[nf:nf + args.eval_size]
            ba = (np.column_stack([rank01(A_raw[A, j]) for j in range(12)]),
                  np.column_stack([rank01(R_raw[A, j]) for j in range(12)]))
            bb = (np.column_stack([rank01(A_raw[B, j]) for j in range(12)]),
                  np.column_stack([rank01(R_raw[B, j]) for j in range(12)]))
            # one shared weight, fitted on the fitting set
            gsc = []
            for a in GRID:
                v = [auc(Ys[A, j][Ms[A, j]], ((1 - a) * ba[0][:, j] + a * ba[1][:, j])[Ms[A, j]])
                     for j in range(12) if Ms[A, j].sum() >= 20 and (~Ms[A, j]).sum() >= 0]
                v = [x for x in v if not np.isnan(x)]
                gsc.append(np.mean(v) if v else np.nan)
            if np.all(np.isnan(gsc)):
                continue
            g = GRID[int(np.nanargmax(gsc))]
            gh = [auc(Ys[B, j][Ms[B, j]], ((1 - g) * bb[0][:, j] + g * bb[1][:, j])[Ms[B, j]])
                  for j in range(12) if Ms[B, j].sum() >= 20]
            gh = [x for x in gh if not np.isnan(x)]
            if not gh:
                continue
            shared.append(float(np.mean(gh)))
            hl, il = [], []
            for j in range(12):
                mA, mB = Ms[A, j], Ms[B, j]
                if mA.sum() < 15 or mB.sum() < 15:
                    continue
                if len(set(Ys[A, j][mA].tolist())) < 2 or len(set(Ys[B, j][mB].tolist())) < 2:
                    continue
                sc = [auc(Ys[A, j][mA], ((1 - a) * ba[0][:, j] + a * ba[1][:, j])[mA]) for a in GRID]
                if all(np.isnan(x) for x in sc):
                    continue
                a_star = GRID[int(np.nanargmax(sc))]
                h = auc(Ys[B, j][mB], ((1 - a_star) * bb[0][:, j] + a_star * bb[1][:, j])[mB])
                sc2 = [auc(Ys[B, j][mB], ((1 - a) * bb[0][:, j] + a * bb[1][:, j])[mB]) for a in GRID]
                sc2 = [x for x in sc2 if not np.isnan(x)]
                if np.isnan(h) or not sc2:
                    continue
                hl.append(h)
                il.append(float(np.max(sc2)))
            if hl:
                honest.append(float(np.mean(hl)))
                insample.append(float(np.mean(il)))
        print("  %-9d %-11.4f %-11.4f %+-11.4f %+-11.4f"
              % (nf, np.mean(honest), np.mean(insample),
                 np.mean(insample) - np.mean(honest), np.mean(honest) - np.mean(shared)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
