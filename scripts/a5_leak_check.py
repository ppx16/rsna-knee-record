"""The pre-registered leakage check for the A5 OOF (FACTS A97 clause 1).

THE QUESTION
------------
`raw[k]` is the prediction of `m_f{k}.pt` -- the checkpoint whose own loader asserts the file/fold
identity.  If that model was trained with fold k held out, then `raw[k]` over fold-k studies is a
genuine out-of-fold prediction and can be used as an instrument.  If it was not, every conclusion
built on it inherits the same leak that invalidated the D2 parent (A93).

At n=48 the A5 fold-0 slice scored 0.9213 against the Rad arm's genuine published OOF at 0.8323, a
gap of 0.089 -- the same magnitude as the parent's measured leak of 0.097.  n=48 cannot separate
"genuinely stronger arm" from "also leaky", so the check was pre-registered and deferred to the full
run rather than guessed.

TWO INDEPENDENT TESTS, because either alone can mislead:

  1. **Against a known-genuine OOF on the SAME studies.**  The Rad arm's `v52_oof.csv` is verified
     genuine (it reproduces the audit's own `rad_gold_macro_auc` exactly).  A large, one-sided
     advantage for `raw[k]` here is the signature of leakage.
  2. **Against the sibling fold models on the SAME studies.**  `raw[k]` is held out from fold k; the
     other four are not.  If the fold field means what it says, the held-out model must score WORSE
     than the average of the models that saw those studies.  At n=48 this gap was +0.041, which is
     real but small -- the full run settles it.

A positive result on both means the fold discipline is real.  Note that test 2 can pass while test 1
fails: a small holdout effect plus a generally strong arm is also consistent with all five models
having seen most of the data.

Usage:
    python scripts/a5_leak_check.py --raw downloads/research/a5oof_full/speed_a5_raw.npz --fold 0
"""
import argparse
import csv
import sys

import numpy as np

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def avg_ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sp = x[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


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


def read_labels(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    ids = np.array([r["StudyInstanceUID"] for r in rows], dtype=object)
    Y = np.zeros((len(rows), 12))
    M = np.zeros((len(rows), 12), dtype=bool)
    for i, r in enumerate(rows):
        for j, t in enumerate(TARGETS):
            v = (r.get(t + "__verdict") or "").strip().upper()
            if v == "YES":
                Y[i, j], M[i, j] = 1.0, True
            elif v == "NO":
                Y[i, j], M[i, j] = 0.0, True
    return ids, Y, M


def macro(Y, M, P):
    per = [auc(Y[M[:, j], j], P[M[:, j], j]) for j in range(12)]
    bad = [v for v in per if not np.isnan(v) and not (0 <= v <= 1)]
    if bad:
        raise AssertionError("AUC outside [0,1]: %r" % bad[:4])
    ok = [v for v in per if not np.isnan(v)]
    return (float(np.mean(ok)) if ok else float("nan")), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--fold", default="0")
    ap.add_argument("--labels", default="downloads/research/oof/report_labels_v2.csv")
    ap.add_argument("--rad", default="downloads/research/oof/v52_oof.csv")
    ap.add_argument("--folds-csv", default="downloads/research/oof/v52_oof.csv")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    d = np.load(args.raw, allow_pickle=True)
    uids = [str(u) for u in d["study_uids"]]
    raw = np.asarray(d["raw_probabilities"], dtype=np.float64)
    print("A5 raw: %s  studies=%d" % (raw.shape, len(uids)))

    fold = {}
    rad = {}
    with open(args.folds_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fold[r["StudyInstanceUID"]] = r["fold"]
    with open(args.rad, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rad[r["StudyInstanceUID"]] = np.array([float(r[t]) for t in TARGETS])

    k = int(args.fold)
    in_fold = [i for i, u in enumerate(uids) if str(fold.get(u)) == str(k)]
    print("fold %d studies present: %d" % (k, len(in_fold)))
    if len(in_fold) < 100:
        raise SystemExit("!! too few studies in fold %d to decide anything" % k)

    ids_l, Y, M = read_labels(args.labels)
    pos = {v: i for i, v in enumerate(ids_l)}
    sel = [i for i in in_fold if uids[i] in pos and uids[i] in rad]
    print("scored on %d of them (having both report labels and the Rad OOF)\n" % len(sel))
    ids = [uids[i] for i in sel]
    Ys = np.array([Y[pos[u]] for u in ids])
    Ms = np.array([M[pos[u]] for u in ids])

    def prep(a):
        return np.column_stack([rank01(a[:, j]) for j in range(12)])

    a5_held = prep(raw[k][sel])
    a5_others = prep(np.mean([raw[j][sel] for j in range(5) if j != k], axis=0))
    radm = prep(np.array([rad[u] for u in ids]))

    m_held, per_held = macro(Ys, Ms, a5_held)
    m_oth, per_oth = macro(Ys, Ms, a5_others)
    m_rad, per_rad = macro(Ys, Ms, radm)
    print("%-42s macro=%.4f" % ("A5 held out on fold %d (raw[%d])" % (k, k), m_held))
    print("%-42s macro=%.4f" % ("A5 mean of the other 4 folds", m_oth))
    print("%-42s macro=%.4f" % ("Rad arm, genuine published OOF", m_rad))
    print()
    print("TEST 2 (fold discipline): held-out minus siblings = %+.4f" % (m_held - m_oth))
    print("TEST 1 (leakage vs a genuine OOF): A5 minus Rad  = %+.4f" % (m_held - m_rad))

    rng = np.random.default_rng(args.seed)
    n = len(sel)
    boot = [rng.integers(0, n, n) for _ in range(args.boot)]
    d1 = np.array([macro(Ys[b], Ms[b], a5_held[b])[0] - macro(Ys[b], Ms[b], radm[b])[0]
                   for b in boot])
    d1 = d1[~np.isnan(d1)]
    d2 = np.array([macro(Ys[b], Ms[b], a5_held[b])[0] - macro(Ys[b], Ms[b], a5_others[b])[0]
                   for b in boot])
    d2 = d2[~np.isnan(d2)]
    lo1, hi1 = np.percentile(d1, [2.5, 97.5])
    lo2, hi2 = np.percentile(d2, [2.5, 97.5])
    print("   A5-Rad   95%% CI [%+.4f, %+.4f]" % (lo1, hi1))
    print("   A5-sib   95%% CI [%+.4f, %+.4f]" % (lo2, hi2))
    print()
    print("per-label, A5(held) vs Rad, largest differences:")
    for j, t in enumerate(TARGETS):
        print("   %-18s A5=%.4f Rad=%.4f  delta=%+.4f" % (t, per_held[j], per_rad[j],
                                                          per_held[j] - per_rad[j]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
