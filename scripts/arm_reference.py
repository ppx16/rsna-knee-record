"""Place our own trained arms on the same scale as the frontier's A5 arm.

WHY THE COMMON SET IS THE WHOLE POINT
-------------------------------------
Our arms are out-of-fold on OUR fold split (`dataset-cv-fold/fold_file.json`); the A5 slice
`raw[k]` is out-of-fold on the PUBLISHER'S split (the `fold` column of `v52_oof.csv`), which is a
different partition.  Comparing the two on either split alone would be wrong in one direction or
the other:

  * on our fold-0 studies, `raw[0]` is out-of-fold only for the subset that also sits in the
    publisher's fold 0 -- for the rest, `m_f0` was trained on them and would look unfairly strong;
  * on the publisher's fold-0 studies, our `cn0` is out-of-fold only for the subset in our fold 0.

So the comparison runs on the **intersection**, where BOTH arms are genuinely held out.  This is the
same discipline that caught the DINOv2 parent (FACTS A93): an arm scored on studies it saw is not an
arm, it is a lookup.

Both arms are scored against the SAME report-label source, so the label instrument is shared.

Usage:
    python scripts/arm_reference.py \
        --ours downloads/rent/raptor_oof_cn0_fold0.npz --our-fold 0 \
        --a5 downloads/research/a5oof_full/speed_a5_raw.npz --a5-fold 0
"""
import argparse
import csv
import json
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


def prep(a):
    return np.column_stack([rank01(np.asarray(a)[:, j]) for j in range(12)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="our arm's OOF npz (pred/truth/ids)")
    ap.add_argument("--our-folds", default="dataset-cv-fold/fold_file.json")
    ap.add_argument("--our-fold", default="0")
    ap.add_argument("--a5", required=True)
    ap.add_argument("--a5-folds", default="downloads/research/oof/v52_oof.csv")
    ap.add_argument("--a5-fold", default="0")
    ap.add_argument("--labels", default="downloads/research/oof/report_labels_v2.csv")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    z = np.load(args.ours, allow_pickle=True)
    ours_ids = [str(u) for u in z["ids"]]
    ours_pred = prep(np.asarray(z["pred"], dtype=np.float64))
    print("our arm : %s  studies=%d" % (args.ours.split("/")[-1], len(ours_ids)))

    our_folds = json.load(open(args.our_folds, encoding="utf-8"))
    our_folds = our_folds.get("folds", our_folds)
    a5_folds = {}
    with open(args.a5_folds, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a5_folds[r["StudyInstanceUID"]] = r["fold"]

    d = np.load(args.a5, allow_pickle=True)
    a5_ids = [str(u) for u in d["study_uids"]]
    raw = np.asarray(d["raw_probabilities"], dtype=np.float64)
    k = int(args.a5_fold)
    a5_held = prep(raw[k])
    print("A5 slice: raw[%d]  studies=%d" % (k, len(a5_ids)))

    a5_pos = {u: i for i, u in enumerate(a5_ids)}
    common = []
    for i, u in enumerate(ours_ids):
        if u not in a5_pos:
            continue
        # both arms must be held out on this study
        if str(our_folds.get(u)) != str(args.our_fold):
            continue
        if str(a5_folds.get(u)) != str(k):
            continue
        common.append(i)
    print("common held-out studies: %d" % len(common))
    if len(common) < 100:
        raise SystemExit("!! too few common held-out studies to compare anything")

    ids_l, Y, M = read_labels(args.labels)
    pos = {v: i for i, v in enumerate(ids_l)}
    sel = [i for i in common if ours_ids[i] in pos]
    print("scored (with report labels): %d\n" % len(sel))
    ids = [ours_ids[i] for i in sel]
    Ys = np.array([Y[pos[u]] for u in ids])
    Ms = np.array([M[pos[u]] for u in ids])
    O = ours_pred[sel]
    A = a5_held[[a5_pos[u] for u in ids]]

    m_o, per_o = macro(Ys, Ms, O)
    m_a, per_a = macro(Ys, Ms, A)
    print("%-46s macro=%.4f" % ("OUR arm (%s)" % args.ours.split("/")[-1][:24], m_o))
    print("%-46s macro=%.4f" % ("frontier A5, held out (raw[%d])" % k, m_a))
    print("gap (ours - A5) = %+.4f" % (m_o - m_a))

    rng = np.random.default_rng(args.seed)
    n = len(sel)
    boot = [rng.integers(0, n, n) for _ in range(args.boot)]
    dl = np.array([macro(Ys[b], Ms[b], O[b])[0] - macro(Ys[b], Ms[b], A[b])[0] for b in boot])
    dl = dl[~np.isnan(dl)]
    lo, hi = np.percentile(dl, [2.5, 97.5])
    print("   95%% CI [%+.4f, %+.4f]  P(ours>A5)=%.2f" % (lo, hi, float((dl > 0).mean())))
    print()
    print("per label (ours vs A5):")
    for j, t in enumerate(TARGETS):
        print("   %-18s ours=%.4f  A5=%.4f  delta=%+.4f" % (t, per_o[j], per_a[j],
                                                            per_o[j] - per_a[j]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
