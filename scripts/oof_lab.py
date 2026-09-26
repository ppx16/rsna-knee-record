"""Local CV instrument over the PUBLISHED out-of-fold predictions (n=4407 training studies).

WHY THIS EXISTS
---------------
Every fusion-layer knob in this project (the Rad calibration weight, the `_RAD_EXCLUDE` list, the
blend alpha, the per-label outer routing) was closed as "cannot be judged" -- FACTS D11-D13 -- because
the only instruments available were

    gold-58        +-0.021      (58 studies, image labels)
    public LB      +-0.008      (B11/B12)

and the effects in question are ~0.001-0.012.  So we were deciding sub-noise knobs with a noisy
instrument, and D11 in particular ("turn the calibrator off") was rejected on a 0.001 LB difference.

The frontier meanwhile publishes real OOF predictions over the whole training set:
`antoinegg1/rsna-knee-e9-radimagenet-heads-v15/v52_oof.csv` and
`antoinegg1/rsna-knee-e11-diverse-heads-v20/v52_e11_oof.csv` are each StudyInstanceUID + the 12
targets + `fold` + `is_gold`, 4407 rows.  Paired with a report-label source that is 4407 studies with
per-label confidence, this gives an instrument two orders of magnitude larger than gold-58 -- on CPU,
for free, with no GPU quota and no rented box.

WHAT IT IS NOT
--------------
These are OTHER PEOPLE'S models' predictions.  It can only judge knobs that operate on those
predictions (blends, calibration, per-label mapping).  It cannot judge a change to the image
pipeline or a new backbone -- for that we would need to produce new OOFs, which needs the images.

⚠️ Report labels are NOISY (read out of free-text reports by a model, with per-label confidence), so
they are not ground truth.  They are usable for *relative* comparisons -- the same caveat the
published fracture-fork notebook leans on, where four independent report-label sources agreed and
only the 58-study gold set dissented.  Always report which label source a number came from.

Usage:
    python scripts/oof_lab.py --pred downloads/research/oof/v52_oof.csv
    python scripts/oof_lab.py --pred A.csv --compare B.csv         # paired bootstrap delta
    python scripts/oof_lab.py --pred A.csv --by-fold
"""
import argparse
import csv
import os
import sys

import numpy as np

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def _auc(y, p):
    """Rank-based AUC (Mann-Whitney), NaN when a class is absent."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    pos, neg = y == 1, y == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def read_pred(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = np.array([r["StudyInstanceUID"] for r in rows], dtype=object)
    P = np.array([[float(r[t]) for t in TARGETS] for r in rows], dtype=np.float64)
    extra = {}
    for k in ("fold", "is_gold"):
        if rows and k in rows[0]:
            extra[k] = np.array([int(float(r[k])) for r in rows], dtype=int)
    return ids, P, extra


def read_labels(path):
    """Report-label file -> (ids, truth, mask).  mask is False where the verdict is UNK."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = np.array([r["StudyInstanceUID"] for r in rows], dtype=object)
    Y = np.zeros((len(rows), len(TARGETS)), dtype=np.float64)
    M = np.zeros((len(rows), len(TARGETS)), dtype=bool)
    for i, r in enumerate(rows):
        for j, t in enumerate(TARGETS):
            v = (r.get(t + "__verdict") or "").strip().upper()
            if v == "YES":
                Y[i, j] = 1.0
                M[i, j] = True
            elif v == "NO":
                Y[i, j] = 0.0
                M[i, j] = True
    return ids, Y, M


def macro(Y, M, P, idx=None):
    """(macro, per-label, support) with per-label UNK masking."""
    if idx is not None:
        Y, M, P = Y[idx], M[idx], P[idx]
    per, sup = [], []
    for j in range(Y.shape[1]):
        m = M[:, j]
        y, p = Y[m, j], P[m, j]
        per.append(_auc(y, p))
        sup.append(int(y.sum()) if len(y) else 0)
    ok = [v for v in per if not np.isnan(v)]
    return (float(np.mean(ok)) if ok else float("nan")), per, sup


def align(ids_a, ids_b):
    """Index arrays putting two prediction sets on their common ids, in a stable order."""
    ib = {v: i for i, v in enumerate(ids_b)}
    common = [v for v in ids_a if v in ib]
    ia = {v: i for i, v in enumerate(ids_a)}
    return (np.array([ia[v] for v in common]), np.array([ib[v] for v in common]),
            np.array(common, dtype=object))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--labels",
                    default="downloads/research/oof/report_labels_v2.csv")
    ap.add_argument("--compare", default=None, help="second prediction csv for a paired delta")
    ap.add_argument("--by-fold", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ids_p, P, extra = read_pred(args.pred)
    ids_l, Y, M = read_labels(args.labels)
    print("pred  : %s  (%d studies)" % (os.path.basename(args.pred), len(ids_p)))
    print("labels: %s  (%d studies)" % (os.path.basename(args.labels), len(ids_l)))
    print("label source is NOISY (model-read from reports) -- relative comparisons only")

    # put predictions in label-file order
    pos = {v: i for i, v in enumerate(ids_p)}
    keep = [i for i, v in enumerate(ids_l) if v in pos]
    Y, M = Y[keep], M[keep]
    P = P[[pos[ids_l[i]] for i in keep]]
    fold = extra.get("fold")
    fold = fold[[pos[ids_l[i]] for i in keep]] if fold is not None else None
    is_gold = extra.get("is_gold")
    is_gold = is_gold[[pos[ids_l[i]] for i in keep]] if is_gold is not None else None
    print("aligned on %d studies\n" % len(keep))

    m, per, sup = macro(Y, M, P)
    print("macro AUC = %.4f   (n=%d, %d labels scored)" % (
        m, len(keep), sum(1 for v in per if not np.isnan(v))))
    print("  %-18s %-8s %s" % ("label", "AUC", "positives"))
    for t, v, s in zip(TARGETS, per, sup):
        print("  %-18s %-8s %d" % (t, "%.4f" % v if not np.isnan(v) else "-", s))

    if args.by_fold and fold is not None:
        print("\nper fold:")
        for f in sorted(set(fold.tolist())):
            idx = np.where(fold == f)[0]
            if len(idx) < 20:
                continue
            mf, _, _ = macro(Y, M, P, idx)
            print("  fold %d  n=%-5d macro=%.4f" % (f, len(idx), mf))

    if args.compare:
        ids_c, Pc, _ = read_pred(args.compare)
        ia, ib, common = align(ids_l[keep], ids_c)
        # ia indexes into the aligned rows; ib into the comparison file
        posc = {v: i for i, v in enumerate(ids_c)}
        # rebuild the comparison matrix in the same aligned order
        Pc_al = np.array([Pc[posc[v]] for v in ids_l[keep] if v in posc], dtype=np.float64)
        Yc, Mc = Y[ia], M[ia]
        Pa = P[ia]
        ma, _, _ = macro(Yc, Mc, Pa)
        mb, _, _ = macro(Yc, Mc, Pc_al)
        rng = np.random.default_rng(args.seed)
        d = np.empty(args.boot)
        n = len(ia)
        for b in range(args.boot):
            s = rng.integers(0, n, n)
            x, _, _ = macro(Yc, Mc, Pa, s)
            y, _, _ = macro(Yc, Mc, Pc_al, s)
            d[b] = y - x
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print("\n--- paired delta (B - A), n=%d common studies, %d boot ---" % (n, len(d)))
        print("  A = %s   macro %.4f" % (os.path.basename(args.pred), ma))
        print("  B = %s   macro %.4f" % (os.path.basename(args.compare), mb))
        print("  delta = %+.4f   95%% CI [%+.4f, %+.4f]   P(delta>0)=%.2f"
              % (mb - ma, lo, hi, float((d > 0).mean())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
