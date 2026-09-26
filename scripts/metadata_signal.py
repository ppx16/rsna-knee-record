"""Does acquisition metadata carry ordering information the arms do not already have?

THE QUESTION, AND WHY IT IS WORTH A NUMBER RATHER THAN AN ARGUMENT
------------------------------------------------------------------
Macro ROC-AUC counts only the ordering inside each column, so a monotone transform of a score is
free and invisible (maverickss26: "only ordering counts, per column").  Feature engineering can
therefore only pay off where it feeds a blend, and in this pipeline the single feature-engineering
stage is the 88-feature calibrator.  Its weight is `rad_cal_w` -- the one knob the frontier changed
(0.40 -> 0.00) and that we measured as negative on our own stack (D11) and cannot re-measure (A105).

We happen to own the raw material for that kind of feature engineering for free: a per-series
acquisition table for all 4407 training studies (FACTS A101), built by a CPU probe.  So instead of
arguing about whether header features could help, measure it.

WHAT IT MEASURES
----------------
Two configurations, both trained strictly out-of-fold on the PUBLISHER'S split (the `fold` column of
v52_oof.csv), because that is the split on which A5's `raw[k]` is a genuine held-out prediction:

  1. **metadata only** -- can acquisition headers predict the labels at all?  This is the floor.
  2. **metadata + the A5 rank** -- does adding metadata to the strongest arm we can validate improve
     on the arm alone?  THIS is the question a calibrator would have to answer yes to.

A third number is reported for context: **A5 alone**, on the same held-out folds.

⚠️ Interpretation caveat, stated up front: acquisition metadata is largely a scanner/site proxy, and
scanner was measured to be a covariate shared between training and the hidden set (A101), not a
removable group.  So even a positive result here would not automatically transfer, and a negative
result closes the route outright.

Usage:
    python scripts/metadata_signal.py
"""
import argparse
import collections
import csv
import re
import sys

import numpy as np

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

PLANE_RX = re.compile(r"\b(sag|cor|ax)", re.I)
FATSAT_RX = re.compile(r"(fs|fat|fatsat|t2fs|stir|spair|\bSTIR\b)", re.I)
NUMCOLS = ["n_files", "SliceThickness", "RepetitionTime", "EchoTime", "Rows", "Columns"]


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


def auc(y, p):
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = avg_ranks(p)
    return float((r[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def macro(Y, M, P):
    per = [auc(Y[M[:, j], j], P[M[:, j], j]) for j in range(12)]
    bad = [v for v in per if not np.isnan(v) and not (0 <= v <= 1)]
    if bad:
        raise AssertionError("AUC outside [0,1] %r" % bad[:4])
    ok = [v for v in per if not np.isnan(v)]
    return (float(np.mean(ok)) if ok else float("nan")), per


def num(v, lo=None, hi=None):
    try:
        x = float(str(v).split("|")[0])
    except Exception:
        return np.nan
    if lo is not None and x < lo:
        return np.nan
    if hi is not None and x > hi:
        return np.nan
    return x


def build_features(probe_csv):
    rows = list(csv.DictReader(open(probe_csv, newline="", encoding="utf-8")))
    print("probe rows (series): %d" % len(rows))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["StudyInstanceUID"]].append(r)

    models = collections.Counter(r.get("ManufacturerModelName", "") for r in rows)
    top_models = [m for m, _ in models.most_common(15) if m]
    manufs = [m for m, _ in collections.Counter(
        r.get("Manufacturer", "") for r in rows).most_common(8) if m]

    ids, feats, names = [], [], None
    for uid, rs in by.items():
        f, nm = [], []
        # counts by plane x fat-sat, the same shape of feature the calibrator's protocol block uses
        for pl in ("sag", "cor", "ax"):
            for fs in (0, 1):
                c = sum(1 for r in rs
                        if (PLANE_RX.search(r.get("SeriesDescription", "") or "")
                            and PLANE_RX.search(r["SeriesDescription"]).group(1).lower() == pl)
                        and (1 if FATSAT_RX.search((r.get("SeriesDescription", "") or "") + " "
                                                   + (r.get("ScanOptions", "") or "")) else 0) == fs)
                f.append(float(c)); nm.append("nslots_%s_fs%d" % (pl, fs))
        f.append(float(len(rs))); nm.append("n_series")
        f.append(float(len(set(r.get("SeriesDescription", "") for r in rs)))); nm.append("n_desc")
        for col in NUMCOLS:
            v = np.array([num(r.get(col)) for r in rs], dtype=float)
            v = v[np.isfinite(v)]
            for stat, val in (("mean", np.mean(v) if len(v) else np.nan),
                              ("std", np.std(v) if len(v) else np.nan),
                              ("min", np.min(v) if len(v) else np.nan),
                              ("max", np.max(v) if len(v) else np.nan)):
                f.append(float(val)); nm.append("%s_%s" % (col, stat))
        for m in manufs:
            f.append(1.0 if any(r.get("Manufacturer", "") == m for r in rs) else 0.0)
            nm.append("manuf_" + re.sub(r"\W+", "_", m)[:18])
        for m in top_models:
            f.append(1.0 if any(r.get("ManufacturerModelName", "") == m for r in rs) else 0.0)
            nm.append("model_" + re.sub(r"\W+", "_", m)[:18])
        ids.append(uid); f = np.array(f, dtype=np.float64)
        feats.append(f)
        if names is None:
            names = nm
    X = np.array(feats, dtype=np.float64)
    print("study-level metadata features: %d studies x %d features" % X.shape)
    # median-impute then standardise
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    idx = np.where(~np.isfinite(X))
    X[idx] = np.take(med, idx[1])
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return np.array(ids, dtype=object), (X - mu) / sd


def ridge_fit(X, y, lam):
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y
    return np.linalg.solve(A, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="downloads/research/siteprobe/site_probe.csv")
    ap.add_argument("--folds", default="downloads/research/oof/v52_oof.csv")
    ap.add_argument("--labels", default="downloads/research/oof/report_labels_v2.csv")
    ap.add_argument("--a5", default="downloads/research/a5oof_full/speed_a5_raw.npz")
    ap.add_argument("--arm", default="a5", choices=["a5", "rad"],
                    help="which genuine OOF to test metadata against; testing a SECOND arm is what "
                         "separates a real, arm-independent effect from a fold-specific accident")
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ids_p, X = build_features(args.probe)
    pos_p = {u: i for i, u in enumerate(ids_p)}

    fold = {}
    with open(args.folds, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fold[r["StudyInstanceUID"]] = int(r["fold"])

    rows = list(csv.DictReader(open(args.labels, newline="", encoding="utf-8")))
    ids_l = np.array([r["StudyInstanceUID"] for r in rows], dtype=object)
    Y = np.zeros((len(rows), 12)); M = np.zeros((len(rows), 12), dtype=bool)
    for i, r in enumerate(rows):
        for j, t in enumerate(TARGETS):
            v = (r.get(t + "__verdict") or "").strip().upper()
            if v == "YES":
                Y[i, j], M[i, j] = 1.0, True
            elif v == "NO":
                Y[i, j], M[i, j] = 0.0, True

    d = np.load(args.a5, allow_pickle=True)
    a5_ids = [str(u) for u in d["study_uids"]]
    raw = np.asarray(d["raw_probabilities"], dtype=np.float64)
    pos_a = {u: i for i, u in enumerate(a5_ids)}
    if args.arm == "a5":
        # the genuinely out-of-fold A5 rank for each study: its own fold's model
        a5_oof = np.full((len(a5_ids), 12), np.nan)
        for u, i in pos_a.items():
            k = fold.get(u)
            if k is not None:
                a5_oof[i] = raw[k][i]
    else:
        # the Rad arm's published OOF is genuine in its own right (it reproduces the audit's
        # rad_gold_macro_auc exactly), so it is a second, independent arm to test metadata against.
        a5_oof = np.full((len(a5_ids), 12), np.nan)
        with open(args.folds, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                u = r["StudyInstanceUID"]
                if u in pos_a:
                    a5_oof[pos_a[u]] = np.array([float(r[t]) for t in TARGETS])
    print("arm under test: %s" % args.arm)

    sel = [i for i, u in enumerate(ids_l) if u in pos_p and u in pos_a]
    print("scored studies (metadata + A5 + labels): %d\n" % len(sel))
    ids = [ids_l[i] for i in sel]
    Ys = np.array([Y[i] for i in sel]); Ms = np.array([M[i] for i in sel])
    Xs = np.array([X[pos_p[u]] for u in ids])
    A = np.array([a5_oof[pos_a[u]] for u in ids])
    kf = np.array([fold.get(u, -1) for u in ids])

    def rank01(a):
        out = np.empty_like(a, dtype=np.float64)
        for j in range(12):
            out[:, j] = (avg_ranks(a[:, j]) - 1) / max(len(a) - 1, 1)
        return out

    A_rank = rank01(A)
    md = np.full_like(A, np.nan)
    for k in range(5):
        te = kf == k
        tr = ~te
        Xtr, Xte = Xs[tr], Xs[te]
        for j in range(12):
            m = Ms[tr, j]
            if m.sum() < 20 or (~m).sum() < 20:
                continue
            y = Ys[tr, j][m]
            w = ridge_fit(Xtr[m], y - y.mean(), args.lam)
            md[te, j] = Xte @ w + y.mean()
    md_rank = rank01(md)

    # THE HONEST TEST OF "DOES METADATA ADD ANYTHING": fit the blend weight ON THE TRAINING FOLDS
    # and apply it to the held-out fold, exactly as the pipeline fits its own weights on a selection
    # set.  The first version of this script instead threw one raw A5 column in beside 55 standardised
    # metadata columns under a single ridge penalty, which shrinks the one strong feature far harder
    # than the many weak ones -- so it could not fall back on A5, and its answer (-0.10) measured my
    # construction rather than the data.  This version can always choose alpha = 0.
    def blend(a_rank, m_rank, alpha):
        return (1.0 - alpha) * a_rank + alpha * m_rank

    chosen = {}
    combo = np.full_like(A, np.nan)
    grid = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    for j in range(12):
        for k in range(5):
            te = kf == k
            tr = ~te
            mtr = Ms[tr, j]
            ytr = Ys[tr, j]
            if mtr.sum() < 20 or (~mtr).sum() < 20:
                continue
            best, best_a = -1.0, 0.0
            for a in grid:
                s = blend(A_rank[tr, j], md_rank[tr, j], a)
                v = auc(ytr[mtr], s[mtr])
                if not np.isnan(v) and v > best:
                    best, best_a = v, a
            chosen.setdefault(j, []).append(best_a)
            combo[te, j] = blend(A_rank[te, j], md_rank[te, j], best_a)

    m_md, per_md = macro(Ys, Ms, md_rank)
    m_a5, per_a5 = macro(Ys, Ms, A_rank)
    m_cb, per_cb = macro(Ys, Ms, combo)
    print("%-44s macro=%.4f" % ("1. metadata ONLY (no images, no arm)", m_md))
    print("%-44s macro=%.4f" % ("2. A5 arm alone (genuine OOF)", m_a5))
    print("%-44s macro=%.4f" % ("3. best train-fold-weight blend of the two", m_cb))
    print("   stacked minus A5 = %+.4f" % (m_cb - m_a5))
    print("   alpha chosen on the TRAINING folds, per label per fold:")
    for j, t in enumerate(TARGETS):
        vals = chosen.get(j, [])
        if vals:
            print("      %-18s mean alpha=%.3f   %s" % (t, float(np.mean(vals)),
                                                        ["%.2f" % v for v in vals]))

    rng = np.random.default_rng(args.seed)
    n = len(sel)
    boot = [rng.integers(0, n, n) for _ in range(args.boot)]
    dd = np.array([macro(Ys[b], Ms[b], rank01(combo)[b])[0] - macro(Ys[b], Ms[b], A_rank[b])[0]
                   for b in boot])
    dd = dd[~np.isnan(dd)]
    lo, hi = np.percentile(dd, [2.5, 97.5])
    print("   95%% CI [%+.4f, %+.4f]  P(stack>A5)=%.2f" % (lo, hi, float((dd > 0).mean())))
    print()
    print("per label:")
    for j, t in enumerate(TARGETS):
        print("   %-18s meta-only=%.4f  A5=%.4f  stack=%.4f" % (t, per_md[j], per_a5[j], per_cb[j]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
