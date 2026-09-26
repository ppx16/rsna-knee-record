"""Build the ensemble we CAN build, from our own genuine out-of-fold artifacts, and use it to test
the fusion questions that the frontier's stack cannot answer locally.

WHY A FOLD-0 ENSEMBLE IS THE HONEST THING TO BUILD
--------------------------------------------------
Our artefacts cover folds unevenly (13 arms on fold 0, five on fold 1, three on fold 2, none on
folds 3-4), so a five-fold ensemble OOF does not exist.  Inventing one by mixing folds would compare
different ensembles per fold, so the plan is a **fold-0 ensemble**: one fold, one arm set, 870 studies.

That size is not an accident of what we happen to have: FACTS A111 measured that a per-label tuning
set needs roughly 870 studies before the tuning stops transferring worse than a single shared weight.
So this ensemble is exactly large enough for the fusion questions to be asked without the answer being
an artefact of the tuning set.

WHAT IT CAN AND CANNOT SETTLE
-----------------------------
⚠️ Our arms are weak: the best measures 0.873 against the frontier A5's 0.914 on a common held-out set
(A106).  **Absolute conclusions therefore do not transfer to the 0.943 stack.**  What does transfer is
method: whether rank-mean ensembling helps, whether per-label weighting beats one shared weight at
this sample size, whether pruning members helps, and -- the question that decides whether to spend ¥12
and several days -- **how much acquisition metadata still adds once an ensemble sits on top.**

Usage:
    python scripts/our_ensemble.py --fold 0
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


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


def macro(Y, M, P):
    per = [auc(Y[M[:, j], j], P[M[:, j], j]) for j in range(12)]
    bad = [v for v in per if not np.isnan(v) and not (0 <= v <= 1)]
    if bad:
        raise AssertionError("AUC outside [0,1] %r" % bad[:4])
    ok = [v for v in per if not np.isnan(v)]
    return (float(np.mean(ok)) if ok else float("nan")), per


def read_labels(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    ids = [r["StudyInstanceUID"] for r in rows]
    Y = np.zeros((len(rows), 12)); M = np.zeros((len(rows), 12), dtype=bool)
    for i, r in enumerate(rows):
        for j, t in enumerate(TARGETS):
            v = (r.get(t + "__verdict") or "").strip().upper()
            if v == "YES":
                Y[i, j], M[i, j] = 1.0, True
            elif v == "NO":
                Y[i, j], M[i, j] = 0.0, True
    return ids, Y, M


def collect(fold):
    """tag -> (ids, pred) for the plain (non-variant) artefact of each arm on this fold."""
    out = {}
    pats = [r"downloads\cv\**\raptor_oof_*.npz", r"downloads\rent\raptor_oof_*.npz"]
    for pat in pats:
        for p in glob.glob(pat, recursive=True):
            m = re.search(r"raptor_oof_([A-Za-z0-9]+?)(?:_(ep\d+|epens|swa))?_fold(\d)\.npz$",
                          os.path.basename(p))
            if not m or int(m.group(3)) != fold or m.group(2):
                continue
            d = np.load(p, allow_pickle=True)
            out[m.group(1)] = ([str(u) for u in d["ids"]],
                               np.asarray(d["pred"], dtype=np.float64))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--labels", default="downloads/research/oof/report_labels_v2.csv")
    ap.add_argument("--metadata", action="store_true",
                    help="also test how much acquisition metadata still adds once an ensemble sits on "
                         "top -- this is what decides whether the image+metadata plan is worth ¥12")
    ap.add_argument("--probe", default="downloads/research/siteprobe/site_probe.csv")
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    arms = collect(args.fold)
    print("fold %d arms found: %d -> %s\n" % (args.fold, len(arms), sorted(arms)))
    if not arms:
        raise SystemExit("no arms on this fold")

    ids_l, Y, M = read_labels(args.labels)
    pos = {u: i for i, u in enumerate(ids_l)}

    # keep only arms that share a common study set, so the ensemble is one ensemble
    sizes = {k: len(v[0]) for k, v in arms.items()}
    common = None
    for k, (ids, _) in arms.items():
        s = set(ids)
        common = s if common is None else (common & s)
    common = [u for u in ids_l if u in common]
    keep = {k: v for k, v in arms.items() if len(v[0]) >= 800}
    print("common studies across all arms: %d ; arms used (>=800 studies): %d\n"
          % (len(common), len(keep)))

    ids = [u for u in common if u in pos]
    Ys = np.array([Y[pos[u]] for u in ids])
    Ms = np.array([M[pos[u]] for u in ids])
    preds = {}
    for k, (aids, P) in keep.items():
        p = {u: i for i, u in enumerate(aids)}
        preds[k] = np.array([P[p[u]] for u in ids])

    print("%-12s %s" % ("arm", "macro AUC vs report labels"))
    scores = {}
    for k in sorted(preds):
        m, _ = macro(Ys, Ms, preds[k])
        scores[k] = m
        print("%-12s %.4f" % (k, m))
    best = max(scores, key=scores.get)
    print("\nbest single arm: %s  %.4f" % (best, scores[best]))

    R = {k: np.column_stack([rank01(preds[k][:, j]) for j in range(12)]) for k in preds}
    ens_all = np.mean([R[k] for k in sorted(R)], axis=0)
    m_all, _ = macro(Ys, Ms, ens_all)
    top = sorted(scores, key=scores.get, reverse=True)
    for n in (3, 5, 8, len(top)):
        sub = top[:n]
        e = np.mean([R[k] for k in sub], axis=0)
        m, _ = macro(Ys, Ms, e)
        print("rank-mean of top %-2d arms (%s): %.4f" % (n, ",".join(sub[:4]) + ("..." if n > 4 else ""), m))
    print("\nall %d arms rank-mean: %.4f" % (len(R), m_all))
    print("  -> ensembling gain over the best single arm: %+.4f" % (m_all - scores[best]))

    if args.metadata:
        # THE QUESTION THAT DECIDES THE ¥12 PLAN: metadata added +0.008 to a weak single arm and
        # +0.002 to a strong one (A109).  An ensemble is stronger than both, so does anything survive?
        # Half-split within the fold, weight fitted on one half only -- never on the half it is
        # scored on, which is the mistake that made the first metadata draft report -0.10.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from metadata_signal import build_features
        ids_p, Xall = build_features(args.probe)
        pos_p = {u: i for i, u in enumerate(ids_p)}
        if not all(u in pos_p for u in ids):
            raise SystemExit("!! probe table does not cover every ensemble study")
        X = np.array([Xall[pos_p[u]] for u in ids])
        ens = np.mean([R[k] for k in top[:3]], axis=0)
        rng = np.random.default_rng(args.seed)
        grid = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
        honest, insample = [], []
        for rep in range(args.reps):
            perm = rng.permutation(len(ids))
            A, B = perm[: len(ids) // 2], perm[len(ids) // 2:]
            # metadata-only score, fitted on A
            md = np.full((len(ids), 12), np.nan)
            actx = rank01(ens[A].mean(1, keepdims=True) * 0) if False else None
            ea = np.column_stack([rank01(ens[A, j]) for j in range(12)])
            eb = np.column_stack([rank01(ens[B, j]) for j in range(12)])
            for j in range(12):
                mA = Ms[A, j]
                if mA.sum() < 20 or (~mA).sum() < 20:
                    continue
                Xa = X[A][mA]
                med = np.median(Xa, axis=0); med = np.where(np.isfinite(med), med, 0.0)
                Xa = np.where(np.isfinite(Xa), Xa, med)
                mu, sd = Xa.mean(0), Xa.std(0); sd = np.where(sd > 1e-9, sd, 1.0)
                y = Ys[A, j][mA]
                w = np.linalg.solve(((Xa - mu) / sd).T @ ((Xa - mu) / sd) + args.lam * np.eye(X.shape[1]),
                                    ((Xa - mu) / sd).T @ (y - y.mean()))
                Xb = np.where(np.isfinite(X[B]), X[B], med)
                md[B, j] = ((Xb - mu) / sd) @ w + y.mean()
            mdB = np.column_stack([rank01(md[B, j]) for j in range(12)])
            h, i2 = [], []
            for j in range(12):
                mA, mB = Ms[A, j], Ms[B, j]
                if mA.sum() < 20 or mB.sum() < 20:
                    continue
                sc = [auc(Ys[A, j][mA], (1 - a) * ea[:, j][mA] + a * rank01(md[A, j])[mA])
                      for a in grid]
                sc = [x for x in sc if not np.isnan(x)]
                if not sc:
                    continue
                a_star = grid[int(np.nanargmax(sc))]
                hv = auc(Ys[B, j][mB], (1 - a_star) * eb[:, j][mB] + a_star * mdB[:, j][mB])
                sv = [auc(Ys[B, j][mB], (1 - a) * eb[:, j][mB] + a * mdB[:, j][mB]) for a in grid]
                sv = [x for x in sv if not np.isnan(x)]
                if np.isnan(hv) or not sv:
                    continue
                h.append(hv); i2.append(max(sv))
            if h:
                honest.append(float(np.mean(h))); insample.append(float(np.mean(i2)))
        base, _ = macro(Ys, Ms, np.mean([R[k] for k in top[:3]], axis=0))
        print("\n--- metadata on top of a REAL ensemble (half-split, weight fitted on the other half) ---")
        print("  ensemble (top-3) alone, full fold      : %.4f" % base)
        print("  ensemble + metadata, honest half-split : %.4f" % np.mean(honest))
        print("  ensemble + metadata, same-half (optimistic): %.4f" % np.mean(insample))
        print("  => honest gain over the ensemble alone : %+.4f"
              % (np.mean(honest) - base))
        print("  => optimism of tuning on the scored half: %+.4f"
              % (np.mean(insample) - np.mean(honest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
