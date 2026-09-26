"""Re-litigate the frontier's Rad blending grid on a large, report-labelled cohort.

WHAT PROBLEM THIS SOLVES
------------------------
The frontier chose `rad_alpha`/`rad_second_alpha`/the two-label `_RAD_EXCLUDE` list from an
out-of-fold grid computed on **58 image labels** -- and their own audit records
`selection_supported: False` with an evidence boundary of "58 official image labels" (FACTS A83).
Fifty-eight studies cannot separate a real effect from noise, which is precisely why they then
shipped a change (`rad_cal_w` 0.40 -> 0.00) whose only predictive effect we measured as NEGATIVE on
our own stack (D11) yet could not resolve.

This script redoes that grid on **report labels over the whole fold**, where the instrument is
roughly an order of magnitude sharper, and reports it **per label** -- because a +0.0118 gain on one
column is only ~+0.001 on the macro, and the per-label view is the one a per-label decision needs.

HOW THE TWO ARMS COME TOGETHER
------------------------------
* The Rad arm's OOF is published: `antoinegg1/rsna-knee-e9-radimagenet-heads-v15/v52_oof.csv`
  (4407 studies x 12 targets + fold + is_gold).  Verified identity: its gold-58 macro is exactly the
  audit's `rad_gold_macro_auc`, i.e. it is the arm ALONE, not an already-blended output.
* The parent arm (`E2`, the 20-member DINOv2 rank ensemble) is produced by
  `ppxl16/rsna-knee-e2oof`, which saves each member's prediction over a training fold.
* ⚠️ **FOLD DISCIPLINE, AND ITS HONEST LIMIT.**  A study in fold k may only be predicted by members
  whose training held fold k out.  There are ~4 such members of the 20, so the OOF parent is a
  4-member average while the deployed parent is a 20-member average.  The OOF parent is therefore
  *weaker* than the real one, which biases the fitted alpha **upward** (you lean on Rad more when the
  parent looks weak).  Use this for the RELATIVE re-litigation of the frontier's grid -- which has
  exactly the same structure -- and for per-label signs, NOT as a drop-in estimate of the deployed
  optimum.

Usage:
    python scripts/blend_lab.py --e2 downloads/research/e2oof_fold0 --fold 0 \
        --fold-log state/logs_e2oof_fold0.txt
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def avg_ranks(x):
    """1-based average ranks (ties share the mean rank)."""
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
    """Average-tie ranks scaled to [0,1] -- the notebook's `rsna_rank01` convention."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 1:
        return np.full_like(x, 0.5)
    return (avg_ranks(x) - 1.0) / (len(x) - 1.0)


def auc(y, p):
    """Mann-Whitney AUC.  Uses 1-based average ranks, so the tie correction is
    `npos*(npos+1)/2` -- writing `npos*(npos-1)/2` here (as the first draft did) yields AUCs below
    0 and a negative macro, which is what the self-test caught."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = avg_ranks(p)
    return float((r[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def read_pred(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = np.array([r["StudyInstanceUID"] for r in rows], dtype=object)
    P = np.array([[float(r[t]) for t in TARGETS] for r in rows], dtype=np.float64)
    fold = None
    if rows and "fold" in rows[0]:
        fold = np.array([int(float(r["fold"])) for r in rows], dtype=int)
    return ids, P, fold


def read_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = np.array([r["StudyInstanceUID"] for r in rows], dtype=object)
    Y = np.zeros((len(rows), len(TARGETS)))
    M = np.zeros((len(rows), len(TARGETS)), dtype=bool)
    for i, r in enumerate(rows):
        for j, t in enumerate(TARGETS):
            v = (r.get(t + "__verdict") or "").strip().upper()
            if v == "YES":
                Y[i, j], M[i, j] = 1.0, True
            elif v == "NO":
                Y[i, j], M[i, j] = 0.0, True
    return ids, Y, M


def member_folds(log_path):
    """member id -> held-out fold, read off the run log (`banked <id> fold N`)."""
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    out = {}
    for m in re.finditer(r"banked\s+(\w+)\s+fold\s+(\d+)", txt):
        out[m.group(1)] = int(m.group(2))
    return out


def macro(Y, M, P, idx=None):
    if idx is not None:
        Y, M, P = Y[idx], M[idx], P[idx]
    per = []
    for j in range(Y.shape[1]):
        m = M[:, j]
        per.append(auc(Y[m, j], P[m, j]))
    bad = [v for v in per if not np.isnan(v) and not (0.0 <= v <= 1.0)]
    if bad:
        # Positive control: an AUC outside [0,1] means the metric itself is wrong, and every number
        # downstream of it is meaningless.  The first draft of this file shipped that bug (a
        # mis-typed tie correction) and the smoke-data self-test is what exposed it.
        raise AssertionError("AUC outside [0,1]: %r -- the metric is broken, not the model" % bad[:4])
    ok = [v for v in per if not np.isnan(v)]
    return (float(np.mean(ok)) if ok else float("nan")), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rad", default="downloads/research/oof/v52_oof.csv")
    ap.add_argument("--e2", required=True, help="dir with diagnostics/dino_*.npz")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--fold-log", default=None,
                    help="kernel log to read the member->fold map from (banked <id> fold N)")
    ap.add_argument("--labels", default="downloads/research/oof/report_labels_v2.csv")
    ap.add_argument("--ref-alpha", type=float, default=0.55,
                    help="the frontier's shipped alpha, used as the comparison point")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # ---- the parent arm, fold-disciplined ------------------------------------
    diag = os.path.join(args.e2, "diagnostics")
    if not os.path.isdir(diag):
        diag = args.e2
    files = sorted(glob.glob(os.path.join(diag, "dino_*.npz")))
    if not files:
        raise SystemExit("!! no dino_*.npz under %s" % args.e2)
    if not args.fold_log:
        raise SystemExit("!! --fold-log is required: member->fold cannot be inferred from the npz "
                         "names, and guessing it is how an OOF silently leaks")
    mf = member_folds(args.fold_log)
    if len(mf) != 20:
        raise SystemExit("!! parsed %d member folds from the log (need 20)" % len(mf))
    use = sorted([k for k, v in mf.items() if v == int(args.fold)])
    print("members held out on fold %s: %d of %d -> %s" % (args.fold, len(use), len(mf), use))
    if len(use) < 3:
        raise SystemExit("!! too few members for a stable parent rank mean")

    ids_e, mats = None, []
    for k in use:
        p = os.path.join(diag, "dino_%s.npz" % k)
        if not os.path.isfile(p):
            raise SystemExit("!! missing %s" % p)
        d = np.load(p, allow_pickle=True)
        u = [str(x) for x in d["study_uids"]]
        v = np.asarray(d["values"], dtype=np.float64)
        if ids_e is None:
            ids_e = u
        elif u != ids_e:
            raise SystemExit("!! member %s covers a different cohort" % k)
        mats.append(v)
    P_e2 = np.mean([np.column_stack([rank01(m[:, j]) for j in range(m.shape[1])])
                    for m in mats], axis=0)
    print("parent (E2) OOF: %d studies, rank-mean of %d members" % (len(ids_e), len(mats)))

    # ---- the Rad arm ---------------------------------------------------------
    ids_r, P_rad, _ = read_pred(args.rad)
    pos_r = {v: i for i, v in enumerate(ids_r)}
    keep = [i for i, v in enumerate(ids_e) if v in pos_r]
    if len(keep) < 0.8 * len(ids_e):
        raise SystemExit("!! only %d of %d parent studies are in the Rad OOF"
                         % (len(keep), len(ids_e)))
    ids = [ids_e[i] for i in keep]
    E2 = P_e2[keep]
    RAD = np.array([P_rad[pos_r[v]] for v in ids])
    RAD = np.column_stack([rank01(RAD[:, j]) for j in range(12)])
    print("aligned with the Rad OOF on %d studies\n" % len(ids))

    # ---- labels --------------------------------------------------------------
    ids_l, Y, M = read_labels(args.labels)
    pos_l = {v: i for i, v in enumerate(ids_l)}
    sel = [i for i, v in enumerate(ids) if v in pos_l]
    ids = [ids[i] for i in sel]
    E2, RAD = E2[sel], RAD[sel]
    Y = np.array([Y[pos_l[v]] for v in ids])
    M = np.array([M[pos_l[v]] for v in ids])
    print("scored on %d studies against %s\n" % (len(ids), os.path.basename(args.labels)))

    # ---- the alpha grid ------------------------------------------------------
    grid = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.45, 0.55, 0.65]
    base_macro, base_per = macro(Y, M, E2)          # alpha = 0 == no Rad at all
    print("alpha grid (blend = (1-a)*parent + a*Rad);  reference point a=%s is the shipped value"
          % args.ref_alpha)
    print("  %-7s %-9s %s" % ("alpha", "macro", "delta vs alpha=0 (95% CI, P>0)"))
    rng = np.random.default_rng(args.seed)
    n = len(ids)
    boot_idx = [rng.integers(0, n, n) for _ in range(args.boot)]
    results = {}
    for a in grid:
        Pa = (1 - a) * E2 + a * RAD
        m, per = macro(Y, M, Pa)
        results[a] = (m, per)
        d = np.array([macro(Y, M, Pa, s)[0] - macro(Y, M, E2, s)[0] for s in boot_idx])
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        mark = "  <-- shipped" if abs(a - args.ref_alpha) < 1e-9 else ""
        print("  %-7.3f %-9.4f %+.4f [%+.4f, %+.4f]  P=%.2f%s"
              % (a, m, m - base_macro, lo, hi, float((d > 0).mean()), mark))

    # ---- per-label view: the `_RAD_EXCLUDE` decision -------------------------
    print("\nper label, at the shipped alpha = %s:" % args.ref_alpha)
    a = args.ref_alpha
    Pa = (1 - a) * E2 + a * RAD
    m_a, per_a = results.get(a) or macro(Y, M, Pa)
    print("  %-18s %-9s %-9s %-9s %s" % ("label", "no-Rad", "with-Rad", "delta", "1000-boot CI"))
    for j, t in enumerate(TARGETS):
        yv, mv = Y[:, j], M[:, j]
        p0, p1 = E2[mv, j], Pa[mv, j]
        d0, d1 = auc(yv[mv], p0), auc(yv[mv], p1)
        dd = []
        for s in boot_idx[:1000]:
            # label-level paired bootstrap on that label's own studies
            idxs = np.where(mv)[0]
            if len(idxs) == 0:
                continue
            b = rng.choice(idxs, len(idxs), replace=True)
            x0, x1 = auc(yv[b], E2[b, j]), auc(yv[b], Pa[b, j])
            if not (np.isnan(x0) or np.isnan(x1)):
                dd.append(x1 - x0)
        dd = np.array(dd)
        if len(dd):
            lo, hi = np.percentile(dd, [2.5, 97.5])
            ci = "[%+.4f, %+.4f]" % (lo, hi)
            flag = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "unresolved")
        else:
            ci, flag = "-", "-"
        print("  %-18s %-9s %-9s %+9.4f %s  %s"
              % (t, "%.4f" % d0 if not np.isnan(d0) else "-",
                 "%.4f" % d1 if not np.isnan(d1) else "-", d1 - d0, ci, flag))
    print("\nREMINDER: the OOF parent is a ~%d-member average, the deployed parent is 20;"
          % len(mats))
    print("that makes the parent look weaker than it is and biases the fitted alpha UP.")
    print("Read the per-label SIGNS and the frontier-relative comparison, not the absolute optimum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
