"""Does the 88-feature calibrator carry information beyond the arms? A necessary-condition test.

THE QUESTION, AND WHY IT IS THE RIGHT ONE TO ASK FIRST
-----------------------------------------------------
The frontier's v38 changed exactly one predictive thing: `rad_cal_w` 0.40 -> 0.00, i.e. it stopped
blending the 88-feature calibrator's output into the Rad stage.  That knob is the least safe in the
pipeline -- its grid was selected on 58 gold labels (rad_e9_audit.json: rows 58, selection_supported
false) -- and it is the one knob none of our five arms touches.  So a candidate exists, but it cannot
be tested the way the pipeline ships it, because `rad_cal_w` affects both the calibrator's input block
and the blend, and both are built from **the parent ensemble**, the one arm with no legitimate
out-of-fold predictions.

What CAN be asked, with only genuine out-of-fold material, is the necessary condition:

    Does the calibrator's output carry ordering information beyond the arms it is blended into?

If it does not, then any nonzero weight on it can only hurt, v38 is right, and the candidate is dead.
If it does, the candidate has a mechanism and the question becomes whether the weight is right.

HOW IT IS MADE TESTABLE
-----------------------
* `baseline_rank` <- **our own genuine fold-0 ensemble** (11 arms, 870 studies, FACTS A112) as a
  stand-in for the parent.  It is a real ensemble with real out-of-fold predictions; it is simply a
  weaker one (~0.88 against the frontier's ~0.94), so the answer transfers as a mechanism, not as a
  number.
* `reference_rank` / `second_rank` <- the two published Rad out-of-fold files (e9 and e11), aligned on
  the same studies.
* `protocol` <- the acquisition table we already built for all 4407 studies (A101), assembled in the
  exact column order the payload records.
* The calibrator itself is a **published artifact**, not fitted here, so comparing weight 0 against
  weight 0.40 on the same studies carries no tuning bias.

⚠️ Honest limits, stated up front: our stand-in parent is weaker than the real one, the two Rad ranks
are e9/e11 rather than the E10/E13 the pipeline names, and report labels are noisy.  A negative result
is still decisive for the mechanism; a positive result only earns the candidate a closer look.

Usage:
    python scripts/calibrator_probe.py
"""
import argparse
import base64
import collections
import csv
import glob
import json
import os
import re
import sys
import zlib

import numpy as np

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PLANE_RX = re.compile(r"\b(sag|cor|ax)", re.I)
FATSAT_RX = re.compile(r"(fs|fat|fatsat|t2fs|stir|spair)", re.I)
FLUID_RX = re.compile(r"(t2|pd|fluid|stir)", re.I)
RAD_ALPHA = 0.55          # the shipped candidate blend for this stage


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


def rank_columns(a):
    return np.column_stack([rank01(np.asarray(a)[:, j]) for j in range(np.asarray(a).shape[1])])


def auc(y, p):
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = avg_ranks(p)
    return float((r[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def macro(Y, M, P, labels):
    per = {t: auc(Y[M[:, j], j], P[M[:, j], j]) for j, t in enumerate(TARGETS)}
    bad = [v for v in per.values() if not np.isnan(v) and not (0 <= v <= 1)]
    if bad:
        raise AssertionError("AUC outside [0,1] %r" % bad[:4])
    ok = [per[t] for t in labels if not np.isnan(per[t])]
    return float(np.mean(ok)), per


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


def load_payload():
    nbs = glob.glob(r"downloads\research\versia5\*.ipynb")
    if not nbs:
        raise SystemExit("!! pull evgendvorkin/rsna-versia-5 first")
    nb = json.load(open(nbs[0], encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    m = re.search(r"_RAD_CAL_PAYLOAD\s*=\s*['\"]([A-Za-z0-9+/=\s]{200,})['\"]", code)
    if not m:
        raise SystemExit("!! payload not found")
    raw = base64.b64decode(re.sub(r"\s", "", m.group(1)))
    return json.loads(zlib.decompress(raw))


def load_probe(path):
    """study -> the protocol counts the payload's protocol_columns asks for."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["StudyInstanceUID"]].append(r)
    out = {}
    for uid, rs in by.items():
        c = collections.Counter()
        for r in rs:
            d = (r.get("SeriesDescription") or "") + " " + (r.get("ScanOptions") or "")
            p = PLANE_RX.search(d)
            plane = p.group(1).lower() if p else ""
            fat = bool(FATSAT_RX.search(d))
            flu = bool(FLUID_RX.search(d))
            c["n_series"] += 1
            if plane == "sag":
                c["n_Sag"] += 1
            elif plane == "cor":
                c["n_Cor"] += 1
            elif plane == "ax":
                c["n_Axi"] += 1
            if fat:
                c["Fat"] += 1
                if plane == "sag":
                    c["Fat_Sag"] += 1
                elif plane == "cor":
                    c["Fat_Cor"] += 1
                elif plane == "ax":
                    c["Fat_Axi"] += 1
            if flu:
                c["Flu"] += 1
                if plane == "sag":
                    c["Flu_Sag"] += 1
                elif plane == "cor":
                    c["Flu_Cor"] += 1
                elif plane == "ax":
                    c["Flu_Axi"] += 1
        out[uid] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", default="downloads/rent/raptor_oof_cn0_fold0.npz",
                    help="unused fallback; the ensemble is built by our_ensemble.py logic below")
    ap.add_argument("--folds", default="downloads/research/oof/v52_oof.csv")
    ap.add_argument("--labels", default="downloads/research/oof/report_labels_v2.csv")
    ap.add_argument("--probe", default="downloads/research/siteprobe/site_probe.csv")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--baseline-npz", default="",
                    help="use this (ids, pred) npz as the stand-in parent instead of the fold-0 "
                         "ensemble mean; scripts/oof_surrogate.py builds the strongest one we have")
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cal = load_payload()
    print("calibrator payload keys: %s" % sorted(cal.keys()))
    print("  gate (%d): %s" % (len(cal["gate"]), cal["gate"]))
    print("  protocol_columns (%d): %s" % (len(cal["protocol_columns"]), cal["protocol_columns"]))
    print("  coef shape: %s" % (np.asarray(cal["coef"]).shape,))

    # --- the stand-in parent ----------------------------------------------------
    # Default: our own fold-0 ensemble (genuine OOF, 870 studies, 0.8823 -- A112).
    # `--baseline-npz` supplies a STRONGER honest baseline, e.g. the Rad-free surrogate built by
    # scripts/oof_surrogate.py (4406 studies, 0.9151).  Strength is the whole point: A109/A112
    # showed the same gain shrinking as the base strengthens (+0.008 at 0.842, +0.002 at 0.909,
    # ~0 on a 0.882 ensemble), so the 0.88 stand-in can only bound this answer from below -- and
    # whether the calibrator survives at all is exactly what decides the `rad_cal_w` question.
    arms = {}
    for pat in (r"downloads\cv\**\raptor_oof_*.npz", r"downloads\rent\raptor_oof_*.npz"):
        for p in glob.glob(pat, recursive=True):
            m = re.search(r"raptor_oof_([A-Za-z0-9]+?)(?:_(ep\d+|epens|swa))?_fold0\.npz$",
                          os.path.basename(p))
            if not m or m.group(2):
                continue
            d = np.load(p, allow_pickle=True)
            if len(d["ids"]) >= 800:
                arms[m.group(1)] = ([str(u) for u in d["ids"]],
                                    np.asarray(d["pred"], dtype=np.float64))
    ids_l, Y, M = read_labels(args.labels)
    pos_l = {u: i for i, u in enumerate(ids_l)}
    if args.baseline_npz:
        d = np.load(args.baseline_npz, allow_pickle=True)
        b_ids = [str(u) for u in d["ids"]]
        b_pred = np.asarray(d["pred"], dtype=np.float64)
        pos_b = {u: i for i, u in enumerate(b_ids)}
        ids = [u for u in ids_l if u in pos_b]
        base = rank_columns(np.array([b_pred[pos_b[u]] for u in ids]))
        label = "SUPPLIED baseline %s" % os.path.basename(args.baseline_npz)
    else:
        if not arms:
            raise SystemExit("!! no fold-0 arms found")
        common = None
        for b_ids, _ in arms.values():
            common = set(b_ids) if common is None else (common & set(b_ids))
        ids = [u for u in ids_l if u in common]
        base = np.mean([rank_columns(np.array([P[{u: i for i, u in enumerate(a)}[u]]
                                                for u in ids])) for a, P in arms.values()], axis=0)
        label = "%d arms (our own fold-0 ensemble)" % len(arms)
    Ys = np.array([Y[pos_l[u]] for u in ids]); Ms = np.array([M[pos_l[u]] for u in ids])
    print("\nstand-in parent: %s, %d studies" % (label, len(ids)))

    # --- the two published Rad ranks -------------------------------------------
    rad = {}
    for p in (r"downloads\research\oof\v52_oof.csv", r"downloads\research\oof\v52_e11_oof.csv"):
        rr = {}
        for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
            rr[r["StudyInstanceUID"]] = np.array([float(r[t]) for t in TARGETS])
        rad[os.path.basename(p)] = rr
    ref = np.array([rad["v52_oof.csv"][u] for u in ids])
    sec = np.array([rad["v52_e11_oof.csv"][u] for u in ids])

    # --- protocol, in the payload's own column order ---------------------------
    prot = load_probe(args.probe)
    missing = [c for c in cal["protocol_columns"] if c not in next(iter(prot.values()))]
    print("protocol columns the probe cannot supply: %s" % (missing or "none"))
    P = np.array([[prot[u].get(c, 0) for c in cal["protocol_columns"]] for u in ids], dtype=np.float64)

    # --- assemble the block vector exactly as the pipeline does ---------------
    br, rr_, sr = rank_columns(base), rank_columns(ref), rank_columns(sec)
    mean = (br + rr_ + sr) / 3.0
    blocks = [br, rr_, sr, rr_ - br, sr - br, mean]
    for g in cal["groups"]:
        idx = [TARGETS.index(t) for t in g]
        blocks.append(mean[:, idx].mean(axis=1, keepdims=True))
    blocks.append(P)
    x = np.concatenate(blocks, axis=1)
    center = np.asarray(cal["mean"]); spread = np.asarray(cal["scale"])
    coef = np.asarray(cal["coef"]); bias = np.asarray(cal["intercept"])
    if x.shape[1] != coef.shape[1]:
        raise SystemExit("!! block width %d != coef width %d" % (x.shape[1], coef.shape[1]))
    if not bool((spread > 0).all()):
        raise SystemExit("!! non-positive scale in the payload")
    adjusted = rank_columns(((x - center) / spread) @ coef.T + bias)

    # --- the comparison: weight 0 vs weight 0.40, on the gated labels ----------
    values = (1 - RAD_ALPHA) * br + RAD_ALPHA * rr_        # the candidate blend before calibration
    gated = [t for t in cal["gate"] if t in TARGETS]
    m0, per0 = macro(Ys, Ms, rank_columns(values), gated)
    v40 = values.copy()
    for t in gated:
        j = TARGETS.index(t)
        v40[:, j] = 0.6 * values[:, j] + 0.4 * adjusted[:, j]
    m40, per40 = macro(Ys, Ms, rank_columns(v40), gated)
    adj_only, per_adj = macro(Ys, Ms, adjusted, gated)
    print("\non the %d GATED labels (%s):" % (len(gated), ", ".join(gated)))
    print("  candidate blend, rad_cal_w = 0.00 : %.4f" % m0)
    print("  candidate blend, rad_cal_w = 0.40 : %.4f" % m40)
    print("  calibrator output ALONE            : %.4f" % adj_only)
    print("  => weight 0.40 minus weight 0.00   : %+.4f" % (m40 - m0))

    rng = np.random.default_rng(args.seed)
    n = len(ids)
    boot = [rng.integers(0, n, n) for _ in range(args.boot)]
    d = []
    for b in boot:
        a = macro(Ys[b], Ms[b], rank_columns(values)[b], gated)[0]
        c = macro(Ys[b], Ms[b], rank_columns(v40)[b], gated)[0]
        if not (np.isnan(a) or np.isnan(c)):
            d.append(c - a)
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print("     95%% CI [%+.4f, %+.4f]  P(0.40 > 0.00) = %.2f" % (lo, hi, float((d > 0).mean())))
    print("\nper gated label:")
    for t in gated:
        print("   %-16s w=0 %.4f  w=0.40 %.4f  calibrator-alone %.4f"
              % (t, per0[t], per40[t], per_adj[t]))

    # --- the DOSE, which 0.40 does not settle ---------------------------------
    # 0.40 is the shipped value, but A115 traced it to a selection on 58 gold labels -- below
    # every safety line A111 measured.  So "does the calibrator help" and "is 0.40 the right
    # dose" are separate questions, and only the first one has been asked.  A monotone-rising
    # sweep means the shipped value is too small; a peak at 0.40 means the 58-label choice
    # happened to land in the right place.
    print("\ncalibration-weight sweep on the gated labels (shipped value is 0.40):")
    sweep = {}
    for w in np.arange(0.0, 1.0001, 0.1):
        vw = values.copy()
        for t in gated:
            j = TARGETS.index(t)
            vw[:, j] = (1 - w) * values[:, j] + w * adjusted[:, j]
        mw, _ = macro(Ys, Ms, rank_columns(vw), gated)
        sweep[round(float(w), 2)] = mw
        print("   rad_cal_w=%.1f   %.4f%s" % (w, mw, "   <- shipped" if abs(w - 0.4) < 1e-9 else ""))
    w_star = max(sweep, key=sweep.get)
    print("   best on these studies: rad_cal_w=%.1f (%.4f), %+.4f vs shipped 0.40"
          % (w_star, sweep[w_star], sweep[w_star] - sweep[0.4]))
    print("   NOTE the dose is chosen on the same studies it is scored on, so that gap is an")
    print("   upper bound: a weight picked from 11 points on n=%d buys roughly +0.0005-0.001" % len(ids))
    print("   of pure selection (HANDOFF 86.2 measured +0.0012 for 300 draws on this scale).")

    # --- the honest dose: choose w on four folds, apply it to the fifth ---------
    # Without this the sweep above cannot distinguish "0.7 is genuinely better" from "0.7 won a
    # lottery among 11 candidates".  The fold map rides along inside the baseline npz.
    fold = None
    if args.baseline_npz:
        d = np.load(args.baseline_npz, allow_pickle=True)
        if "fold" in d.files:
            fmap = {str(u): int(k) for u, k in zip(d["ids"], d["fold"])}
            fold = np.array([fmap.get(u, -1) for u in ids])
    if fold is None:
        print("\n(no fold map on the supplied baseline; honest dose test skipped)")
        return 0

    grid = np.arange(0.0, 1.0001, 0.1)

    def blend_at(w):
        vw = values.copy()
        for t in gated:
            j = TARGETS.index(t)
            vw[:, j] = (1 - w) * values[:, j] + w * adjusted[:, j]
        return rank_columns(vw)

    ok = fold >= 0
    P_fix = np.zeros_like(values)
    P_dose = np.zeros_like(values)
    chosen = []
    for k in sorted(set(fold[ok].tolist())):
        te = fold == k
        tr = ok & ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        sc = [macro(Ys[tr], Ms[tr], blend_at(w)[tr], gated)[0] for w in grid]
        w_k = float(grid[int(np.argmax(sc))])
        chosen.append((k, w_k))
        P_fix[te] = blend_at(0.4)[te]
        P_dose[te] = blend_at(w_k)[te]
    print("\nhonest dose test (w chosen on the other four folds):")
    print("   chosen per fold: %s" % ", ".join("f%d->%.1f" % (k, w) for k, w in chosen))
    m_fix = macro(Ys[ok], Ms[ok], P_fix[ok], gated)[0]
    m_dose = macro(Ys[ok], Ms[ok], P_dose[ok], gated)[0]
    m_zero = macro(Ys[ok], Ms[ok], blend_at(0.0)[ok], gated)[0]
    print("   held-out gated macro: w=0.0 %.4f | w=0.4 (shipped) %.4f | fold-chosen w %.4f"
          % (m_zero, m_fix, m_dose))
    rng = np.random.default_rng(args.seed + 1)
    n = int(ok.sum())
    idx_all = np.where(ok)[0]
    dd = []
    for _ in range(args.boot):
        b = idx_all[rng.integers(0, n, n)]
        a = macro(Ys[b], Ms[b], P_dose[b], gated)[0]
        c = macro(Ys[b], Ms[b], P_fix[b], gated)[0]
        if not (np.isnan(a) or np.isnan(c)):
            dd.append(a - c)
    dd = np.array(dd)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    print("   fold-chosen minus shipped: %+.4f  95%% CI [%+.4f, %+.4f]  P(>0)=%.2f"
          % (m_dose - m_fix, lo, hi, float((dd > 0).mean())))
    # and what a change here is worth on the FULL twelve-label macro: the calibrator only touches
    # gated labels, so the effect is diluted by 5/12.
    all_lab = macro(Ys[ok], Ms[ok], P_dose[ok], TARGETS)[0]
    all_fix = macro(Ys[ok], Ms[ok], P_fix[ok], TARGETS)[0]
    all_zero = macro(Ys[ok], Ms[ok], blend_at(0.0)[ok], TARGETS)[0]
    print("   same three on the FULL 12-label macro: w=0.0 %.4f | w=0.4 %.4f | fold-chosen %.4f"
          % (all_zero, all_fix, all_lab))

    # --- the rival explanation: is the calibrator just "more Rad"? --------------
    # `values = (1-rad_alpha)*base + rad_alpha*reference`, and the calibrator is then blended on
    # top of that.  On a Rad-FREE base, a calibrator that mostly re-expresses the Rad heads would
    # look valuable here and be worth ~nothing in the real pipeline, whose `baseline_rank` already
    # contains the Rad arm at rad_alpha = 0.55.  So: can raising rad_alpha alone recover the same
    # gated gain?  If yes, the calibrator's apparent value is an artefact of our base being
    # Rad-free, and the real knob to move is rad_alpha, not rad_cal_w.
    print("\nrival explanation test -- rad_alpha sweep, with and without the calibrator")
    print("   (gated labels, n=%d).  rad_alpha shipped = %.2f" % (int(ok.sum()), RAD_ALPHA))
    alphas = np.arange(0.0, 1.0001, 0.1)
    print("   %-8s %-12s %-12s %-12s" % ("alpha", "cal_w=0.0", "cal_w=0.40", "cal_w=0.80"))
    for a in alphas:
        vals_a = (1 - a) * br + a * rr_
        cells = []
        for w in (0.0, 0.4, 0.8):
            vv = vals_a.copy()
            for t in gated:
                j = TARGETS.index(t)
                vv[:, j] = (1 - w) * vals_a[:, j] + w * adjusted[:, j]
            cells.append(macro(Ys[ok], Ms[ok], rank_columns(vv)[ok], gated)[0])
        print("   %-8.2f %-12.4f %-12.4f %-12.4f%s"
              % (a, cells[0], cells[1], cells[2], "   <- shipped" if abs(a - RAD_ALPHA) < 1e-9 else ""))
    print("   => if the cal_w columns rise with alpha, the calibrator is partly re-expressing Rad;")
    print("      if the rows are flat in alpha, the calibrator carries something Rad does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
