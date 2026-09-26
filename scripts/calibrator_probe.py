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
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cal = load_payload()
    print("calibrator payload keys: %s" % sorted(cal.keys()))
    print("  gate (%d): %s" % (len(cal["gate"]), cal["gate"]))
    print("  protocol_columns (%d): %s" % (len(cal["protocol_columns"]), cal["protocol_columns"]))
    print("  coef shape: %s" % (np.asarray(cal["coef"]).shape,))

    # --- the stand-in parent: our own fold-0 ensemble (genuine OOF) -------------
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
    if not arms:
        raise SystemExit("!! no fold-0 arms found")
    common = None
    for ids, _ in arms.values():
        common = set(ids) if common is None else (common & set(ids))
    ids_l, Y, M = read_labels(args.labels)
    pos_l = {u: i for i, u in enumerate(ids_l)}
    ids = [u for u in ids_l if u in common]
    base = np.mean([rank_columns(np.array([P[{u: i for i, u in enumerate(a)}[u]]
                                            for u in ids])) for a, P in arms.values()], axis=0)
    Ys = np.array([Y[pos_l[u]] for u in ids]); Ms = np.array([M[pos_l[u]] for u in ids])
    print("\nstand-in parent: %d arms, %d studies (our own fold-0 ensemble)" % (len(arms), len(ids)))

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
