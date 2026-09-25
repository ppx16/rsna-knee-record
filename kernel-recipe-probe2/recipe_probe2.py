# -*- coding: utf-8 -*-
"""Recipe probe v2 -- v1 fixed the RULE (linspace), v2 fixes the PARAMETERS and the SUBSTITUTION.

What v1 established (kernel-recipe-probe, 8 studies x 5 groups = 40 groups):
  * `np.linspace(lo, hi, n)` reproduces the matched slice indices to within half a slice in
    39/40 groups (the one exception is also the only group whose correlation was ~0.80, i.e.
    a bad match, not a different rule).
  * the span is a PER-PLANE quantity: the n=12 and n=10 sagittal groups share a span
    (23/23, 22/22, 20/20 slices), and so do the n=8 / n=6 coronal groups.
  * the span fraction sits in 0.667..0.72 and the midpoint is ~0.483 of the stack.

What v1 could NOT fix -- and what this kernel is for:
  1. `lo`/`hi` are only resolvable to +-1 slice from indices alone.  If the builder used a
     PHYSICAL span (millimetres) then the index span varies with slice spacing and the
     fraction varies with stack length -- both of which we can now measure directly, because
     ImagePositionPatient gives every matched index a millimetre position.
  2. 2 of 40 groups reported NO matching series, and one of them matched all-zero at
     correlation 0.000 -- so those corpus rows came from SOMEWHERE ELSE.  v2 pools every
     series of the study and asks which one actually explains those rows.

Design notes
  * Every series of a study is decoded ONCE and reused for all five groups (v1 re-decoded
    per group); the pooled pass therefore costs about the same as v1.
  * Matching stays on 32x32 crops with the 140 mm centre crop, so a crop mismatch shows up
    as low correlation instead of as a confident wrong answer.
  * Results are also written to /kaggle/working/recipe2.json so the numbers can be analysed
    without re-parsing a log.

CPU kernel on purpose: it consumes no GPU and no TPU quota.
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import pydicom

GROUPS = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
          ("Coronal", 0, 6), ("Axial", -1, 8)]
OFF = np.cumsum([0] + [g[2] for g in GROUPS])          # 0,12,22,30,36,44
CROP_MM = 140.0
MATCH_PX = 32
N_STUDIES = 8
OUT_JSON = "/kaggle/working/recipe2.json"


def find(name):
    hits = sorted(glob.glob("/kaggle/input/**/" + name, recursive=True))
    return hits[0] if hits else None


def _resize(a, size):
    try:
        import cv2
        return cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        idx = np.linspace(0, a.shape[0] - 1, size).astype(int)
        idy = np.linspace(0, a.shape[1] - 1, size).astype(int)
        return a[np.ix_(idx, idy)]


def prep(ds, crop_mm=CROP_MM, out=336):
    a = ds.pixel_array.astype(np.float32)
    a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    ps = getattr(ds, "PixelSpacing", None)
    px = float(ps[0]) if ps is not None else None
    if px and px > 0:
        want = int(round(crop_mm / px))
        h, w = a.shape
        if 16 < want < min(h, w):
            cy, cx = h // 2, w // 2
            half = want // 2
            a = a[cy - half:cy - half + want, cx - half:cx - half + want]
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        a = (a - lo) / (hi - lo)
    return _resize(a, out)


def scan_series(dcm_dir):
    """Order one series along its slice normal and keep per-file geometry."""
    files = sorted(glob.glob(os.path.join(dcm_dir, "*.dcm")))
    rows = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            iop = [float(x) for x in ds.ImageOrientationPatient]
            ipp = [float(x) for x in ds.ImagePositionPatient]
            n = np.cross(iop[:3], iop[3:])
            rows.append({
                "f": f,
                "s": float(np.dot(n, ipp)),
                "thick": float(getattr(ds, "SliceThickness", 0) or 0),
                "gap": float(getattr(ds, "SpacingBetweenSlices", 0) or 0),
                "px": [float(x) for x in getattr(ds, "PixelSpacing", [0, 0])],
            })
        except Exception:
            continue
    rows.sort(key=lambda r: r["s"])
    if len(rows) > 2:
        d = np.diff([r["s"] for r in rows])
        rows_meta = {"spacing_med": float(np.median(d)), "spacing_min": float(d.min()),
                     "spacing_max": float(d.max())}
    else:
        rows_meta = {"spacing_med": 0.0, "spacing_min": 0.0, "spacing_max": 0.0}
    return rows, rows_meta


def norm32(mat):
    a = np.asarray(mat, np.float32).reshape(-1)
    a = a - a.mean()
    return a / max(float(np.linalg.norm(a)), 1e-6)


def main():
    t0 = time.time()
    ids_path, vols_path, ts_path = find("all_ids.npy"), find("all_vols.npy"), find("train_series.csv")
    print("all_ids  -> %s" % ids_path, flush=True)
    print("all_vols -> %s" % vols_path, flush=True)
    if not (ids_path and vols_path and ts_path):
        print("!! missing input", flush=True)
        raise SystemExit(0)

    ids = np.array([str(x) for x in np.load(ids_path)])
    vols = np.load(vols_path, mmap_mode="r")
    ts = pd.read_csv(ts_path)
    print("corpus %s | ids %d | train_series %d" % (vols.shape, len(ids), len(ts)), flush=True)
    root = os.path.join(os.path.dirname(ts_path), "train_series")
    if not os.path.isdir(root):
        c = glob.glob("/kaggle/input/**/train_series", recursive=True)
        root = c[0] if c else root
    print("train_series root: %s\n" % root, flush=True)

    pool = [u for u in ids if u in set(ts.StudyInstanceUID)]
    sample = list(np.random.default_rng(0).choice(pool, size=min(N_STUDIES, len(pool)), replace=False))

    results = []
    for s in sample:
        i = int(np.where(ids == s)[0][0])
        sub = ts[ts.StudyInstanceUID == s]
        print("=" * 90, flush=True)
        print("study %s (corpus row %d)  series=%d" % (s[:22], i, len(sub)), flush=True)

        # --- decode every series of this study once -------------------------------
        series = []
        for _, row in sub.iterrows():
            d = os.path.join(root, s, str(row.SeriesInstanceUID))
            rows, meta = scan_series(d)
            if not rows:
                continue
            vecs, kept = [], []
            for r in rows:
                try:
                    vecs.append(norm32(_resize(prep(pydicom.dcmread(r["f"])), MATCH_PX)))
                    kept.append(r)
                except Exception:
                    pass
            series.append({
                "iuid": str(row.SeriesInstanceUID),
                "plane": str(row.Anatomical_Plane),
                "fluid": int(row.Fluid_Sensitive),
                "rows": kept, "meta": meta,
                "B": np.stack(vecs) if vecs else np.zeros((0, MATCH_PX * MATCH_PX), np.float32),
            })
        print("  decoded %d series: %s" % (
            len(series), [(x["plane"][:3], x["fluid"], len(x["rows"])) for x in series]), flush=True)

        pooled_B = np.concatenate([x["B"] for x in series], 0) if series else np.zeros((0, MATCH_PX * MATCH_PX), np.float32)
        owner = [k for k, x in enumerate(series) for _ in range(len(x["rows"]))]
        pooled_pos = [r["s"] for x in series for r in x["rows"]]

        for gi, (plane, fluid, n) in enumerate(GROUPS):
            off = int(OFF[gi])
            corpus = np.stack([norm32(_resize(np.asarray(vols[i, off + k]).astype(np.float32) / 255.0,
                                             MATCH_PX)) for k in range(n)])
            cands = [x for x in series if x["plane"] == plane and (fluid == -1 or x["fluid"] == fluid)]
            rec = {"row": i, "plane": plane, "fluid": fluid, "n": n, "own_series": bool(cands)}

            if cands:
                best = max(cands, key=lambda x: len(x["rows"]))
                C = corpus @ best["B"].T
                idx = C.argmax(1)
                cor = C.max(1)
                pos_mm = np.array([best["rows"][k]["s"] for k in idx])
                sp = best["meta"]["spacing_med"]
                rec.update({
                    "iuid": best["iuid"][:12], "stack": len(best["rows"]),
                    "spacing": round(sp, 4), "thick": round(best["rows"][0]["thick"], 3),
                    "spacing_min": round(best["meta"]["spacing_min"], 4),
                    "spacing_max": round(best["meta"]["spacing_max"], 4),
                    "px": best["rows"][0]["px"],
                    "idx": [int(x) for x in idx], "corr": [round(float(c), 3) for c in cor],
                    "pos_mm": [round(float(p), 2) for p in pos_mm],
                    "span_mm": round(float(pos_mm[-1] - pos_mm[0]), 2),
                    "span_frac": round(float((idx[-1] - idx[0]) / max(1, len(best["rows"]) - 1)), 4),
                    "center_frac": round(float((idx[0] + idx[-1]) / 2 / max(1, len(best["rows"]) - 1)), 4),
                })
                print("  %-9s fluid=%-3d stack=%-4d sp=%.3fmm thick=%.2f  idx=%s" % (
                    plane, fluid, rec["stack"], sp, rec["thick"], rec["idx"]), flush=True)
                print("            mm=%s" % rec["pos_mm"], flush=True)
                print("            span_mm=%.1f step_mm=%.2f span_frac=%.4f center_frac=%.4f corr=%.3f/%.3f/%.3f" % (
                    rec["span_mm"], rec["span_mm"] / max(1, n - 1), rec["span_frac"], rec["center_frac"],
                    min(rec["corr"]), sorted(rec["corr"])[len(rec["corr"]) // 2], max(rec["corr"])), flush=True)
            else:
                print("  %-9s fluid=%-3d : NO series with this plane/fluid" % (plane, fluid), flush=True)

            # --- pooled pass: who ELSE could explain these rows? -------------------
            if len(pooled_B):
                Cp = corpus @ pooled_B.T
                ip = Cp.argmax(1)
                cp = Cp.max(1)
                from collections import Counter
                own = Counter(owner[k] for k in ip)
                desc = ", ".join("%s/fluid=%d:%d" % (series[k]["plane"], series[k]["fluid"], c)
                                 for k, c in own.most_common(3))
                rec["pooled_best"] = desc
                rec["pooled_corr"] = [round(float(c), 3) for c in cp]
                print("            pooled -> %s   corr med=%.3f" % (desc, float(np.median(cp))), flush=True)
            results.append(rec)
        print("  [%d s elapsed]" % (time.time() - t0), flush=True)

    # ---- cross-study summary ----------------------------------------------------
    print("\n" + "=" * 90, flush=True)
    print("SUMMARY -- is the span PHYSICAL (mm) or FRACTIONAL (of the stack)?", flush=True)
    print("%-6s %-9s %-4s %8s %9s %10s %11s %8s" % (
        "row", "plane", "n", "stack", "spacing", "span_mm", "span_frac", "step_mm"), flush=True)
    for r in results:
        if "idx" not in r:
            continue
        print("%-6d %-9s %-4d %8d %9.3f %10.1f %11.4f %8.2f" % (
            r["row"], r["plane"], r["n"], r["stack"], r["spacing"], r["span_mm"], r["span_frac"],
            r["span_mm"] / max(1, r["n"] - 1)), flush=True)

    try:
        import statistics as st
        for plane in ("Sagittal", "Coronal", "Axial"):
            for n in sorted({r["n"] for r in results if r["plane"] == plane and "idx" in r}):
                sm = [r["span_mm"] for r in results if r["plane"] == plane and r["n"] == n and "idx" in r]
                sf = [r["span_frac"] for r in results if r["plane"] == plane and r["n"] == n and "idx" in r]
                if sm:
                    print("%-9s n=%-3d span_mm mean %6.1f sd %5.1f | span_frac mean %.4f sd %.4f" % (
                        plane, n, st.mean(sm), st.pstdev(sm), st.mean(sf), st.pstdev(sf)), flush=True)
        subs = [r for r in results if r.get("idx") is None or not r.get("own_series")]
        print("\nsubstitution cases (no own series): %d" % len(subs), flush=True)
        for r in subs:
            print("  row %d %s fluid=%d -> pooled %s" % (r["row"], r["plane"], r["fluid"],
                                                         r.get("pooled_best")), flush=True)
    except Exception as e:
        print("summary error: %s" % e, flush=True)

    try:
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=1)
        print("\nwrote %s (%d records)" % (OUT_JSON, len(results)), flush=True)
    except Exception as e:
        print("json write failed: %s" % e, flush=True)
    print("DONE in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
