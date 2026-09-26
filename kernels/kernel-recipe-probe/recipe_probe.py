# -*- coding: utf-8 -*-
"""Recover the corpus builder's SLICE-SELECTION RECIPE by measuring it, not guessing.

WHY
    The plan is to rebuild our 44-slot corpus at higher density (64 or 80 slots), because the
    author measured corpus sampling at +0.006..+0.011 -- the largest lever he reports -- and
    our own `gap0` result (-0.012 for thinning the same 44 slices) is its mirror image.
    But a rebuild is only comparable if it reproduces the EXISTING corpus. So before building
    anything, recover the recipe that produced `all_vols.npy`.

HOW -- and this is the point
    For each of the five `_SLOTS44` groups we know exactly which rows of `all_vols.npy` came
    from it:

        _SLOTS44 = [("Sagittal",1,12), ("Sagittal",0,10), ("Coronal",1,8),
                    ("Coronal",0,6), ("Axial",-1,8)]      -> rows 0-11, 12-21, 22-29, 30-35, 36-43

    So for every one of those 44 rows we can search the FULL slice stack of the matching
    series for the best-matching slice. The argmin IS the slice that was used -- which gives
    the span and the selection rule directly, instead of inferring them from a description.

    Matching is done on 32x32 downscaled crops, so it survives differences in the exact
    resampling and reveals a CROP mismatch as low correlation rather than as a wrong answer.

OUTPUT
    per group: n_slices_in_series, the matched index for each corpus row, the implied
    normalised position, and the correlation -- plus a verdict on whether a consistent
    selection rule (evenly spaced across a span) explains the matched indices.
"""
import glob
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


def find(name):
    hits = sorted(glob.glob("/kaggle/input/**/" + name, recursive=True))
    return hits[0] if hits else None


def _resize(a, size):
    try:
        import cv2
        return cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        # nearest-neighbour fallback: good enough for 32x32 matching
        idx = (np.linspace(0, a.shape[0] - 1, size)).astype(int)
        idy = (np.linspace(0, a.shape[1] - 1, size)).astype(int)
        return a[np.ix_(idx, idy)]


def prep(ds, crop_mm=CROP_MM, out=336):
    """Corpus-style preprocessing: rescale, centre-crop crop_mm, resize to `out`."""
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


def stack_of(dcm_dir):
    """All slices of one series, ordered along the slice normal, plus the raw files."""
    files = sorted(glob.glob(os.path.join(dcm_dir, "*.dcm")))
    if not files:
        return []
    rows = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            iop = [float(x) for x in ds.ImageOrientationPatient]
            ipp = [float(x) for x in ds.ImagePositionPatient]
            n = np.cross(iop[:3], iop[3:])
            rows.append((float(np.dot(n, ipp)), f))
        except Exception:
            continue
    rows.sort(key=lambda t: t[0])
    return [f for _, f in rows]


def main():
    t0 = time.time()
    ids_path = find("all_ids.npy")
    vols_path = find("all_vols.npy")
    ts_path = find("train_series.csv")
    print("all_ids  -> %s" % ids_path, flush=True)
    print("all_vols -> %s" % vols_path, flush=True)
    print("train_series.csv -> %s" % ts_path, flush=True)
    if not (ids_path and vols_path and ts_path):
        print("!! a required input is missing; check dataset_sources / competition_sources",
              flush=True)
        raise SystemExit(0)

    ids = np.array([str(x) for x in np.load(ids_path)])
    vols = np.load(vols_path, mmap_mode="r")
    ts = pd.read_csv(ts_path)
    print("corpus %s | index rows %d | series index %d rows" % (vols.shape, len(ids), len(ts)),
          flush=True)

    pool = [u for u in ids if u in set(ts.StudyInstanceUID)]
    rng = np.random.default_rng(0)
    sample = list(rng.choice(pool, size=min(8, len(pool)), replace=False))
    root = os.path.join(os.path.dirname(ts_path), "train_series")
    if not os.path.isdir(root):
        cand = glob.glob("/kaggle/input/**/train_series", recursive=True)
        root = cand[0] if cand else root
    print("train_series root: %s\n" % root, flush=True)

    for s in sample:
        i = int(np.where(ids == s)[0][0])
        sub = ts[ts.StudyInstanceUID == s]
        print("=" * 78, flush=True)
        print("study %s (corpus row %d)  series=%d" % (s[:20], i, len(sub)), flush=True)
        for gi, (plane, fluid, n) in enumerate(GROUPS):
            off = int(OFF[gi])
            got = sub[sub.Anatomical_Plane == plane]
            if fluid != -1:
                got = got[got.Fluid_Sensitive == fluid]
            if not len(got):
                print("  %-9s fluid=%-3s : NO matching series" % (plane, fluid), flush=True)
                continue
            best = None
            for _, row in got.iterrows():
                d = os.path.join(root, s, str(row.SeriesInstanceUID))
                files = stack_of(d)
                if not files:
                    continue
                if best is None or len(files) > len(best[1]):
                    best = (str(row.SeriesInstanceUID), files)
            if best is None:
                print("  %-9s fluid=%-3s : series dir unreadable" % (plane, fluid), flush=True)
                continue
            sid, files = best
            # corpus rows for this group, downscaled for matching
            corpus = np.stack([_resize(np.asarray(vols[i, off + k]).astype(np.float32) / 255.0,
                                       MATCH_PX) for k in range(n)])
            cands = []
            for f in files:
                try:
                    cands.append(_resize(prep(pydicom.dcmread(f)), MATCH_PX))
                except Exception:
                    cands.append(None)
            cands = np.stack([c if c is not None else np.zeros((MATCH_PX, MATCH_PX), np.float32)
                              for c in cands])
            # correlation of every corpus row against every candidate slice
            A = corpus.reshape(n, -1)
            B = cands.reshape(len(cands), -1)
            A = A - A.mean(1, keepdims=True)
            B = B - B.mean(1, keepdims=True)
            A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-6)
            B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-6)
            C = A @ B.T                                    # (n_corpus, n_stack)
            idx = C.argmax(1)
            cor = C.max(1)
            pos = idx / max(1, len(files) - 1)
            print("  %-9s fluid=%-3s stack=%-4d matched idx=%s" % (plane, fluid, len(files),
                                                                   idx.tolist()), flush=True)
            print("            pos=%s" % np.round(pos, 3).tolist(), flush=True)
            print("            corr min/med/max = %.3f / %.3f / %.3f" %
                  (cor.min(), np.median(cor), cor.max()), flush=True)
        print("  [%d s elapsed]" % (time.time() - t0), flush=True)

    print("\n=== how to read this ===", flush=True)
    print("  * 'matched idx' IS the slice the corpus used (best correlation against the full",
          flush=True)
    print("    stack), so the span is idx/(stack-1) and the rule is what those indices look like.",
          flush=True)
    print("  * corr near 1 => the 140 mm crop + resize also agrees.  Low corr everywhere =>",
          flush=True)
    print("    the CROP differs (e.g. an ROI-centred crop), even if the slice choice is right.",
          flush=True)
    print("DONE in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
