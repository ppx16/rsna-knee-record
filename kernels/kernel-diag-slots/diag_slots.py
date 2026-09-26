"""Is there MORE slice sampling available in the raw DICOMs than the corpus kept?

WHY (this decides whether path (ii) is worth 1-2 weeks)
    The fusion kernel defines its slice layouts explicitly:

        _SLOTS44 = [("Sagittal",1,12), ("Sagittal",0,10), ("Coronal",1,8), ("Coronal",0,6), ("Axial",-1,8)]
        _SLOTS64 = [("Sagittal",1,18), ("Sagittal",0,14), ("Coronal",1,12), ("Coronal",0,8), ("Axial",-1,12)]

    -- i.e. a study contributes N slices from each of five (plane, fluid-sensitive) series types,
    44 total in the coarse layout and 64 in the dense one.  Our whole corpus is the 44 layout.

    Two facts make this the most promising lever left:
      * HANDOFF 91.1: the author measured corpus slice sampling at +0.006..+0.011 vs +0.001 for
        architecture.
      * HANDOFF 115: a zero-cost sampling change (window spacing) moved our own CV by 0.012 --
        the axis is LIVE -- but in the WRONG direction (wider is worse), so the good direction
        should be FINER.
      * HANDOFF 96.1: the frontier's heaviest arm, maxspan-v5 (weight 0.55), uses _SLOTS64.
    So "train our arms on a denser sampling" is a concrete, frontier-backed hypothesis.

    BUT it is only possible if the raw series actually CONTAIN more slices than the corpus kept.
    If a typical series has ~44 slices and the corpus took all of them, the idea dies here and
    path (ii) should not be started.  This measures exactly that, plus the decode throughput so
    a full 4407-study rebuild can be costed.

OUTPUT
    * slices available per (plane, fluid) series type, across a sample of studies
    * how many studies can fill 44 / 64 / 88 slots
    * slices/second decode throughput -> hours for a full rebuild
"""
import glob
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pydicom

COMP = "/kaggle/input/rsna-knee-abnormality-detection"
CORPUS = "/kaggle/input/**/knee-raptor-corpus"
LAYOUTS = {
    "44 (our corpus)": [("Sagittal", 1, 12), ("Sagittal", 0, 10),
                        ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)],
    "64 (frontier maxspan)": [("Sagittal", 1, 18), ("Sagittal", 0, 14),
                              ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)],
    "88 (double the corpus)": [("Sagittal", 1, 24), ("Sagittal", 0, 20),
                               ("Coronal", 1, 16), ("Coronal", 0, 12), ("Axial", -1, 16)],
}


def find(name):
    hits = sorted(glob.glob("/kaggle/input/**/" + name, recursive=True))
    return hits[0] if hits else None


def main():
    t0 = time.time()
    print("=== corpus index files ===", flush=True)
    for n in ("all_ids.npy", "all_masks.npy", "window_index.npy"):
        p = find(n)
        if not p:
            print("  %-18s MISSING" % n, flush=True)
            continue
        a = np.load(p, mmap_mode="r")
        print("  %-18s shape %-22s dtype %s" % (n, a.shape, a.dtype), flush=True)
        # one decoded sample row so the layout can be understood
        try:
            row = np.asarray(a[0])
            print("      first row: %s%s" % (row[:12], " ..." if row.size > 12 else ""), flush=True)
        except Exception as e:
            print("      (row decode failed: %s)" % e, flush=True)

    ids_path = find("all_ids.npy")
    ids = np.array([str(x) for x in np.load(ids_path)]) if ids_path else np.array([])
    print("\nsampled corpus ids: %d" % len(ids), flush=True)

    # ⚠️ Do NOT hardcode the competition mount path.  The first version of this probe died with
    # FileNotFoundError on '/kaggle/input/rsna-knee-abnormality-detection/train_series.csv' --
    # the same class of mistake as HANDOFF 81 (a source does not necessarily mount where its slug
    # suggests).  So: print what IS under /kaggle/input, then glob for the file.
    print("\n=== what is actually mounted under /kaggle/input ? ===", flush=True)
    for d in sorted(glob.glob("/kaggle/input/*")):
        print("  %s" % d, flush=True)
    ts_path = find("train_series.csv")
    print("  train_series.csv -> %s" % ts_path, flush=True)
    if not ts_path:
        print("  !! the competition data is not visible to this kernel.  Check competition_sources",
              flush=True)
        raise SystemExit(0)
    ts = pd.read_csv(ts_path)
    print("train_series.csv: %d rows, %d studies" % (len(ts), ts.StudyInstanceUID.nunique()), flush=True)
    comp_root = os.path.dirname(ts_path)
    print("competition root: %s" % comp_root, flush=True)

    rng = np.random.default_rng(0)
    pool = [u for u in ids if u in set(ts.StudyInstanceUID)] if ids.size else \
        list(ts.StudyInstanceUID.unique())
    print("studies present in both corpus and series index: %d" % len(pool), flush=True)
    sample = list(rng.choice(pool, size=min(60, len(pool)), replace=False))

    # ⚠️ 2026-09-19 BUG FIX (v3). v2 grouped series by
    # `getattr(hdr, "AnatomicalPlane", "")` -- which is NOT a pydicom keyword, so it
    # returned "" for EVERY series. Every layout then matched no group at all, and the
    # `or [0]` fallback turned that into a literal "0 / 25 studies (NOT available)".
    # **The v2 verdict "44/64/88 NOT available" was an artifact and is void.**
    # Lesson (same class as HANDOFF 16.1): a zero-match must never be able to read as a
    # real result, so (a) the grouping now comes from train_series.csv, and (b) "no
    # series of this type exists in the study" is counted and printed SEPARATELY from
    # "series of this type has too few slices".
    smap = {}
    for _r in ts.itertuples(index=False):
        smap[getattr(_r, "SeriesInstanceUID", None)] = (
            str(getattr(_r, "Anatomical_Plane", "") or "").strip(),
            getattr(_r, "Fluid_Sensitive", None))
    for _c in ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"):
        if _c in ts.columns:
            print("  [csv] %-18s %s" % (_c, dict(ts[_c].value_counts(dropna=False).head(8))),
                  flush=True)

    root = os.path.join(comp_root, "train_series")
    per_type = defaultdict(list)          # (plane, fluid) -> slice counts
    per_study = {}                        # study -> {(plane,fluid): n_slices}
    n_series_seen = 0
    spacings = []
    for s in sample:
        sd = os.path.join(root, s)
        if not os.path.isdir(sd):
            continue
        got = {}
        for series in sorted(os.listdir(sd)):
            files = glob.glob(os.path.join(sd, series, "*.dcm"))
            if not files:
                continue
            n_series_seen += 1
            hdr = pydicom.dcmread(files[0], stop_before_pixels=True)
            # CSV first (it is the only source that actually carries these fields),
            # DICOM header only as a fallback.
            cplane, cfluid = smap.get(series, ("", None))
            plane = cplane or str(getattr(hdr, "AnatomicalPlane", "") or "")
            try:
                fluid = int(cfluid) if cfluid is not None and str(cfluid).strip() not in ("", "nan") else -1
            except Exception:
                fluid = -1
            if fluid == -1:
                _f = getattr(hdr, "FluidSensitive", None)
                fluid = int(_f) if _f is not None else -1
            got[(plane, int(fluid) if fluid is not None else -1)] = len(files)
            per_type[(plane, int(fluid) if fluid is not None else -1)].append(len(files))
            sp = getattr(hdr, "SpacingBetweenSlices", None) or getattr(hdr, "SliceThickness", None)
            if sp:
                spacings.append(float(sp))
        per_study[s] = got

    print("\n=== slices available per series type (over %d studies, %d series) ==="
          % (len(per_study), n_series_seen), flush=True)
    print("  %-22s %5s %8s %8s %8s" % ("(plane, fluid)", "n", "median", "min", "max"), flush=True)
    for key in sorted(per_type, key=lambda k: (-len(per_type[k]), str(k)))[:14]:
        v = np.array(per_type[key])
        print("  %-22s %5d %8.0f %8d %8d"
              % ("%s/%s" % key, len(v), np.median(v), v.min(), v.max()), flush=True)
    if spacings:
        sp = np.array(spacings)
        print("\n  slice spacing: median %.2f mm  p10 %.2f  p90 %.2f  (n=%d)"
              % (np.median(sp), np.percentile(sp, 10), np.percentile(sp, 90), len(sp)), flush=True)

    def _have(got, plane, fluid):
        """Slices available for one (plane, fluid) group in one study.
        Returns (n_slices, matched?) -- the `matched` flag is why this returns a tuple:
        a study with NO series of this type must be distinguishable from one whose
        series of this type has 0 slices."""
        m = [v for (pl, fl), v in got.items() if pl == plane and (fluid == -1 or fl == fluid)]
        return (max(m), True) if m else (0, False)

    print("\n=== per-group availability (the actual constraint) ===", flush=True)
    # ⚠️ 2026-09-19: this loop originally said `(("SLOTS44", _SLOTS44), ("SLOTS64", _SLOTS64))`
    # and died with NameError: those two names exist ONLY inside this module's DOCSTRING
    # (lines 6-7, written as a code example).  `grep '_SLOTS44'` finds them and looks like
    # proof they are defined -- it is not.  The real table is LAYOUTS below.  Iterating it
    # also picks up the 88 ceiling, which is exactly where the coronal cap should bite.
    for gname, spec in LAYOUTS.items():
        for plane, fluid, n in spec:
            have = np.array([_have(got, plane, fluid)[0] for got in per_study.values()])
            if have.size:
                print("  %-8s %-9s fluid=%-3s need>=%2d : %2d/%2d studies "
                      "(median have %.0f)"
                      % (gname, plane, fluid, n, int((have >= n).sum()), have.size,
                         float(np.median(have))), flush=True)

    print("\n=== how many studies can FILL each layout (all groups at once)? ===", flush=True)
    for name, spec in LAYOUTS.items():
        total = sum(n for _p, _f, n in spec)
        ok = 0
        nomatch = 0
        deficit = []
        for s, got in per_study.items():
            need = 0
            for plane, fluid, n in spec:
                h, matched = _have(got, plane, fluid)
                if not matched:
                    nomatch += 1
                need += min(h, n)
            if need >= total:
                ok += 1
            deficit.append(total - need)
        deficit = np.array(deficit)
        print("  %-24s filled by %2d / %2d studies | "
              "group-lookups with NO matching series: %d | short by median %.0f slots"
              % (name, ok, len(per_study), nomatch, float(np.median(deficit))), flush=True)

    # decode throughput -> rebuild cost
    print("\n=== decode throughput ===", flush=True)
    files = []
    for s in sample[:4]:
        sd = os.path.join(root, s)
        if os.path.isdir(sd):
            for series in sorted(os.listdir(sd))[:2]:
                files += glob.glob(os.path.join(sd, series, "*.dcm"))[:25]
    files = files[:120]
    if files:
        t1 = time.time()
        npx = 0
        for f in files:
            d = pydicom.dcmread(f)
            npx += d.pixel_array.size
        dt = time.time() - t1
        print("  %d slices in %.1f s -> %.1f slices/s (%.1f Mpix/s)"
              % (len(files), dt, len(files) / dt, npx / dt / 1e6), flush=True)
        # 4407 studies x (64 slices + 3x window stacking + resize) -- slices only, lower bound
        print("  lower bound for a 4407-study 64-slice rebuild: %.1f h (decode only, 1 core)"
              % (4407 * 64 / (len(files) / dt) / 3600), flush=True)
        print("  with 4 workers: %.1f h" % (4407 * 64 / (len(files) / dt) / 3600 / 4), flush=True)
    print("\nDONE in %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
