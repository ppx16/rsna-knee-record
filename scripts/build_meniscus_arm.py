"""Build `kernel-0943cfmsft2m-men`: our best arm plus the public0033 meniscus specialist.

WHAT THIS IS
    `mattiaangeli/rsna-knee-public0033-meniscus-bag-mirror` ships a trained DINOv2-Base
    meniscus specialist plus the author's own immutable runtime, whose entrypoint is
        run_cached_inference(cache, mask, studies, output_paths)
    with cache = uint8 (N, 6, 12, 336, 336), mask = float32 (N, 6).

    The author's preregistered recipe (`public0033_meniscus10_residual.yaml`, `runtime:` block
    all false -- never executed) applies it at a FIXED 90:10 to ONLY "Medial Meniscus" and
    "Lateral Meniscus", leaving the other ten targets untouched, and forbids retuning the weight
    after seeing a public score.

WHY IT IS CHEAP TO WIRE, WHICH WAS NOT OBVIOUS
    The runtime wants 6 slots x 12 slices @336, crop 130 mm, band (0.20, 0.80), and
    "joint_12_slice_p1_p99" normalisation. Our stack's `read_slot` already
        vol = stack(selected slices); p1,p99 = percentile(vol, [1,99])   # joint over the STACK
    i.e. called with n_slice=12 it IS joint_12_slice_p1_p99. And `build_cache` already emits
    exactly (N, 6, n_slice, img, img) uint8 plus (N, 6) float32 mask.
    So the ONLY deltas from our DINO geometry are slice band (0.12,0.88) -> (0.20,0.80) and
    slice count -> 12, both of which are globals the Rad stage already overrides in exactly
    this way. No renderer is re-implemented.

RISK, STATED PLAINLY
    The geometry is a RECONSTRUCTION from the YAML. If it is off, the specialist emits
    plausible-looking-but-wrong predictions, and we have NO local instrument that can see that
    (the arm would be judged only by a board that cannot resolve +0.0006). Mitigations: reuse
    the stack's own renderer rather than a fresh one; assert the cache contrete contract before
    and after; print sanity statistics (per-label mean/spread, rank correlation with our stack)
    so a degenerate or wildly-disagreeing member is visible in the log; and fail CLOSED.

FAIL-CLOSED
    Any exception leaves `submission.csv` exactly as the base pipeline wrote it, so this arm
    degrades to `0943cfmsft2m` rather than to a broken file.

Usage:
    python scripts/build_meniscus_arm.py            # write files
    python scripts/build_meniscus_arm.py --push     # write + push
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels" / "kernel-0943cfmsft2m"
NB = "rsna-knee-restructured-version-3.ipynb"
SLUG = "rsna-knee-0943cfmsft2m-men"
TITLE = "RSNA knee 0943cfmsft2m men"
BUNDLE = "mattiaangeli/rsna-knee-public0033-meniscus-bag-mirror"

CELL = '''# ---------------------------------------------------------------------------
# public0033 meniscus specialist: rebuild the cache at ITS geometry, run the
# author's immutable runtime, and blend 90:10 on the TWO meniscus targets only.
#
# FAIL-CLOSED: any exception leaves submission.csv exactly as the base pipeline
# wrote it, so this arm degrades to 0943cfmsft2m and never to a broken file.
# ---------------------------------------------------------------------------
import glob as _m_g, importlib.util as _m_ilu, json as _m_json, traceback as _m_tb
import numpy as _m_np, pandas as _m_pd

MEN_W = 0.10
MEN_TARGETS = ["Medial Meniscus", "Lateral Meniscus"]
_SUB = "/kaggle/working/submission.csv"

try:
    _roots = [p for p in _m_g.glob("/kaggle/input/**/public0033_runtime.py", recursive=True)]
    if len(_roots) != 1:
        raise RuntimeError("expected exactly one public0033_runtime.py, found %r" % _roots)
    _bundle = os.path.dirname(_roots[0])
    print("[men] bundle:", _bundle, flush=True)
    _spec = _m_ilu.spec_from_file_location("public0033_runtime", _roots[0])
    _rt = _m_ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_rt)
    print("[men] runtime imported; entrypoint:", _rt.ENTRYPOINT, flush=True)

    # --- 1. rebuild the cache at the specialist's geometry, using OUR renderer -----------------
    _tser = _m_pd.read_csv(ROOT / "test_series.csv",
                           dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    _plane = dict(zip(_tser.SeriesInstanceUID, _tser.Anatomical_Plane))
    _hdr = annotate(walk("test_series"))
    _side = lat_of(_hdr, "meniscus")
    _slots = pick_slots(_hdr, _plane)
    _keep = {k: globals().get(k) for k in
             ("N_SLOT", "CACHE_SLICES", "IMG", "CACHE_IMG", "CROP_MM", "SLICE_BAND", "RULES")}
    print("[men] geometry in force before override: N_SLOT=%s CACHE_SLICES=%s IMG=%s CROP_MM=%s SLICE_BAND=%s"
          % (_keep["N_SLOT"], _keep["CACHE_SLICES"], _keep["IMG"], _keep["CROP_MM"], _keep["SLICE_BAND"]),
          flush=True)
    globals().update(N_SLOT=6, CACHE_SLICES=12, IMG=336, CACHE_IMG=336,
                     CROP_MM=130.0, SLICE_BAND=(0.20, 0.80))
    try:
        _st, _cache, _mask = build_cache(_slots, _plane, _side, "meniscus")
    finally:
        globals().update(_keep)
    _st = [str(s) for s in _st]
    print("[men] cache %s %s | mask %s %s | studies %d"
          % (_cache.shape, _cache.dtype, _mask.shape, _mask.dtype, len(_st)), flush=True)

    # --- 2. enforce the runtime's contract HERE, so a mismatch is a clear message --------------
    if _cache.dtype != _m_np.uint8 or _cache.shape[1:] != (6, 12, 336, 336):
        raise RuntimeError("cache contract violated: %s %s" % (_cache.shape, _cache.dtype))
    if _mask.dtype != _m_np.float32 or _mask.shape != (_cache.shape[0], 6):
        raise RuntimeError("mask contract violated: %s %s" % (_mask.shape, _mask.dtype))
    if _mask.sum(axis=1).min() <= 0:
        raise RuntimeError("a study has an all-zero mask row; the runtime forbids neutralisation")

    # the runtime requires SORTED, unique studies -- sort and carry the permutation
    _order = _m_np.argsort(_m_np.asarray(_st))
    _st_s = [_st[i] for i in _order]
    _cache_s = _cache[_order]
    _mask_s = _mask[_order]
    if len(set(_st_s)) != len(_st_s):
        raise RuntimeError("duplicate study UIDs after sorting")

    # --- 3. run the author's runtime ------------------------------------------------------------
    _out = {"bag_raw_csv": "/kaggle/working/_meniscus_raw.csv",
            "receipt_json": "/kaggle/working/_meniscus_receipt.json",
            "work_dir": "/kaggle/working/_meniscus_work"}
    import time as _m_t
    _t0 = _m_t.time()
    _rec = _rt.run_cached_inference(_cache_s, _mask_s, _st_s, _out)
    print("[men] inference %.1fs for %d studies -> %s" % (_m_t.time() - _t0, len(_st_s), _out["bag_raw_csv"]),
          flush=True)
    print("[men] receipt status:", _rec.get("status"), flush=True)
    del _cache, _cache_s, _mask, _mask_s

    # --- 4. blend 90:10 on the two meniscus targets ONLY ----------------------------------------
    _mem = _m_pd.read_csv(_out["bag_raw_csv"], dtype={"StudyInstanceUID": str})
    _mem = _mem.set_index("StudyInstanceUID").loc[_st].reset_index()
    if _mem.StudyInstanceUID.tolist() != _st:
        raise RuntimeError("specialist/base study alignment failed")

    _base = _m_pd.read_csv(_SUB, dtype={"StudyInstanceUID": str})
    if _base.columns.tolist() != ["StudyInstanceUID", *LABELS]:
        raise RuntimeError("submission schema drift: %r" % _base.columns.tolist())
    _mem = _mem.set_index("StudyInstanceUID").loc[_base.StudyInstanceUID].reset_index()

    print("[men] specialist per-target mean / std / min / max, and spearman vs our stack:", flush=True)
    for _c in MEN_TARGETS:
        if _c not in _mem.columns:
            raise RuntimeError("specialist did not emit %r" % _c)
        _v = _mem[_c].to_numpy(_m_np.float64)
        if not _m_np.isfinite(_v).all():
            raise RuntimeError("specialist produced non-finite %r" % _c)
        _rho = float(_base[_c].corr(_mem[_c], method="spearman"))
        print("        %-18s mean=%.4f std=%.4f min=%.4f max=%.4f  rho=%.4f"
              % (_c, _v.mean(), _v.std(), _v.min(), _v.max(), _rho), flush=True)
        if _v.std() == 0.0:
            raise RuntimeError("specialist output for %r is constant -- refusing to blend" % _c)

    _blended = _base.copy()
    for _c in MEN_TARGETS:
        _b = _base[_c].rank(pct=True)
        _mm = _mem[_c].rank(pct=True)
        _blended[_c] = (1.0 - MEN_W) * _b + MEN_W * _mm
    if _blended.columns.tolist() != _base.columns.tolist():
        raise RuntimeError("blend changed the schema")
    if _blended.StudyInstanceUID.tolist() != _base.StudyInstanceUID.tolist():
        raise RuntimeError("blend changed the study order")
    _untouched = [c for c in LABELS if c not in MEN_TARGETS]
    if not _blended[_untouched].equals(_base[_untouched]):
        raise RuntimeError("the blend touched a target outside the two meniscus columns")

    _blended.to_csv(_SUB, index=False)
    _chk = _m_pd.read_csv(_SUB, dtype={"StudyInstanceUID": str})
    assert _chk.columns.tolist() == _base.columns.tolist() and len(_chk) == len(_base)
    assert _m_np.isfinite(_chk[list(LABELS)].to_numpy()).all()
    print("[men] BLEND APPLIED at w=%.2f on %s -> %s" % (MEN_W, MEN_TARGETS, _SUB), flush=True)
except Exception as _e:
    print("[men] SPECIALIST FAILED, base submission left untouched:", flush=True)
    _m_tb.print_exc()
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--dst-suffix", default="men")
    a = ap.parse_args()

    slug = SLUG if a.dst_suffix == "men" else SLUG.replace("-men", "-" + a.dst_suffix)
    dst = ROOT / "kernels" / ("kernel-0943cfmsft2m-men" if a.dst_suffix == "men"
                              else f"kernel-0943cfmsft2m-{a.dst_suffix}")

    nb = json.loads((SRC / NB).read_text(encoding="utf-8"))
    n_before = len(nb["cells"])
    for c in nb["cells"]:
        s = c.get("source", "")
        if isinstance(s, list):
            s = "".join(s)
        if "public0033_runtime" in s:
            print("!! meniscus cell already present -- refusing to append twice")
            return 2
    nb["cells"].append({"cell_type": "code", "execution_count": None, "id": "meniscus0033",
                        "metadata": {}, "outputs": [], "source": CELL})
    assert len(nb["cells"]) == n_before + 1

    # assert on the RESULT: the cell must be present, syntactically valid, and must not have
    # acquired a raw newline inside the geometry tuple
    src = "".join(nb["cells"][-1]["source"])
    compile(src, "<meniscus-cell>", "exec")
    for must in ("SLICE_BAND=(0.20, 0.80)", "CACHE_SLICES=12", "MEN_W = 0.10",
                 "run_cached_inference", "MEDIAL" if False else "Medial Meniscus"):
        assert must in src, "cell is missing %r" % must
    assert "globals().update(_keep)" in src, "geometry restore missing"

    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = "ppxl16/" + slug
    meta["title"] = TITLE if a.dst_suffix == "men" else TITLE.replace("men", a.dst_suffix)
    ds = list(meta.get("dataset_sources") or [])
    if BUNDLE not in ds:
        ds.append(BUNDLE)
    meta["dataset_sources"] = ds
    meta["enable_gpu"] = True
    meta["enable_internet"] = False
    meta.pop("id_no", None)

    dst.mkdir(parents=True, exist_ok=True)
    (dst / NB).write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    print("wrote %s" % dst)
    print("  cells %d -> %d   datasets=%d   bundle mounted: %s"
          % (n_before, len(nb["cells"]), len(ds), BUNDLE in ds))
    print("  meniscus weight = 0.10 on %s only" % ["Medial Meniscus", "Lateral Meniscus"])

    if not a.push:
        print("\nnot pushed (pass --push).")
        return 0
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        api.kernels_push(str(dst))
        print("pushed -> ppxl16/%s" % slug)
    except Exception as e:
        print("push failed: %s: %s" % (type(e).__name__, str(e)[:500]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
