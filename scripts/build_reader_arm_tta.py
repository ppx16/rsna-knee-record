"""Build `kernels/kernel-0943cfmsft2-rdrtta` — the reader leg with Stack++'s inference. NOT FIRED.

GATED ON THE BASE LEG. `docs/VERIFY-2026-10-06-reader-leg-is-clean.md` fixes the entry condition: this arm
only runs if submission `56872230` (the plain reader leg) moves the **in-block percentile** by >= 3 points
against **16.03 %**. Building it now is not preparation for its own sake -- it is the **verification** of
last round's extraction: three extracted files have to compose into a kernel that actually runs, and
finding out after the base result lands would cost the decision, not just wall-clock.

WHAT DIFFERS FROM `build_reader_arm.py` (three things only):
  1. the reader's three source files come from `ppxl16/rsna-knee-stackpp-reader-src` (ours, verified) instead
     of from goodpjw2008's dataset;
  2. `RSNA_READER_TTA="mid,edge"` is set for the 3-checkpoint case, as the author's caller does;
  3. the blend uses **per-label weights**: `Synovitis = 0.08`, everything else `0.30`.

⚠️ Item 3 is the ONLY fitted quantity in the change set, and its evidence is 58 gold studies whose paired CI
is +/-0.021. It is adopted **verbatim** from the author -- NOT tuned by us -- and the prereg names it as the
first thing to vary if this arm is ever iterated.

Usage:
    python scripts/build_reader_arm_tta.py --check
    python scripts/build_reader_arm_tta.py --build [--push]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import py_compile
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kernels" / "kernel-0943cfmsft2"
DST = ROOT / "kernels" / "kernel-0943cfmsft2-rdrtta"
NB = "rsna-knee-restructured-version-3.ipynb"
KID = "ppxl16/rsna-knee-0943cfmsft2-rdrtta"
CKPT_DS = "goodpjw2008/rsna-knee-2-5d-convnext-reader"
SRC_DS = "ppxl16/rsna-knee-stackpp-reader-src"

CELL = '''# ==== appended by scripts/build_reader_arm_tta.py: the Stack++ reader leg ====
# Stack++ inference (rank-space fold x TTA ensemble, overlapping windows, slice-grid TTA) blended into
# the stack with PER-LABEL weights. Source: romantamrazov/rsna-knee-dinosaur-v5, extracted verbatim.
# The stack's submission is written by the previous cell; this cell only ever REPLACES it on success.
import glob as _o_glob, os as _o_os, shutil as _o_shutil, subprocess as _o_sp, sys as _o_sys, time as _o_time
import pandas as _o_pd
OWN_W = 0.30                     # the author's global weight
SYNOVITIS_W = __SYNOVITIS_W__   # per-label reduction on Synovitis; see docs/EXTRACT-2026-10-06g
#   The author uses 0.08, justified by the observation that most radiology reports never mention
#   Synovitis, so report-derived labels teach it wrongly. That justification rests on 58 gold studies
#   whose paired CI is +/-0.021, which is why this constant is a parameter here rather than a fact.
_o_pub, _o_bak = "/kaggle/working/submission.csv", "/kaggle/working/_stack_submission.csv"
_o_shutil.copy(_o_pub, _o_bak)
print("reader leg (tta): stack submission backed up (%d bytes)" % _o_os.path.getsize(_o_bak), flush=True)
try:
    def _o_find_dir(name):
        for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/"]:
            for d in sorted(_o_glob.glob(pat)):
                if _o_os.path.exists(_o_os.path.join(d, name)):
                    return d.rstrip("/")
        raise FileNotFoundError(name)
    _o_ck_dir = _o_find_dir("cnxt_v0_fold0.pt")
    _o_src_dir = _o_find_dir("preprocess.py")
    _o_data = _o_find_dir("test_series.csv")
    _o_ck = sorted(_o_glob.glob(_o_ck_dir + "/cnxt_v0_fold*.pt"))
    if len(_o_ck) < 3:
        raise RuntimeError("Stack++ requires >=3 reader checkpoints, found %d" % len(_o_ck))
    print("reader leg (tta): %d checkpoints; src=%s; data=%s"
          % (len(_o_ck), _o_src_dir, _o_data), flush=True)

    _o_src = "/kaggle/working/own_src"
    _o_os.makedirs(_o_src, exist_ok=True)
    for _f in ("infer.py", "knee.py", "preprocess.py"):
        _p = _o_os.path.join(_o_src_dir, _f)
        if not _o_os.path.exists(_p):
            raise FileNotFoundError("%s missing from %s" % (_f, _o_src_dir))
        _o_shutil.copy(_p, _o_os.path.join(_o_src, _f))

    _o_env = dict(_o_os.environ)
    _o_target = "/kaggle/working/_own_env"
    _o_sp.run([_o_sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
               "--target", _o_target, "--find-links", _o_ck_dir,
               "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
    _o_env["PYTHONPATH"] = _o_target + _o_os.pathsep + _o_env.get("PYTHONPATH", "")
    # the author's caller sets "mid,edge" when at most three checkpoints are available
    _o_env["RSNA_READER_TTA"] = "mid,edge" if len(_o_ck) <= 3 else "mid"

    try:
        import gc as _o_gc, torch as _o_torch
        _o_gc.collect(); _o_torch.cuda.empty_cache()
        _o_free = [_o_torch.cuda.mem_get_info(i)[0] for i in range(_o_torch.cuda.device_count())]
        _o_env["CUDA_VISIBLE_DEVICES"] = str(max(range(len(_o_free)), key=lambda i: _o_free[i]))
        print("reader leg (tta): free GPU memory (GB)", [round(f / 1e9, 1) for f in _o_free], flush=True)
    except Exception as _o_e:
        print("reader leg (tta): GPU pick skipped:", repr(_o_e), flush=True)

    _o_t0 = _o_time.time()
    _o_sp.run([_o_sys.executable, _o_src + "/infer.py", _o_data, "test",
               "/kaggle/working/_own.csv"] + _o_ck, check=True, env=_o_env)
    print("reader leg (tta): inference took %.0fs" % (_o_time.time() - _o_t0), flush=True)

    _o_own = _o_pd.read_csv("/kaggle/working/_own.csv", dtype={"StudyInstanceUID": str})
    _o_stack = _o_pd.read_csv(_o_bak, dtype={"StudyInstanceUID": str})
    _o_lab = [c for c in _o_stack.columns if c != "StudyInstanceUID"]
    _o_own = _o_own.set_index("StudyInstanceUID").loc[_o_stack.StudyInstanceUID].reset_index()
    assert _o_own[_o_lab].notna().all().all(), "reader predictions contain NaN"
    assert (_o_own[_o_lab].nunique() > 1).all(), "reader predictions are constant"
    _o_rk = lambda d: d[_o_lab].rank(method="average", pct=True)
    _o_rs, _o_ro = _o_rk(_o_stack), _o_rk(_o_own)
    _o_final = _o_stack.copy()
    for _c in _o_lab:
        _w = SYNOVITIS_W if _c == "Synovitis" else OWN_W
        _o_final[_c] = ((1.0 - _w) * _o_rs[_c] + _w * _o_ro[_c]).rank(method="average", pct=True)
    _o_final.to_csv(_o_pub, index=False)
    print("reader leg (tta): BLENDED %d ckpts, Synovitis_w=%.2f other_w=%.2f, %.0fs"
          % (len(_o_ck), SYNOVITIS_W, OWN_W, _o_time.time() - _o_t0), flush=True)
except Exception as _o_err:
    _o_shutil.copy(_o_bak, _o_pub)
    print("reader leg (tta) FAILED -- the stack's submission is kept unchanged:", repr(_o_err), flush=True)
# ==== end reader leg (tta) ====
'''


def build_nb(synovitis_w: float = 0.08) -> dict:
    src = SRC / NB
    if not src.is_file():
        raise SystemExit("!! missing %s" % src)
    nb = json.loads(src.read_text(encoding="utf-8", errors="replace"))
    cells = nb["cells"]
    pub = [i for i, c in enumerate(cells)
           if "os.replace(_tmp,_final)" in (("".join(c.get("source", []))
                                             if isinstance(c.get("source"), list)
                                             else (c.get("source") or "")))]
    if len(pub) != 1:
        raise SystemExit("ANCHOR FAILURE: expected exactly 1 publish cell, found %d" % len(pub))
    if any("build_reader_arm_tta" in (("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                                       else (c.get("source") or ""))) for c in cells):
        raise SystemExit("the tta reader cell is already present -- refusing to append twice")
    body = CELL.replace("__SYNOVITIS_W__", repr(float(synovitis_w)))
    cells.insert(pub[0] + 1, {"cell_type": "code", "execution_count": None, "metadata": {},
                              "outputs": [], "source": body.splitlines(keepends=True)})
    nb["cells"] = cells
    print("   appended the tta reader cell at index %d (publish cell was %d, Synovitis_w=%.2f)"
          % (pub[0] + 1, pub[0], synovitis_w))
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--synovitis-w", type=float, default=0.08,
                    help="the per-label reduction on Synovitis. 0.08 is the author's value; 0.30 disables "
                         "the reduction entirely and is the pre-registered FIRST variation if this arm is "
                         "ever iterated (docs/VERIFY-2026-10-06-reader-leg-is-clean.md)")
    a = ap.parse_args()

    nb = build_nb(a.synovitis_w)
    text = "\n".join("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                     else (c.get("source") or "") for c in nb["cells"])
    probes = [("build_reader_arm_tta.py", "the cell is present"),
              ("SYNOVITIS_W = %r" % float(a.synovitis_w), "the per-label weight"),
              ("RSNA_READER_TTA", "the TTA mode"),
              ("_stack_submission.csv", "the fallback backup"),
              ("_o_shutil.copy(_o_bak, _o_pub)", "the restore path"),
              ("len(_o_ck) < 3", "the checkpoint-count guard"),
              ('rank(method="average", pct=True)', "rank blending")]
    missing = [w for p, w in probes if p not in text]
    if missing:
        raise SystemExit("!! built notebook is missing: %s" % missing)
    tmp = ROOT / "work" / "_rdrtta_cell.py"
    tmp.write_text(CELL.replace("__SYNOVITIS_W__", repr(float(a.synovitis_w))), encoding="utf-8")
    py_compile.compile(str(tmp), doraise=True)
    print("PATTERN CHECK OK: cell compiles, all %d probes present" % len(probes))

    if not a.build:
        print("(--check only)")
        return 0

    dst = DST if abs(a.synovitis_w - 0.08) < 1e-9 else ROOT / "kernels" / (
        "kernel-0943cfmsft2-rdrtta-sy%02d" % round(a.synovitis_w * 100))
    kid = KID if dst is DST else "ppxl16/" + dst.name.replace("kernel-", "rsna-knee-")
    dst.mkdir(parents=True, exist_ok=True)
    # ⚠️ ensure_ascii=True IS LOAD-BEARING -- see the long note in scripts/build_reader_arm.py.
    # `KaggleApi.kernels_push` reads the code_file with the SYSTEM default encoding (GBK here), so a
    # notebook written as raw UTF-8 raises UnicodeDecodeError inside the SDK before anything is uploaded.
    (dst / NB).write_text(json.dumps(nb, ensure_ascii=True), encoding="utf-8")
    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = kid
    meta["title"] = dst.name.replace("kernel-", "RSNA knee ")
    meta["code_file"] = NB
    meta["enable_gpu"] = True
    meta["enable_tpu"] = False
    meta["enable_internet"] = False
    ds = list(meta.get("dataset_sources") or [])
    for r in (CKPT_DS, SRC_DS):
        if r not in ds:
            ds.append(r)
    meta["dataset_sources"] = ds
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("built %s (%d datasets, id %s)" % (dst.relative_to(ROOT), len(ds), kid))
    if a.push:
        print("REFUSING to push: this arm is GATED on the base reader leg (prereg). "
              "Push deliberately, by hand, once that gate opens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
