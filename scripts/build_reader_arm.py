"""Build `kernels/kernel-0943cfmsft2-rdr` — our best arm PLUS the public 2.5D ConvNeXt reader.

WHY (docs/STRATEGY-2026-10-06-two-new-levers.md). `goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944`
is the community 0.943 stack plus one independently trained reader, rank-blended at 30 %, and it scored
**0.944**. Its own weight sweep: 0.15 -> 0.944, 0.30 -> 0.944, 0.45 -> 0.942, stack alone 0.943.

⚠️ THE DIFFERENCE FROM EVERYTHING WE HAVE TRIED. Our `cn0` and `msft2` members are added INSIDE the
Raptor sub-blend -- a small share of one family. This adds a **separate 30 % leg of the final rank blend**,
from a **different architecture family** (2.5D ConvNeXt-Tiny + a 2-layer study transformer), that we did
not train. We have never tested that COMPOSITION.

WHAT IT PATCHES: nothing inside the stack. It appends one code cell AFTER the cell that writes
`/kaggle/working/submission.csv`, which:
  1. backs the stack's submission up;
  2. copies the reader's own source out of the mounted dataset (infer.py needs knee.py beside it);
  3. installs the shipped pylibjpeg wheels into a private dir (compressed-DICOM decoders, no internet);
  4. runs the reader as a SUBPROCESS against the competition test dir;
  5. percentile-rank-blends at OWN_W, re-ranks, and rewrites submission.csv;
  6. ⚠️ on ANY exception restores the backup, so a reader failure cannot lose the submission.

⚠️ RUNTIME IS THE REAL RISK. The author states the stack needs 6.5-8 h of the 9 h cap, and the reader
adds ~20 min. Their ordering -- STACK FIRST, reader second, stack's CSV kept as fallback -- is what makes
an overrun survivable, and it is copied deliberately. Do not reorder it.

Usage:
    python scripts/build_reader_arm.py --check
    python scripts/build_reader_arm.py --build
    python scripts/build_reader_arm.py --build --push
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
DST = ROOT / "kernels" / "kernel-0943cfmsft2-rdr"
NB = "rsna-knee-restructured-version-3.ipynb"
KID = "ppxl16/rsna-knee-0943cfmsft2-rdr"
READER_DS = "goodpjw2008/rsna-knee-2-5d-convnext-reader"
OWN_W = 0.30

CELL = '''# ==== appended by scripts/build_reader_arm.py: the public 2.5D ConvNeXt reader as a 30% leg ====
# Source: goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944 (public LB 0.944 at this weight).
# The stack's submission is written by the previous cell; this cell only ever REPLACES it on success.
import glob as _o_glob, os as _o_os, shutil as _o_shutil, subprocess as _o_sp, sys as _o_sys, time as _o_time
import pandas as _o_pd
OWN_W = __OWN_W__   # chosen before submitting, not tuned on the leaderboard
_o_pub, _o_bak = "/kaggle/working/submission.csv", "/kaggle/working/_stack_submission.csv"
_o_shutil.copy(_o_pub, _o_bak)
print("reader leg: stack submission backed up (%d bytes)" % _o_os.path.getsize(_o_bak), flush=True)
try:
    def _o_find_dir(name):
        for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/"]:
            for d in sorted(_o_glob.glob(pat)):
                if _o_os.path.exists(_o_os.path.join(d, name)):
                    return d.rstrip("/")
        raise FileNotFoundError(name)
    _o_assets = _o_find_dir("cnxt_v0_fold0.pt")
    _o_data = _o_find_dir("test_series.csv")
    _o_ck = sorted(_o_glob.glob(_o_assets + "/cnxt_v0_fold*.pt"))
    if not _o_ck:
        raise RuntimeError("no cnxt_v0_fold*.pt under %s" % _o_assets)
    print("reader leg: %d checkpoints from %s" % (len(_o_ck), _o_assets), flush=True)
    print("reader leg: test data dir %s" % _o_data, flush=True)

    # infer.py does `from knee import ...`, so the source must run from a dir holding all three files
    _o_src = "/kaggle/working/_own_src"
    _o_os.makedirs(_o_src, exist_ok=True)
    for _f in ("infer.py", "knee.py", "preprocess.py"):
        _p = _o_os.path.join(_o_assets, _f)
        if not _o_os.path.exists(_p):
            raise FileNotFoundError("%s missing from the reader dataset" % _f)
        _o_shutil.copy(_p, _o_os.path.join(_o_src, _f))

    # compressed-DICOM decoders, shipped IN the dataset so this works with internet off
    _o_env = dict(_o_os.environ)
    _o_target = "/kaggle/working/_own_env"
    _o_sp.run([_o_sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
               "--target", _o_target, "--find-links", _o_assets,
               "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
    _o_env["PYTHONPATH"] = _o_target + _o_os.pathsep + _o_env.get("PYTHONPATH", "")

    try:
        import gc as _o_gc, torch as _o_torch
        _o_gc.collect(); _o_torch.cuda.empty_cache()
        _o_free = [_o_torch.cuda.mem_get_info(i)[0] for i in range(_o_torch.cuda.device_count())]
        _o_env["CUDA_VISIBLE_DEVICES"] = str(max(range(len(_o_free)), key=lambda i: _o_free[i]))
        print("reader leg: free GPU memory (GB)", [round(f / 1e9, 1) for f in _o_free], flush=True)
    except Exception as _o_e:
        print("reader leg: GPU pick skipped:", repr(_o_e), flush=True)

    _o_t0 = _o_time.time()
    _o_sp.run([_o_sys.executable, _o_src + "/infer.py", _o_data, "test",
               "/kaggle/working/_own.csv"] + _o_ck, check=True, env=_o_env)
    print("reader leg: inference took %.0fs" % (_o_time.time() - _o_t0), flush=True)

    _o_own = _o_pd.read_csv("/kaggle/working/_own.csv", dtype={"StudyInstanceUID": str})
    _o_stack = _o_pd.read_csv(_o_bak, dtype={"StudyInstanceUID": str})
    _o_lab = [c for c in _o_stack.columns if c != "StudyInstanceUID"]
    _o_own = _o_own.set_index("StudyInstanceUID").loc[_o_stack.StudyInstanceUID].reset_index()
    assert _o_own[_o_lab].notna().all().all(), "reader predictions contain NaN"
    assert (_o_own[_o_lab].nunique() > 1).all(), "reader predictions are constant"
    _o_rk = lambda d: d[_o_lab].rank(method="average", pct=True)
    _o_final = _o_stack.copy()
    _o_final[_o_lab] = ((1 - OWN_W) * _o_rk(_o_stack) + OWN_W * _o_rk(_o_own)).rank(method="average", pct=True)
    _o_final.to_csv(_o_pub, index=False)
    _o_chg = float((_o_final[_o_lab].to_numpy() != _o_stack[_o_lab].to_numpy()).mean())
    print("reader leg: BLENDED %d checkpoints at w=%.2f in %.0fs; %.4f of label cells moved"
          % (len(_o_ck), OWN_W, _o_time.time() - _o_t0, _o_chg), flush=True)
except Exception as _o_err:
    _o_shutil.copy(_o_bak, _o_pub)
    print("reader leg FAILED -- the stack's submission is kept unchanged:", repr(_o_err), flush=True)
# ==== end reader leg ====
'''


def build_nb(own_w: float = OWN_W) -> dict:
    src = SRC / NB
    if not src.is_file():
        raise SystemExit("!! missing %s" % src)
    nb = json.loads(src.read_text(encoding="utf-8", errors="replace"))
    cells = nb["cells"]
    text = "\n".join("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                     else (c.get("source") or "") for c in cells)

    # the anchor must be the cell that PUBLISHES submission.csv -- asserting on the RESULT, not a guess
    pub = [i for i, c in enumerate(cells)
           if "os.replace(_tmp,_final)" in (("".join(c.get("source", []))
                                             if isinstance(c.get("source"), list)
                                             else (c.get("source") or "")))]
    if len(pub) != 1:
        raise SystemExit("ANCHOR FAILURE: expected exactly 1 publish cell, found %d" % len(pub))
    if "submission.csv" not in text:
        raise SystemExit("sanity: no submission.csv anywhere in the source notebook")
    pos = pub[0] + 1

    body = CELL.replace("__OWN_W__", repr(float(own_w)))
    new_cell = {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": body.splitlines(keepends=True)}
    if any("build_reader_arm" in (("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                                   else (c.get("source") or ""))) for c in cells):
        raise SystemExit("the reader cell is already present -- refusing to append twice")
    cells.insert(pos, new_cell)
    nb["cells"] = cells
    print("   appended the reader cell at index %d (publish cell was %d)" % (pos, pub[0]))
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--push", action="store_true")
    # ⚠️ WHY THIS FLAG EXISTS. `OWN_W` used to be a module constant, so building the OTHER point of the
    # author's weight plateau meant hand-editing this file -- and hand-edits under time pressure are how a
    # pre-registered variant silently stops being the variant that was registered. The author's own sweep is
    # 0.15 -> 0.944, 0.30 -> 0.944, 0.45 -> 0.942, so 0.15 is an equally supported point; and their sweep was
    # on the COMMUNITY stack while ours is the 0943cfmsft2 variant, so the better point may differ.
    # The default stays 0.30, which is what submission 56872230 already used, so nothing already built moves.
    ap.add_argument("--own-w", type=float, default=OWN_W,
                    help="reader-leg weight; default %(default)s (the value already submitted)")
    a = ap.parse_args()

    w = float(a.own_w)
    if not 0.0 < w < 1.0:
        raise SystemExit("--own-w must be strictly between 0 and 1 (got %r)" % (a.own_w,))
    is_default = abs(w - OWN_W) < 1e-9
    tag = "" if is_default else "-w%02d" % round(w * 100)
    dst = DST if is_default else (ROOT / "kernels" / ("kernel-0943cfmsft2-rdr-w%02d" % round(w * 100)))
    kid = KID if is_default else ("ppxl16/rsna-knee-0943cfmsft2-rdr-w%02d" % round(w * 100))
    print("reader leg: OWN_W=%r -> %s (%s)" % (w, dst.relative_to(ROOT), kid))

    nb = build_nb(w)
    if a.build:
        dst.mkdir(parents=True, exist_ok=True)
        # ⚠️ ensure_ascii=True IS LOAD-BEARING, not a style choice. `KaggleApi.kernels_push` reads the
        # code_file with `f.read()` -- the SYSTEM default encoding, which is GBK here -- so a notebook
        # written as raw UTF-8 dies INSIDE the SDK, before anything is uploaded:
        #   UnicodeDecodeError: 'gbk' codec can't decode byte 0x8b in position 3768
        # Measured on this exact artefact: 311,810 bytes, 147 of them non-ASCII (the appended cell's
        # comments). Escaping to \uXXXX is byte-identical in meaning, leaves the probes below working (they
        # check the PARSED cells, not the raw text), and removes the dependency on the operator remembering
        # PYTHONUTF8=1 -- which is exactly how the first push attempt was lost.
        (dst / NB).write_text(json.dumps(nb, ensure_ascii=True), encoding="utf-8")

    text = "\n".join("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                     else (c.get("source") or "") for c in nb["cells"])
    probes = [
        ("build_reader_arm.py", "the reader cell is present"),
        ("OWN_W = %r" % float(w), "the weight"),
        ("_stack_submission.csv", "the fallback backup"),
        ('_o_shutil.copy(_o_bak, _o_pub)', "the restore-on-failure path"),
        ("cnxt_v0_fold0.pt", "the checkpoint locator"),
        ("test_series.csv", "the test-dir locator"),
        ('rank(method="average", pct=True)', "percentile-rank blending"),
        ('assert _o_own[_o_lab].notna().all().all()', "the NaN guard"),
    ]
    missing = [why for probe, why in probes if probe not in text]
    if missing:
        raise SystemExit("!! built notebook is missing: %s" % missing)
    tmp = ROOT / "work" / "_reader_arm_cell.py"
    tmp.write_text(CELL.replace("__OWN_W__", repr(float(w))), encoding="utf-8")
    py_compile.compile(str(tmp), doraise=True)
    print("PATTERN CHECK OK: cell compiles, all %d probes present, %d cells total"
          % (len(probes), len(nb["cells"])))

    if not a.build:
        print("(--check only; nothing written to %s)" % dst.relative_to(ROOT))
        return 0

    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = kid
    meta["title"] = "RSNA knee 0943cfmsft2 rdr" + tag
    meta["code_file"] = NB
    meta["enable_gpu"] = True
    meta["enable_tpu"] = False
    meta["enable_internet"] = False
    ds = list(meta.get("dataset_sources") or [])
    if READER_DS not in ds:
        ds.append(READER_DS)
    meta["dataset_sources"] = ds
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("built %s  datasets=%d (reader dataset %s)"
          % (dst.relative_to(ROOT), len(ds), "added" if READER_DS in ds else "MISSING"))

    if a.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.kernels_push(str(dst))
        print("PUSHED %s" % kid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
