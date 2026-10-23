# ==== appended by scripts/build_reader_arm_tta.py: the Stack++ reader leg ====
# Stack++ inference (rank-space fold x TTA ensemble, overlapping windows, slice-grid TTA) blended into
# the stack with PER-LABEL weights. Source: romantamrazov/rsna-knee-dinosaur-v5, extracted verbatim.
# The stack's submission is written by the previous cell; this cell only ever REPLACES it on success.
import glob as _o_glob, os as _o_os, shutil as _o_shutil, subprocess as _o_sp, sys as _o_sys, time as _o_time
import pandas as _o_pd
OWN_W = __OWN_W__                 # parameterised; see --own-w
SYNOVITIS_W = 0.08   # per-label reduction on Synovitis; see docs/EXTRACT-2026-10-06g
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
