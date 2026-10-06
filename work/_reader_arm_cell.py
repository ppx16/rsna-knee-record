# ==== appended by scripts/build_reader_arm.py: the public 2.5D ConvNeXt reader as a 30% leg ====
# Source: goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944 (public LB 0.944 at this weight).
# The stack's submission is written by the previous cell; this cell only ever REPLACES it on success.
import glob as _o_glob, os as _o_os, shutil as _o_shutil, subprocess as _o_sp, sys as _o_sys, time as _o_time
import pandas as _o_pd
OWN_W = 0.3   # chosen before submitting, not tuned on the leaderboard
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
