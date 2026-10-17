# -*- coding: utf-8 -*-
"""
Build the eval-only kernel for 方向一/方向二's prerequisite measurement.

It reuses the ALREADY-PATCHED trainer (kernels/kernel-raptor-train) unchanged below the
prelude, and replaces only the argv prelude + the __main__ hook, so the training code
stays byte-identical and comparable.

It scores FOUR combinations in ONE session:
  v5 maxspan  @ k_eval 24   <- control: must reproduce ~0.9214 (proves loader+geometry)
  v5 maxspan  @ k_eval 62   <- the author's scoring protocol (0.928 LB claim)
  v9 finespacing @ k_eval 24  <- THE new member, comparable to existing arms
  v9 finespacing @ k_eval 62  <- its protocol faithful number

No training happens: --eval_only runs evaluate() on the gold split (--folds 0 -> the 58
competition train studies whose 12 labels are all non-null) and exits.
"""
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kernels" / "kernel-raptor-train" / "knee-mri-training-the-twelve-finding-model.py"
DST_DIR = ROOT / "kernels" / "kernel-eval-v5v9"
DST = DST_DIR / "knee-mri-training-the-twelve-finding-model.py"

OLD_PRELUDE = """if not any(_a.startswith("--") for _a in sys.argv[1:]):
    sys.argv = ['train',
                '--labels', _parq,
                '--arch', 'vit_small_patch16_224',
                '--res', '224', '--epochs', '12', '--bs', '8',
                '--k', '12', '--k_eval', '24',
                '--grad_ckpt', '--seed', '42', '--tag', 'v1']
else:
    print('[prelude] explicit argv supplied -> using it: %s' % ' '.join(sys.argv[1:]), flush=True)"""

NEW_PRELUDE = '''# ---- eval-only prelude (built by work/build_eval_v5v9.py) ------------------------------
# The released Raptor checkpoints nest their tensors under "model", which the original
# load_raptor did not unwrap: it produced an EMPTY dict, load_state_dict(strict=False)
# accepted it, and the run continued from timm weights while printing a success line.
# Verified against the real files: old loader -> 0 tensors, patched loader -> 478 + 8.
_pick = {}
for _m in ("v5_full_swa", "v9_full"):
    _g = glob.glob('/kaggle/input/**/raptor_ft_coatnet_%s.pt' % _m, recursive=True)
    if _g:
        _pick[_m] = _g[0]
print('[prelude] checkpoints found: %s' % _pick, flush=True)
if len(_pick) < 2:
    raise SystemExit('need BOTH raptor_ft_coatnet_v5_full_swa.pt and '
                     'raptor_ft_coatnet_v9_full.pt under /kaggle/input -- are the '
                     'raptor-knee-maxspan and raptor-knee-finespacing datasets attached?')

_ARCH = 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'   # ck['arch'] of both files, read off the artefact
_EVAL_PLAN = []
for _tag, _m, _ke in (('ev_v5maxspan',   'v5_full_swa', '24'),
                      ('ev_v5maxspan62', 'v5_full_swa', '62'),
                      ('ev_v9fine',      'v9_full',     '24'),
                      ('ev_v9fine62',    'v9_full',     '62')):
    _EVAL_PLAN.append(['train', '--labels', _parq,
                       '--arch', _ARCH, '--res', '384', '--bs', '2', '--grad_ckpt',
                       '--ckpt', 'timm',                 # let load_raptor short-circuit
                       '--init_ckpt', _pick[_m],         # ...then load the FULL state, head included
                       '--eval_only',
                       '--k', '12', '--k_eval', _ke, '--seed', '42', '--tag', _tag,
                       '--folds', '0', '--workers', '2'])
sys.argv = _EVAL_PLAN[0]
print('[prelude] %d evals queued: %s' % (len(_EVAL_PLAN), [p[-1] for p in _EVAL_PLAN]), flush=True)'''

OLD_MAIN = '''if __name__ == "__main__":
    main()'''

NEW_MAIN = '''if __name__ == "__main__":
    # One GPU session, four measurements. Each entry re-enters main() with its own argv.
    for _i, _argv in enumerate(_EVAL_PLAN):
        sys.argv = _argv
        print("=" * 90, flush=True)
        print("[driver] %d/%d %s" % (_i + 1, len(_EVAL_PLAN), _argv[-1]), flush=True)
        print("=" * 90, flush=True)
        try:
            main()
        except SystemExit as _e:
            print("[driver] %s exited: %s" % (_argv[-1], _e), flush=True)
        except Exception as _e:
            import traceback
            traceback.print_exc()
            print("[driver] %s FAILED: %r" % (_argv[-1], _e), flush=True)
    import glob as _g2
    print("[driver] outputs:", flush=True)
    for _f in sorted(_g2.glob("/kaggle/working/raptor_evalonly_*.json")):
        print("   " + _f, flush=True)
        print("   " + open(_f).read().strip(), flush=True)'''


def main():
    src = SRC.read_text(encoding="utf-8")
    if OLD_PRELUDE not in src:
        raise SystemExit("FATAL: prelude anchor not found in the patched trainer")
    out = src.replace(OLD_PRELUDE, NEW_PRELUDE, 1)
    if OLD_MAIN not in out:
        raise SystemExit("FATAL: __main__ anchor not found")
    out = out.replace(OLD_MAIN, NEW_MAIN, 1)

    DST_DIR.mkdir(parents=True, exist_ok=True)
    DST.write_text(out, encoding="utf-8")
    compile(out, str(DST), "exec")          # compile(), not ast.parse

    meta = {
        "id": "ppxl16/rsna-knee-eval-v5v9",
        "title": "RSNA knee eval v5v9",
        "code_file": DST.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [
            "dreaddevelopment/knee-raptor-corpus",
            "dreaddevelopment/knee-raptor-corpus-ext",
            "dreaddevelopment/raptor-knee-maxspan",
            "dreaddevelopment/raptor-knee-finespacing",
            "flight0234/rsna-knee-hybrid-report-labels",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [],
        "enable_tpu": False,
    }
    # NOTE: deliberately NO machine_shape. A168 trap 3: an inherited machine_shape makes
    # enable_gpu:true still count as a TPU session and the push returns kernelId 0.
    (DST_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    json.loads((DST_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))

    print(f"built {DST.relative_to(ROOT)}  bytes={len(out.encode())}  "
          f"sha256={hashlib.sha256(out.encode()).hexdigest()[:16]}")
    print(f"built {DST_DIR.relative_to(ROOT)}/kernel-metadata.json  "
          f"(GPU T4, no machine_shape, 5 datasets, competition attached)")


if __name__ == "__main__":
    main()
