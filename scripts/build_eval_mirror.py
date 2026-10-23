# -*- coding: utf-8 -*-
"""
Build kernel-eval-mirror: score ONE checkpoint twice -- plain and anatomical-mirrored -- so the
mirror's effect can be isolated locally.

WHY THIS SHAPE
    The public 0.949 notebook averages a plain view with an anatomical mirror of it. We do not
    know how much of its edge is the mirror ON OUR MODEL, so this measures exactly that and
    nothing else: same checkpoint, same k_eval, same res, same folds -- the ONLY difference
    between the two passes is RAPTOR_MIRROR, flipped between two main() calls in one session.

    Each pass writes its own raptor_gold_<tag>.npz. Averaging happens LOCALLY, so no ensemble
    logic enters the kernel and the comparison stays clean.

WHAT IT DELIBERATELY DOES NOT DO
    The public recipe bundles three changes: the mirror, a native 320px inference crop, and
    K94. Moving all three at once would repeat the C86 mistake -- a density/span effect and a
    normalisation effect confounded into one number. So this moves ONE variable.

    --folds 0 means gold_ids = the 58 competition studies whose 12 labels are all non-null --
    the same instrument every earlier raptor arm was scored on.
"""
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kernels" / "kernel-raptor-train" / "knee-mri-training-the-twelve-finding-model.py"
DST_DIR = ROOT / "kernels" / "kernel-eval-mirror"
DST = DST_DIR / "knee-mri-training-the-twelve-finding-model.py"

OLD_PRELUDE_MARK = "if not any(_a.startswith(\"--\") for _a in sys.argv[1:]):"

NEW_DRIVER = '''
if __name__ == "__main__":
    # ONE checkpoint, TWO passes. The only difference between them is RAPTOR_MIRROR, which the
    # trainer now reads at call time, so both passes run in this one session.
    import glob as _g, os as _o
    _c = _g.glob('/kaggle/input/**/raptor_ft_coatnet_v5_full_swa.pt', recursive=True)
    if not _c:
        raise SystemExit('v5 checkpoint not found -- is dreaddevelopment/raptor-knee-maxspan '
                         'attached?')
    _ARCH = 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
    _PLAN = [('ev_v5_plain', False), ('ev_v5_mirror', True)]
    for _tag, _mir in _PLAN:
        if _mir:
            _o.environ['RAPTOR_MIRROR'] = '1'
        else:
            _o.environ.pop('RAPTOR_MIRROR', None)
        sys.argv = ['train', '--labels', _parq, '--arch', _ARCH, '--res', '384',
                    '--bs', '2', '--grad_ckpt', '--ckpt', 'timm', '--init_ckpt', _c[0],
                    '--eval_only', '--k', '12', '--k_eval', '24', '--seed', '42',
                    '--tag', _tag, '--folds', '0', '--workers', '2']
        print('=' * 90, flush=True)
        print('[driver] %s   RAPTOR_MIRROR=%r' % (_tag, _o.environ.get('RAPTOR_MIRROR')),
              flush=True)
        print('=' * 90, flush=True)
        try:
            main()
        except SystemExit as _e:
            print('[driver] %s exited: %s' % (_tag, _e), flush=True)
        except Exception as _e:
            import traceback
            traceback.print_exc()
            print('[driver] %s FAILED: %r' % (_tag, _e), flush=True)
    print('[driver] receipts:', flush=True)
    for _f in sorted(_g.glob('/kaggle/working/raptor_evalonly_*.json')):
        print('  ' + _f, flush=True)
        print('  ' + open(_f).read().strip(), flush=True)
'''


def main():
    src = SRC.read_text(encoding="utf-8")
    # NOTE: only require RAPTOR_MIRROR. MIRROR_ENABLED is still DEFINED (the import-time
    # constant from the first patch) but no longer USED -- the switch moved to call time -- so
    # testing for its absence was wrong and rejected a correctly patched file.
    if "RAPTOR_MIRROR" not in src:
        raise SystemExit("FATAL: the source trainer is not the mirror-patched one")
    # replace the whole prelude argv guard with a plan-driven one (same approach as
    # build_eval_v5v9.py), then replace the __main__ hook with the two-pass driver
    import re
    m = re.search(r'if not any\(_a\.startswith\("--"\) for _a in sys\.argv\[1:\]\):.*?'
                  r'\nelse:\n    print\(.\[prelude\] explicit argv supplied.*?\n', src, re.S)
    if not m:
        raise SystemExit("FATAL: prelude argv block not found")
    src = src[:m.start()] + ("# prelude argv supplied by the two-pass driver at the bottom\n"
                            "pass\n") + src[m.end():]
    old_main = 'if __name__ == "__main__":\n    main()'
    if src.count(old_main) != 1:
        raise SystemExit("FATAL: __main__ hook not found exactly once")
    src = src.replace(old_main, NEW_DRIVER.strip(), 1)
    compile(src, "eval_mirror", "exec")

    DST_DIR.mkdir(parents=True, exist_ok=True)
    DST.write_text(src, encoding="utf-8")

    meta = {
        "id": "ppxl16/rsna-knee-eval-mirror",
        "title": "RSNA knee eval mirror",
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
            "flight0234/rsna-knee-hybrid-report-labels",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [],
        "enable_tpu": False,
    }
    (DST_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    json.loads((DST_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    print(f"built {DST.relative_to(ROOT)}  bytes={len(src.encode())}  "
          f"sha256={hashlib.sha256(src.encode()).hexdigest()[:16]}")
    print(f"  two passes: ev_v5_plain then ev_v5_mirror (RAPTOR_MIRROR=1)")
    print(f"  no machine_shape (A168 trap 3); GPU only")


if __name__ == "__main__":
    main()
