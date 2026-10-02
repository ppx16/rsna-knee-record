"""Build kernel-msft2-cx: fine-tune apoorvadixit's ConvNeXt-small (v4cx0) on TPU.

WHY A SECOND ARCHITECTURE, AND WHY NOW
    A163: a member earns its place by DECORRELATION, not by solo strength. msft2 (CoAtNet) is a
    0.9069 member but it is the same backbone family as the four Raptor views already in the
    0943 stack. ConvNeXt-small is a different architecture, so it is the cheapest genuinely new
    axis we have -- and `build_backbone` in the author script already special-cases `convnext`,
    so no code change is needed in the training body.

    The TPU window is the reason to do it THIS round: ~17 h remain and they refresh 2026-10-03,
    while GPU is exhausted until the same date. TPU hours are use-it-or-lose-it and cannot run
    the stacking kernel anyway, so a second member is the only thing they can buy.

WHAT IS DIFFERENT FROM kernel-msft2
    Only the prelude: the warm start, the arch, the tag. The author body is byte-identical, so
    the XLA grad-checkpoint shim that made v4 succeed still applies unchanged.

SAFETY RAIL
    The run prints a zero-shot gold58 macro-AUC before it trains. msft2 measured 0.8961 for
    CoAtNet against a reload bar of 0.8963. ConvNeXt is expected LOWER (different arch, and its
    head may not carry at all -> `head tensors carried: 0`). That is acceptable for a warm start,
    but if the printed number is near chance (< ~0.75) the checkpoint did not load the intended
    semantics and the run should be KILLED rather than left to burn the window.

Usage:
    python scripts/build_msft2_cx_kernel.py            # write files
    python scripts/build_msft2_cx_kernel.py --push     # write + push to Kaggle
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels" / "kernel-msft2"
DST = ROOT / "kernels" / "kernel-msft2-cx"
SLUG = "rsna-knee-msft2-cx"
TITLE = "RSNA knee msft2 cx"          # Kaggle derives the slug from the TITLE
CODE = "knee-mri-convnext-finetune.py"

ARCH = "convnext_small.in12k_ft_in1k"
WARM_GLOB = "convnext_small.in12k__latL__v4cx0_best.pt"
TAG = "cx0"

OLD_WARM_BLOCK = """_warm_c = glob.glob('/kaggle/input/**/raptor_ft_coatnet_v5_full_swa.pt', recursive=True)
if not _warm_c:
    raise SystemExit('warm-start checkpoint raptor_ft_coatnet_v5_full_swa.pt not found under /kaggle/input (dataset mounted?)')
_WARM = _warm_c[0]
print('[prelude] warm start -> %s' % _WARM, flush=True)
sys.argv = ['train',
            '--labels', _parq,
            '--arch', 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',
            '--res', '384', '--epochs', '4', '--bs', '2', '--k', '12', '--k_eval', '24',
            '--grad_ckpt', '--ckpt', _WARM,
            '--seed', '42', '--tag', 'msft0', '--folds', '5', '--fold', '0',
            '--fold_file', _FOLD_FILE, '--workers', '0']"""

NEW_WARM_BLOCK = f"""_warm_c = glob.glob('/kaggle/input/**/{WARM_GLOB}', recursive=True)
if not _warm_c:
    raise SystemExit('warm-start checkpoint {WARM_GLOB} not found under /kaggle/input (dataset mounted?)')
_WARM = _warm_c[0]
print('[prelude] warm start -> %s' % _WARM, flush=True)
sys.argv = ['train',
            '--labels', _parq,
            '--arch', '{ARCH}',
            '--res', '384', '--epochs', '4', '--bs', '2', '--k', '12', '--k_eval', '24',
            '--grad_ckpt', '--ckpt', _WARM,
            '--seed', '42', '--tag', '{TAG}', '--folds', '5', '--fold', '0',
            '--fold_file', _FOLD_FILE, '--workers', '0']"""

META = {
    "id": f"ppxl16/{SLUG}",
    "title": TITLE,
    "code_file": CODE,
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "dataset_sources": [
        "dreaddevelopment/knee-raptor-corpus",
        "dreaddevelopment/knee-raptor-corpus-ext",
        "flight0234/rsna-knee-hybrid-report-labels",
        "ppxl16/rsna-knee-cv-folds",
        "apoorvadixit/rsna-knee-weights-v4cx",
    ],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "enable_tpu": True,
}


def main() -> int:
    src_py = SRC / "knee-mri-training-the-twelve-finding-model.py"
    body = src_py.read_text(encoding="utf-8")
    if OLD_WARM_BLOCK not in body:
        print("!! prelude block not found verbatim in %s -- refusing to guess" % src_py)
        return 2
    body = body.replace(OLD_WARM_BLOCK, NEW_WARM_BLOCK, 1)
    # a mismatched title/id pair silently pushes to the wrong slug
    assert ARCH in body and WARM_GLOB in body, "substitution did not take"
    assert "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'," not in body.split("sys.argv")[1][:400]

    DST.mkdir(parents=True, exist_ok=True)
    (DST / CODE).write_text(body, encoding="utf-8")
    (DST / "kernel-metadata.json").write_text(json.dumps(META, indent=2) + "\n", encoding="utf-8")
    print("wrote %s" % (DST / CODE))
    print("wrote %s" % (DST / "kernel-metadata.json"))
    lines = body.splitlines()
    for i, ln in enumerate(lines):
        if "sys.argv = ['train'" in ln:
            print("\n--- injected argv ---")
            print("\n".join(lines[i:i + 8]))
            break

    if "--push" not in sys.argv:
        print("\nnot pushed (pass --push).")
        return 0
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        api.kernels_push(str(DST))
        print("pushed -> ppxl16/%s" % SLUG)
    except Exception as e:
        print("push failed: %s: %s" % (type(e).__name__, str(e)[:500]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
