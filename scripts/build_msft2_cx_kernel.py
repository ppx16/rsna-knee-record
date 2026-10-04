"""Build `kernel-msft2-cx<N>`: fine-tune apoorvadixit's ConvNeXt-small fold N on TPU.

WHY A SECOND ARCHITECTURE, AND WHY NOW
    A163: a member earns its place by DECORRELATION, not by solo strength. msft2 (CoAtNet) is a
    0.9069 member but it is the same backbone family as the four Raptor views already in the
    0943 stack. ConvNeXt-small is a different architecture, so it is the cheapest genuinely new
    axis we have -- and `build_backbone` in the author script already special-cases `convnext`,
    so no code change is needed in the training body.

    The TPU window is the reason to do it THIS round: ~17 h remain and they refresh 2026-10-03,
    while GPU is exhausted until the same date. TPU hours are use-it-or-lose-it and cannot run
    the stacking kernel anyway, so a second member is the only thing they can buy.

WHY --fold EXISTS (2026-10-04)
    Fold 0 is the PROBE. If its SWA gold does not clear the pre-registered bar there is no point
    running folds 1-4, and TPU concurrency is 1 (D18) so they run strictly sequentially anyway.
    Building all five now costs nothing and removes the latency between "fold 0 cleared the bar"
    and "folds 1-4 are running".
    They are BUILT here, NOT pushed: pushing would occupy the single TPU slot for ~13 h before
    fold 0's result is known, which is exactly what the kill rule forbids.
    Pre-registration and kill rule: docs/PREREG-2026-10-04-convnext-member.md

WHAT IS DIFFERENT FROM kernel-msft2
    Only the prelude: the warm start, the arch, the tag and the fold. The author body is
    byte-identical, so the XLA grad-checkpoint shim that made v4 succeed still applies unchanged.

SAFETY RAIL
    The run prints a zero-shot gold58 macro-AUC before it trains. msft2 measured 0.8961 for
    CoAtNet against a reload bar of 0.8963. ConvNeXt is expected LOWER (different arch, and its
    head may not carry at all -> `head tensors carried: 0`). That is acceptable for a warm start,
    but if the printed number is near chance (< ~0.75) the checkpoint did not load the intended
    semantics and the run should be KILLED rather than left to burn the window.

Usage:
    python scripts/build_msft2_cx_kernel.py                  # fold 0, write files
    python scripts/build_msft2_cx_kernel.py --fold 3         # fold 3, write files
    python scripts/build_msft2_cx_kernel.py --fold 3 --push  # write + push
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
SRC = ROOT / "kernels" / "kernel-msft2"
CODE = "knee-mri-convnext-finetune.py"

ARCH = "convnext_small.in12k_ft_in1k"

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

DATASETS = [
    "dreaddevelopment/knee-raptor-corpus",
    "dreaddevelopment/knee-raptor-corpus-ext",
    "flight0234/rsna-knee-hybrid-report-labels",
    "ppxl16/rsna-knee-cv-folds",
    "apoorvadixit/rsna-knee-weights-v4cx",
]


def new_warm_block(fold: int) -> str:
    warm_glob = f"convnext_small.in12k__latL__v4cx{fold}_best.pt"
    tag = f"cx{fold}"
    return f"""_warm_c = glob.glob('/kaggle/input/**/{warm_glob}', recursive=True)
if not _warm_c:
    raise SystemExit('warm-start checkpoint {warm_glob} not found under /kaggle/input (dataset mounted?)')
_WARM = _warm_c[0]
print('[prelude] warm start -> %s' % _WARM, flush=True)
sys.argv = ['train',
            '--labels', _parq,
            '--arch', '{ARCH}',
            '--res', '384', '--epochs', '4', '--bs', '2', '--k', '12', '--k_eval', '24',
            '--grad_ckpt', '--ckpt', _WARM,
            # The v4cx checkpoint's head is a DIFFERENT architecture (pos_proj / freqs /
            # pool.att / head over a 5376-wide hidden layer) and NONE of its 11 head tensors
            # match RaptorClassifier, so v1 of this kernel exited at the head guard -- correctly,
            # since a silent head swap is precisely the 2026-09-28 bug. This flag turns
            # discarding the head into an explicit, logged decision: the backbone (342/342
            # tensors, missing 0, unexpected 0) is what we came for and the head trains from
            # scratch, and the zero-shot gate is skipped because a from-scratch head scores
            # ~chance by construction.
            '--allow_bad_warmstart',
            '--seed', '42', '--tag', '{tag}', '--folds', '5', '--fold', '{fold}',
            '--fold_file', _FOLD_FILE, '--workers', '0']"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    if not 0 <= a.fold <= 4:
        print("!! --fold must be 0..4 (v4cx ships five folds: v4cx0..v4cx4)")
        return 2

    slug = "rsna-knee-msft2-cx" if a.fold == 0 else f"rsna-knee-msft2-cx{a.fold}"
    title = "RSNA knee msft2 cx" if a.fold == 0 else f"RSNA knee msft2 cx{a.fold}"
    dst = ROOT / "kernels" / ("kernel-msft2-cx" if a.fold == 0 else f"kernel-msft2-cx{a.fold}")

    src_py = SRC / "knee-mri-training-the-twelve-finding-model.py"
    body = src_py.read_text(encoding="utf-8")
    if OLD_WARM_BLOCK not in body:
        print("!! prelude block not found verbatim in %s -- refusing to guess" % src_py)
        return 2
    body = body.replace(OLD_WARM_BLOCK, new_warm_block(a.fold), 1)
    assert ARCH in body, "substitution did not take"
    assert f"v4cx{a.fold}_best.pt" in body, "warm start not substituted"
    assert f"'--fold', '{a.fold}'" in body, "fold not substituted"
    assert "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'," not in body.split("sys.argv")[1][:400]

    meta = {
        "id": f"ppxl16/{slug}",
        "title": title,
        "code_file": CODE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": DATASETS,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "enable_tpu": True,
    }

    dst.mkdir(parents=True, exist_ok=True)
    (dst / CODE).write_text(body, encoding="utf-8")
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("fold %d -> %s" % (a.fold, dst))
    print("  slug=%s  tag=cx%d  warm=v4cx%d  tpu=%s gpu=%s  datasets=%d"
          % (slug, a.fold, a.fold, meta["enable_tpu"], meta["enable_gpu"], len(DATASETS)))
    lines = body.splitlines()
    for i, ln in enumerate(lines):
        if "sys.argv = ['train'" in ln:
            print("  --- injected argv ---")
            for x in lines[i:i + 7]:
                print("   ", x)
            break

    if not a.push:
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
