# -*- coding: utf-8 -*-
"""Build the TPU (torch_xla) port of the CV fold kernel.

WHY THIS PORT IS SMALL, AND WHY THE BASELINE'S TPU PORT WAS NOT
    `kernel-raptor-train/` (the CV instrument) reads an ALREADY-DECODED corpus
    memmap (`all_vols.npy`, 3200x44x336x336 uint8) and runs one process. The
    baseline TPU port (`build_tpu_train.py`) could not do that: it had to decode
    DICOM in the parent into a ~34.5 GB slice cache and then `xmp.spawn` 8
    workers off it, and v20 died in a SIGTERM cascade (HANDOFF 30.2). Every
    mechanism that made that port hard is ABSENT here:

        no DICOM decode        -> no 34.5 GB parent
        no xmp.spawn           -> no fork of a process holding a live OpenMP pool
        memmap corpus          -> page cache, not RSS
        one process, workers=0 -> no worker pool at all

    So the port is four mechanisms plus hygiene, and nothing else:

        X1  dev  "cuda" -> "xla" (single process, one XLA device)
        X2  `torch.autocast("cuda", ...)` -> no-op context when not on CUDA.
            The source already gates on `enabled=dev == "cuda"`, but it still
            CONSTRUCTS `torch.autocast("cuda", ...)`, and on a CPU-only torch
            build that is not a safe no-op. Replace the call, keep the flag.
        X3  `opt.step()` -> `xm.optimizer_step()` + `xm.mark_step()`. XLA is
            lazy: without a mark_step the graph grows across steps.
        X4  `xm.mark_step()` BEFORE the two trained-state `.cpu()` transfers
            (best_state, topk). This is the exact hazard class that killed the
            baseline port ("native abort at the single .cpu() that follows a
            device-side accumulation"); here the transfer is small, but the
            ordering fix is free.

    Hygiene: `pin_memory=False` (no CUDA on a TPU VM), and OMP/MKL/OPENBLAS
    threads pinned to 1 before numpy is imported -- HANDOFF 30.2's root cause was
    an inherited live OpenMP worker pool, and pinning costs nothing because the
    TPU work is device-side.

WHY A SINGLE PROCESS AND NOT 8 CORES
    Data parallelism would need `xmp.spawn`, which is precisely what crashed the
    baseline port, and it changes the effective batch size (world * BATCH). A
    paired CV comparison (res 224 vs res 288) is only valid if BOTH arms ran the
    same configuration, so the port must reproduce the GPU arm's semantics
    exactly: one device, one batch, one optimizer. Throughput is bought with a
    separate quota (TPU ~20 h/week vs GPU ~30 h/week), not with concurrency.

USAGE
    python scripts/build_tpu_cv.py --probe        # 1 epoch / 160 studies, the smoke
    python scripts/build_tpu_cv.py --folds 5 --fold 3 --tag r2883 --res 288
    python scripts/build_tpu_cv.py --probe --push # build then push to Kaggle
"""
from __future__ import annotations


# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded
# `print("... \u26a0 ...")` raises UnicodeEncodeError at that line and kills the script MID-REPORT, while
# everything printed earlier is already on screen -- so the run looks complete. Self-contained on purpose:
# it imports `sys` itself and depends on nothing else in this tree.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream: leave it alone, prints still degrade
    pass
# ----------------------------------------------------------------------------------------------------
import argparse
import ast
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
from kpaths import kdir          # kernels/ layout, see kpaths.py
sys.path.insert(0, str(ROOT / "scripts"))

from build_cv_kernel import build  # noqa: E402  (the exact GPU builder, reused)

MACHINE_SHAPE = "Tpu1VmV38"


# --------------------------------------------------------------------------- #
# Prepend: env + deps. Must be the FIRST thing in the file (before numpy/torch).
# --------------------------------------------------------------------------- #
DEPS_BLOCK = '''# ==== TPU port: env + deps (MUST run before numpy/torch are imported) ====
# HANDOFF 30.2 root cause: a live OpenMP/MKL worker pool inherited across fork
# corrupts the children -> native crash, no Python trace. We do not fork, but
# pinning the pools costs nothing here (all real work is device-side) and removes
# the failure mode from the DataLoader/memmap path too.
import os as _os_p, sys as _sys_p, subprocess as _sp_p
for _v_p in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os_p.environ[_v_p] = "1"
# The torch-xla image is not the GPU image: `import timm` / `sklearn.metrics` are
# module-level in the author script below and unguarded, so a missing wheel is a
# hard crash at import. Install only what is actually absent (a plain
# `pip install timm` can drag torchvision around and destabilise torch-xla).
for _imp_p, _pkg_p in (("timm", "timm"), ("sklearn", "scikit-learn")):
    try:
        __import__(_imp_p)
    except Exception as _e_p:
        print("[tpu] installing %s (%s)" % (_pkg_p, repr(_e_p)[:80]), flush=True)
        _sp_p.run([_sys_p.executable, "-m", "pip", "install", "--quiet", _pkg_p],
                  check=True)
        __import__(_imp_p)
print("[tpu] deps ok (timm+sklearn importable, BLAS threads pinned to 1)", flush=True)
# ==== end TPU port env ====

'''

# --------------------------------------------------------------------------- #
# X1 + X2/X3/X4 helpers, injected where `dev` is chosen.
# --------------------------------------------------------------------------- #
DEV_OLD = '    dev = "cuda" if torch.cuda.is_available() else "cpu"'

DEV_NEW = '''    # ---- XLA port (X1): device. Single process on purpose -- see the builder's
    # docstring: xmp.spawn is what crashed the baseline TPU port, and data
    # parallelism would change the effective batch size and invalidate the paired
    # comparison this CV exists to make.
    try:
        import torch_xla.core.xla_model as _xm
        dev = "xla"
        print("[tpu] xla_device=%s (single process, no xmp.spawn)" % _xm.xla_device(),
              flush=True)
    except Exception as _e:
        _xm = None
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print("[tpu] no XLA (%s) -> dev=%s" % (repr(_e)[:100], dev), flush=True)

    # X2: the source gates autocast with enabled=dev=="cuda" but still CONSTRUCTS
    # torch.autocast("cuda", ...); on a CPU-only torch build that is not a safe
    # no-op, so the call site becomes this helper and keeps the same semantics.
    _AMP = (dev == "cuda")

    class _NullAmp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if _AMP else _NullAmp()

    # X3: XLA is lazy. optimizer_step() folds in mark_step (barrier=True), so the
    # per-step loss.item() below reads an already-materialised scalar.
    def _step(_o):
        if _xm is not None:
            _xm.optimizer_step(_o)
        else:
            _o.step()

    # X4: materialise the graph before any trained-state host transfer.
    def _mark():
        if _xm is not None:
            _xm.mark_step()
'''

# The two trained-state transfers. Anchored on the assignment, indentation
# captured, so one rule covers both call sites.
BEST_RE = re.compile(r'(?m)^(?P<i>[ ]*)best_state = \{"model": \{k: v\.detach\(\)\.cpu\(\)')
TOPK_RE = re.compile(r'(?m)^(?P<i>[ ]*)topk\.append\(\{"model": \{k: v\.detach\(\)\.cpu\(\)\.clone\(\)')
AUTOCAST_RE = re.compile(r'torch\.autocast\(dev, dtype=torch\.bfloat16, enabled=dev == "cuda"\)')
STEP_OLD = "opt.step(); sched.step(); tot += loss.item()"
STEP_NEW = "_step(opt); sched.step(); _mark(); tot += loss.item()"
PIN_OLD = "pin_memory=True"
PIN_NEW = "pin_memory=(dev == 'cuda' if isinstance(dev, str) else False)"
# X5: `--grad_ckpt` is XLA-INCOMPATIBLE. torch.utils.checkpoint's non-reentrant path does
#     device_module = getattr(torch, device_type)
# and raises `AttributeError: module 'torch' has no attribute 'xla'` on the FIRST training
# forward -- which is exactly what killed tpu-cv v1 (2026-09-19) before X1-X4 were ever
# exercised. It is a torch/torch_xla limitation, not our bug, so the flag is dropped rather
# than worked around. Watch memory: if this OOMs, lower `--bs` for the TPU arm (a TPU-only
# comparison stays valid as long as BOTH arms share the same bs).
GRAD_CKPT_OLD = "'--grad_ckpt', "
LIMIT_OLD = "    if a.limit: train_ids = train_ids[:a.limit]"
PRELUDE_HEAD = "# ==== prelude added by this project (the author script below is verbatim) ===="
FOLD_ARG = "'--fold_file', _FOLD_FILE"
# X6 (optional, --xla-bf16): the SECOND-biggest speed lever on TPU after using more than one
# core. TPU v3's bf16 throughput is far above its fp32 path, and X2 deliberately disabled
# autocast on non-CUDA devices to be safe -- so on TPU we have been paying full fp32 price.
# Wrapped in try/except and ANNOUNCED: if `torch.autocast("xla", ...)` is unsupported we want
# the log to say so, rather than silently falling back to fp32 and reading as
# "bf16 made no difference" (the HANDOFF 16.1 silent-no-op class).
AMP_OLD = ('    def _amp():\n'
           '        return torch.autocast("cuda", dtype=torch.bfloat16) if _AMP else _NullAmp()')
AMP_BF16 = ('    # X6: bf16 autocast, requested via --xla-bf16.\n'
            '    def _amp():\n'
            '        if _AMP:\n'
            '            return torch.autocast("cuda", dtype=torch.bfloat16)\n'
            '        try:\n'
            '            return torch.autocast("xla", dtype=torch.bfloat16)\n'
            '        except Exception as _e:\n'
            '            print("[tpu] autocast(\'xla\') UNSUPPORTED -> fp32 (%s)"\n'
            '                  % repr(_e)[:110], flush=True)\n'
            '            return _NullAmp()\n'
            '    print("[tpu] X6 bf16 autocast: REQUESTED (dev=%s)" % dev, flush=True)')

# X7 (--xla-sdpa-fix): make CoAtNet trainable on TPU.
#   `kernel-tpu-c224` died with
#       timm/models/maxxvit.py:230  x = F.scaled_dot_product_attention(...)
#       RuntimeError: torch_xla/csrc/helpers.cpp:619
#       stack: BuildAdd -> Promote -> PromoteShapes -> GetPromotedShape
#   i.e. XLA's SDPA lowering adds the additive attention bias the way XLA's broadcast check
#   forbids (it accepts only equal dims, or a 1). timm's CoAtNet passes its shared relative
#   position bias in as that additive mask, which is why the frontier's architecture -- and
#   ONLY it -- fails on TPU while vit_small runs 12 epochs happily.
#   timm's attention modules carry a `fused_attn` switch; turning it off routes them through
#   the explicit `q @ k.T`, bias add, softmax path, where the bias is added with an ordinary
#   broadcast. ANNOUNCED, so a silent non-application cannot read as "the fix did nothing".
SDPA_ANCHOR = "    bb = build_backbone(a.arch, pretrained=use_timm)\n"
SDPA_FIX = (
    "    # X7 (--xla-sdpa-fix): XLA's SDPA cannot broadcast the additive relative-position\n"
    "    # bias timm's CoAtNet passes (BuildAdd -> PromoteShapes), which kills coatnet on TPU.\n"
    "    # Force timm's attention modules onto the explicit matmul+bias+softmax path.\n"
    "    _n_fa = 0\n"
    "    for _m in bb.modules():\n"
    "        if hasattr(_m, \"fused_attn\"):\n"
    "            _m.fused_attn = False\n"
    "            _n_fa += 1\n"
    "    print(\"[tpu] X7 fused_attn disabled on %d module(s)\" % _n_fa, flush=True)\n"
)


def _sub1(code: str, old: str, new: str, what: str) -> str:
    """Replace exactly once, or fail the build. A silent no-op edit reads as
    'the change did nothing', the most expensive failure mode in this project
    (HANDOFF 16.1)."""
    n = code.count(old)
    if n != 1:
        raise SystemExit("%s: anchor matches %d times (need exactly 1): %r" % (what, n, old[:70]))
    return code.replace(old, new, 1)


def tpu_patch(code: str, limit: int | None, tag: str,
              xla_bf16: bool = False,
              xla_sdpa_fix: bool = False) -> tuple[str, dict]:
    applied: dict[str, int] = {}

    # --- env + deps, first thing in the file -------------------------------- #
    code = _sub1(code, PRELUDE_HEAD, DEPS_BLOCK + PRELUDE_HEAD, "deps prelude")
    applied["deps"] = 1

    # --- X1 helpers --------------------------------------------------------- #
    code = _sub1(code, DEV_OLD, DEV_NEW, "device selection")
    applied["device"] = 1

    # --- X2 autocast call sites (5: evaluate, train, OOF best, OOF topk, SWA) - #
    n = len(AUTOCAST_RE.findall(code))
    if n != 5:
        raise SystemExit("autocast: expected 5 call sites, found %d" % n)
    code = AUTOCAST_RE.sub("_amp()", code)
    applied["autocast"] = n

    # --- X3 optimizer step -------------------------------------------------- #
    code = _sub1(code, STEP_OLD, STEP_NEW, "optimizer step")
    applied["step"] = 1

    # --- X4 mark_step before the trained-state host transfers --------------- #
    for what, rx in (("best_state", BEST_RE), ("topk", TOPK_RE)):
        m = rx.search(code)
        if m is None:
            raise SystemExit("X4 %s: anchor not found" % what)
        if len(rx.findall(code)) != 1:
            raise SystemExit("X4 %s: anchor matches more than once" % what)
        code = rx.sub(lambda mm: "%s_mark()\n%s%s" % (mm.group("i"), mm.group("i"), mm.group(0).lstrip()),
                      code, count=1)
        applied["mark_" + what] = 1

    # --- hygiene: pin_memory needs CUDA ------------------------------------ #
    code = _sub1(code, PIN_OLD, PIN_NEW, "pin_memory")
    applied["pin_memory"] = 1

    # --- X5: drop gradient checkpointing (XLA-incompatible) ----------------- #
    code = _sub1(code, GRAD_CKPT_OLD, "", "drop --grad_ckpt")
    applied["drop_grad_ckpt"] = 1

    # --- X6 (optional): bf16 autocast on XLA -------------------------------- #
    if xla_bf16:
        code = _sub1(code, AMP_OLD, AMP_BF16, "xla bf16 autocast")
        applied["xla_bf16"] = 1

    # --- X7 (optional): make CoAtNet trainable on XLA ----------------------- #
    if xla_sdpa_fix:
        code = _sub1(code, SDPA_ANCHOR, SDPA_ANCHOR + SDPA_FIX, "xla sdpa fix")
        applied["xla_sdpa_fix"] = 1

    # --- probe: cap the OOF fold too (--limit only caps the train set) ------ #
    if limit is not None:
        code = _sub1(code, LIMIT_OLD,
                     LIMIT_OLD + "\n    if a.limit: oof_ids = oof_ids[:a.limit]",
                     "oof limit")
        applied["oof_limit"] = 1

    # --- tag must be unique per run (artifact names carry it) --------------- #
    if "'--tag', '" in code:
        applied["tag"] = 1
    return code, applied


def build_one(fold: int, folds: int, tag: str, res: int, epochs: int | None,
              limit: int | None, variant: str | None, push: bool,
              kernel_name: str = "kernel-tpu-cv",
              xla_bf16: bool = False,
              arch: str | None = None,
              xla_sdpa_fix: bool = False) -> pathlib.Path:
    sets = [("--res", str(res))]
    # `--arch` appended into `sets` on purpose: the prelude already sets '--arch', and argparse
    # keeps the LAST occurrence of a flag (the same trick `build_cv_kernel.py` uses for --res).
    # The CV proxy has been `vit_small_patch16_224` (solo gold ~0.83) all along, while the
    # frontier's own arms are `coatnet_rmlp_2_rw_384` (gold 0.9031..0.9167) -- and C16 says a
    # member needs >0.90 solo to be worth adding, so the ARCHITECTURE is the one mechanism
    # that could push a self-trained member across that line.
    if arch:
        sets.append(("--arch", arch))
    if epochs is not None:
        sets.append(("--epochs", str(epochs)))
    if limit is not None:
        sets.append(("--limit", str(limit)))
    # workers=0: no DataLoader worker processes at all -> nothing to fork, and the
    # memmap is read by the main process. This is the single most important
    # difference from the baseline port.
    sets.append(("--workers", "0"))

    dst = build(fold, folds, tag, kdir("kernel-raptor-train"),
                suffix=variant or "tpu", epochs=epochs if epochs is not None else 12,
                sets=sets, kernel_name=kernel_name)

    p = dst / "knee-mri-training-the-twelve-finding-model.py"
    code = p.read_text(encoding="utf-8")
    code, applied = tpu_patch(code, limit, tag, xla_bf16, xla_sdpa_fix)
    ast.parse(code)
    p.write_text(code, encoding="utf-8")

    meta = json.loads((dst / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["enable_gpu"] = False
    meta["enable_tpu"] = True
    meta["machine_shape"] = MACHINE_SHAPE
    meta["enable_internet"] = True          # timm pulls ImageNet weights at build_backbone
    meta.pop("kernel_sources", None)
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("built %s" % dst.name)
    print("  patches      %s" % ", ".join("%s=%d" % kv for kv in sorted(applied.items())))
    print("  tpu          enable_tpu=%s machine_shape=%s enable_gpu=%s"
          % (meta["enable_tpu"], meta["machine_shape"], meta["enable_gpu"]))
    print("  datasets     %s" % ", ".join(meta["dataset_sources"]))
    print("  argv tail    %s" % _argv_tail(code))
    if push:
        # ⚠️ `kaggle kernels push` EXITS 0 even when it prints "Kernel push error" --
        # observed 2026-09-20 with "Maximum batch TPU session count of 1 reached." and rc=0.
        # So the return code cannot be the check; the success SENTENCE has to be. (Same
        # silent-success class as `kernels output` truncating on an SSL error, HANDOFF 15.5.)
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(dst)], cwd=str(ROOT),
                           capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        ok = "successfully pushed" in out
        print("push %s (rc=%d)" % ("OK" if ok else "FAILED", r.returncode))
        for ln in [x for x in out.strip().splitlines() if x.strip()][-4:]:
            print("   " + ln.strip())
        if not ok:
            raise SystemExit("push did NOT succeed -- the message above is the reason")
    return dst


def _argv_tail(code: str) -> str:
    i = code.find("sys.argv = [")
    if i < 0:
        return "(argv not found)"
    j = code.find("]", i)
    return " ".join(code[i:j].split())[-260:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="1 epoch / 160 studies / real k,bs,res -- measures per-step TPU "
                         "cost and exercises train+save+OOF+gold. This is the smoke.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--res", type=int, default=288)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--kernel-name", default="kernel-tpu-cv",
                    help="kernel directory/ref to build into. Distinct names let us QUEUE "
                         "SEVERAL TPU jobs at once: pushing is free and the TPU queue is "
                         "capacity-bound, so a populated queue is what turns a capacity slot "
                         "into a finished experiment instead of an idle gap.")
    ap.add_argument("--xla-bf16", action="store_true",
                    help="X6: enable bf16 autocast on XLA (TPU v3's bf16 path is far faster "
                         "than fp32; X2 had disabled autocast on non-CUDA devices). The kernel "
                         "ANNOUNCES whether autocast('xla') was accepted, so a fallback to fp32 "
                         "cannot read as 'bf16 made no difference'.")
    ap.add_argument("--xla-sdpa-fix", action="store_true",
                    help="X7: force timm's attention modules onto the explicit matmul path so "
                         "CoAtNet can train on TPU. `c224` died inside XLA's "
                         "scaled_dot_product_attention lowering (BuildAdd -> PromoteShapes) "
                         "because the additive relative-position bias it passes cannot be "
                         "broadcast under XLA's rule. Announced at run time.")
    ap.add_argument("--arch", default=None,
                    help="timm backbone instead of the prelude's vit_small_patch16_224. Use "
                         "`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` -- the frontier's own "
                         "architecture -- to test whether OUR corpus can produce a member with "
                         "the solo AUC C16 requires (>0.90).")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    epochs, limit, tag = a.epochs, a.limit, a.tag
    if a.probe:
        epochs = 1 if epochs is None else epochs
        limit = 160 if limit is None else limit
        tag = tag or "tpuprobe"
    if tag is None:
        tag = "tpu%d" % a.fold

    build_one(a.fold, a.folds, tag, a.res, epochs, limit, None, a.push,
              kernel_name=a.kernel_name, xla_bf16=a.xla_bf16, arch=a.arch,
              xla_sdpa_fix=a.xla_sdpa_fix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
