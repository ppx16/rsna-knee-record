"""Build the MaxSpan-v5 FINE-TUNE kernel (Phase 2's first arm with a >=0.90 starting point).

WHY THIS ONE (FACTS A167 / HANDOFF 223)
    A public notebook measured that injecting our OWN members (holdout 0.85-0.87) at 15% made
    things WORSE (0.936 vs 0.941) and concluded: **"below ~0.90 solo, a member costs more than its
    diversity buys."**  That is the number our member line was missing: our arms sit at 0.85-0.88,
    which is exactly why 24 arms scored weight 0 (A158).

    The published `raptor_ft_coatnet_v5_full_swa.pt` (dreaddevelopment/raptor-knee-maxspan) carries
    its own metadata and reads `gold_auc = 0.9214`, `arch = coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`,
    `res = 384` -- i.e. ABOVE the 0.90 bar, in the CoAtNet family our gate says is the only one that
    passes, and structurally IDENTICAL to our own `RaptorClassifier` (478 `backbone.*` tensors plus
    the same 8 head keys `clsW/clsb/norm/att`).  `indarkarhana` fine-tuned exactly this checkpoint
    and reports 0.932 standalone against its 0.924 parent.

    So Phase 2's first arm is: **warm-start from that checkpoint and fine-tune on our data.**

TWO TRAPS THIS BUILDER EXISTS TO AVOID
    1. ⚠️ SILENT NO-OP WARM START.  The trainer's `load_raptor` does
           bbst = {k[len("backbone."):]: v for k, v in st.items() if k.startswith("backbone.")}
       on the TOP LEVEL.  The published checkpoint nests everything under `"model"`, so that
       comprehension yields **zero** tensors, `load_state_dict(strict=False)` accepts the empty
       dict, and the run trains from plain timm weights while printing a success-looking line.
       We would have paid a full GPU/TPU session for nothing.  This builder REWRITES the loader to
       unwrap `student`/`model`/`state_dict` and to **raise** when it finds no backbone tensors.
    2. ⚠️ `--ckpt` must point at a file the kernel can actually see.  The converted checkpoint is
       published as a dataset and located by glob in the prelude, then passed as an absolute path.

Usage:
    python scripts/build_msft_kernel.py --check
    python scripts/build_msft_kernel.py --build
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
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kernels" / "kernel-tpu-c224"
DST = ROOT / "kernels" / "kernel-tpu-msft"
TRAINER = "knee-mri-training-the-twelve-finding-model.py"
KID = "ppxl16/rsna-knee-tpu-msft"
TITLE = "RSNA knee tpu msft"
# ⚠️ NO NEW DATASET IS NEEDED.  The first version published a converted copy as
# ppxl16/rsna-knee-msft-warmstart, and `datasets create` failed with an EMPTY response body (the
# CLI surfaced it as JSONDecodeError) -- the same empty-body signature as the CreateCodeSubmission
# 400.  But the parent checkpoint is ALREADY public: dreaddevelopment/raptor-knee-maxspan ships
# `raptor_ft_coatnet_v5_full_swa.pt`.  Mounting that and letting the patched loader unwrap `model`
# is simpler, has no publication step, and keeps the parent verifiable by hash against the original
# rather than against our own copy.
WEIGHTS_DS = "dreaddevelopment/raptor-knee-maxspan"
WARM_NAME = "raptor_ft_coatnet_v5_full_swa.pt"

# the published parent's own numbers -- the PRE-REGISTERED bar the fine-tune must beat
PARENT_GOLD = 0.9214
PARENT_BACKBONE_TENSORS = 478

OLD_LOAD = '''    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = ck["student"] if "student" in ck else ck
    bbst = {k[len("backbone."):]: v for k, v in st.items() if k.startswith("backbone.")}
    missing, unexpected = bb.load_state_dict(bbst, strict=False)'''

NEW_LOAD = '''    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # ⚠️ The published raptor checkpoints (e.g. raptor_ft_coatnet_v5_full_swa.pt) wrap the tensors
    # under "model"; the original loader looked for TOP-LEVEL "backbone." keys, found none, and
    # `load_state_dict(strict=False)` quietly accepted an EMPTY dict -- so the run trained from plain
    # timm weights while printing a success-shaped line.  Unwrap, then REFUSE to continue empty.
    for _wrap in ("student", "model", "state_dict"):
        if isinstance(ck, dict) and isinstance(ck.get(_wrap), dict):
            print(f"[raptor] unwrapped checkpoint key {_wrap!r}", flush=True)
            ck = ck[_wrap]
            break
    bbst = {k[len("backbone."):]: v for k, v in ck.items() if k.startswith("backbone.")}
    if len(bbst) == 0:
        raise SystemExit(f"!! --ckpt {ckpt_path} yielded 0 backbone tensors; refusing to silently "
                         f"train from timm weights (top-level keys: {list(ck)[:6]})")
    missing, unexpected = bb.load_state_dict(bbst, strict=False)'''

OLD_ARGS = re.compile(r"sys\.argv = \['train',.*?\]\n", re.S)
NEW_ARGS = """sys.argv = ['train',
            '--labels', _parq,
            '--arch', 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',
            '--res', '384', '--epochs', '4', '--bs', '2', '--k', '12', '--k_eval', '24',
            '--grad_ckpt', '--ckpt', _WARM,
            '--seed', '42', '--tag', 'msft0', '--folds', '5', '--fold', '0',
            '--fold_file', _FOLD_FILE, '--workers', '0']
"""

WARM_PROBE = """_warm_c = glob.glob('/kaggle/input/**/%s', recursive=True)
if not _warm_c:
    raise SystemExit('warm-start checkpoint %s not found under /kaggle/input (dataset mounted?)')
_WARM = _warm_c[0]
print('[prelude] warm start -> %%s' %% _WARM, flush=True)
""" % (WARM_NAME, WARM_NAME)


def build(device="tpu"):
    if not (SRC / TRAINER).is_file():
        raise SystemExit("!! source kernel missing: %s" % (SRC / TRAINER))
    text = (SRC / TRAINER).read_text(encoding="utf-8", errors="replace")
    if OLD_LOAD not in text:
        raise SystemExit("!! the loader anchor did not match -- refusing to write (upstream changed?)")
    text = text.replace(OLD_LOAD, NEW_LOAD, 1)
    n = len(OLD_ARGS.findall(text))
    if n != 1:
        raise SystemExit("!! argv prelude matched %d times (need exactly 1)" % n)
    text = OLD_ARGS.sub(NEW_ARGS.replace("\\", "\\\\"), text, count=1)
    # the warm-start probe goes immediately before the argv assignment
    text = text.replace("sys.argv = ['train',", WARM_PROBE + "sys.argv = ['train',", 1)

    # ---- assert the RESULT, not the edit ----
    assert "refusing to silently" in text, "the silent-no-op guard did not take"
    assert "'--ckpt', _WARM" in text, "--ckpt is not wired to the probe"
    assert "_warm_c = glob.glob" in text, "the warm-start probe is missing"
    assert "'--res', '384'" in text and "'--epochs', '4'" in text, "argv is not the fine-tune config"
    assert "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k" in text, "arch changed"

    DST.mkdir(parents=True, exist_ok=True)
    (DST / TRAINER).write_text(text, encoding="utf-8")
    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"], meta["title"] = KID, TITLE
    # ⚠️ DEVICE.  This was built for TPU, but Kaggle allows only ONE concurrent TPU session and our
    # own `kernel-tpu-3dsmoke` was holding it (the push came back "Maximum batch TPU session count of
    # 1 reached", with kernelId 0 -- i.e. nothing was created).  A QUEUED kernel cannot be cancelled
    # from the CLI, so the choice was to wait on our own low-priority probe or move.  The user asked
    # for score first, and both GPU slots are free, so --device gpu runs it now.
    # Memory: the trainer's own note says CoAtNet@384 at bs 8 + grad_ckpt is 23.7 GB; at bs 2 +
    # grad_ckpt it should fit a 16 GB T4.
    meta["enable_tpu"], meta["enable_gpu"] = (device == "tpu"), (device == "gpu")
    meta["enable_internet"] = True
    # ⚠️ `machine_shape` OVERRIDES THE FLAGS.  The source kernel is a TPU kernel and carries
    # "machine_shape": "Tpu1VmV38".  Setting enable_tpu=False / enable_gpu=True was NOT enough --
    # the push still came back "Maximum batch TPU session count of 1 reached" with kernelId 0,
    # because Kaggle read the machine_shape and counted a TPU session anyway.  Drop it for GPU.
    if device == "gpu":
        meta.pop("machine_shape", None)
    else:
        meta["machine_shape"] = "Tpu1VmV38"
    ds = [d for d in (meta.get("dataset_sources") or []) if d]
    if WEIGHTS_DS not in ds:
        ds.append(WEIGHTS_DS)
    meta["dataset_sources"] = ds
    meta.pop("id_no", None)
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
    print("built %s" % DST)
    print("   arch  : coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k @384, 4 epochs, bs 2, grad_ckpt")
    print("   ckpt  : %s (from dataset %s)" % (WARM_NAME, WEIGHTS_DS))
    print("   argv  : %s" % NEW_ARGS.strip().splitlines()[1].strip())
    print("\n   PRE-REGISTERED BAR: the fine-tune must beat the parent's gold_auc %.4f" % PARENT_GOLD)
    print("   (the parent also has %d backbone tensors; the loader now refuses anything empty)"
          % PARENT_BACKBONE_TENSORS)
    print("   ⚠️ memory: the trainer's own note says CoAtNet@384 bs 8 + grad_ckpt = 23.7 GB, so bs 2")
    print("      + grad_ckpt is the bet for a 16 GB TPU core. An OOM here is cheap information.")
    return 0


def check():
    t = (DST / TRAINER).read_text(encoding="utf-8", errors="replace")
    for probe, why in (("refusing to silently", "silent no-op guard"),
                       ("_warm_c = glob.glob", "warm-start probe"),
                       ("'--ckpt', _WARM", "--ckpt wiring"),
                       ("'--epochs', '4'", "fine-tune epoch count"),
                       ("unwrapped checkpoint key", "wrapper unwrapping")):
        if probe not in t:
            raise SystemExit("!! %s missing from the built payload" % why)
    m = json.loads((DST / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert m["enable_tpu"] or m["enable_gpu"], "must run on an accelerator"
    assert WEIGHTS_DS in m["dataset_sources"], "warm-start dataset not mounted"
    print("CHECK OK: %s carries the guard, the probe, the accelerator flags (%s) and the weights "
          "dataset" % (DST.name, "tpu" if m["enable_tpu"] else "gpu"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="tpu", choices=["tpu", "gpu"])
    a = ap.parse_args()
    if a.check:
        return check()
    return build(a.device)


if __name__ == "__main__":
    sys.exit(main())
