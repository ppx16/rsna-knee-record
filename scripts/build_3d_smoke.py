"""Build/push the 3D volumetric smoke kernel (Phase 2's first, cheapest step).

The smoke answers exactly one question -- does a Conv3d model fit and compile on Kaggle/TPU --
because every family in the stack is 2D and the one untried inductive bias is a real volume.
It is deliberately NOT an arm: no checkpoint, no accuracy claim.

    python scripts/build_3d_smoke.py --check     # static preflight (no quota)
    python scripts/build_3d_smoke.py --push      # create/version the kernel on Kaggle
    python scripts/build_3d_smoke.py --status
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
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
KDIR = ROOT / "kernels" / "kernel-tpu-3dsmoke"
CODE = KDIR / "kernel_main.py"
KID = "ppxl16/rsna-knee-tpu-3dsmoke"
TITLE = "RSNA knee tpu 3dsmoke"

# The corpus is the volume source: (4407, 44, 336, 336) uint8, already mounted by other kernels.
# NOTE `barun2104/rsna-knee-mri-processed-3d-volumes` was checked and is 20 files / 0.01 GB -- a
# sample, not a dataset -- so it is deliberately NOT a source here.
DATASETS = ["dreaddevelopment/knee-raptor-corpus",
            "flight0234/rsna-knee-hybrid-report-labels"]

# probe -> why it matters.  A smoke that cannot find its corpus or its device is not a smoke.
PROBES = [
    ("torch_xla.core.xla_model", "must ask for the XLA device explicitly"),
    ("xm.optimizer_step", "XLA needs its own step call; a plain opt.step() is a silent no-op"),
    ("nn.Conv3d", "the whole point: a 3D conv, not a 2D conv over stacked slices"),
    ("AdaptiveAvgPool3d", "global 3D pooling; some ops are unsupported on XLA"),
    ("mmap_mode='r'", "the corpus is 14.8 GB -- it must be memmapped, never loaded"),
    ("all_vols.npy", "the actual volume source"),
    ("find_labels", "labels are optional here; random targets are an accepted fallback"),
    ("VERDICT: PASS", "the kernel must state its own verdict so the log is greppable"),
]


def check():
    if not CODE.is_file():
        raise SystemExit("!! %s is missing" % CODE)
    py_compile.compile(str(CODE), doraise=True)          # syntax only; torch_xla is not local
    text = CODE.read_text(encoding="utf-8")
    missing = [why for probe, why in PROBES if probe not in text]
    if missing:
        raise SystemExit("!! payload is missing: %s" % "; ".join(missing))
    print("CHECK OK: %s compiles and contains all %d probes" % (CODE.name, len(PROBES)))
    print("   corpus source : %s" % DATASETS[0])
    print("   NOTE: torch_xla is NOT importable on this laptop, so the only real test of the")
    print("         mechanism is the run itself. That is why the smoke is cheap and self-verdicting.")
    return 0


def write_meta():
    meta = {"id": KID, "title": TITLE, "code_file": CODE.name, "language": "python",
            "kernel_type": "script", "is_private": True,
            "enable_gpu": False, "enable_tpu": True, "enable_internet": True,
            "keywords": [], "dataset_sources": DATASETS,
            "kernel_sources": [], "competition_sources": [], "model_sources": []}
    (KDIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote %s" % (KDIR / "kernel-metadata.json"))
    return meta


def push():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    write_meta()
    r = api.kernels_push(str(KDIR))
    print("pushed:", r)
    # a TPU refusal is explicit (the concurrency cap is 1) -- report it rather than retry blindly
    try:
        st = api.kernels_status(KID)
        s = getattr(st, "status", None)
        print("status:", getattr(s, "name", s))
    except Exception as e:
        print("status: unknown (%s)" % type(e).__name__)
    return 0


def status():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    st = api.kernels_status(KID)
    s = getattr(st, "status", None)
    print("status:", getattr(s, "name", s))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    check()
    if a.push:
        return push()
    if a.status:
        return status()
    write_meta()
    return 0


if __name__ == "__main__":
    sys.exit(main())
