"""DECISIVE PROBE: does the TAAF vLLM wheelhouse import on cc 7.5 (Tesla T4)?

Context (all measured, from the ARC ledger):
  * FINDINGS F43.2: a probe kernel requesting `machine_shape: NvidiaRtxPro6000` pushed, ran and
    COMPLETED -- and was given **2x Tesla T4 (cc 7.5)**. `machine_shape` is accepted and then
    SILENTLY IGNORED on this account. So the FP8 duck-harness variant CANNOT run: a 27B FP8
    model is ~27 GB against 14.6 GB per device, and cc 7.5 does not support FP8 at all
    (it needs cc >= 8.9). The wheelhouse is even named "h100-wheelhouse".
  * FINDINGS F43.4: the tractable version is the SAME harness + SAME TAAF bundle with the
    largest quantization that actually executes on 2x T4 (GPTQ/AutoRound W4A16 27B, or an
    8B coder at 4-bit).
  * F43.4's risk #1, to be tested FIRST: "`driessmit1/arc3-vllm-h100-wheelhouse-v3` may not
    import on cc 7.5 at all".

This kernel answers exactly that, and nothing else. It is cheap on purpose: it must be the
first thing we learn, not the last.

PRE-REGISTERED PREDICTION, written before the run (so the result cannot be rationalised after):
  P1. The machine reports Tesla T4 at cc 7.5, NOT an RTX Pro 6000 -- because machine_shape is
      silently ignored (F43.2).
  P2. `pip install --no-index` from the wheelhouse either finds no compatible wheel, or installs
      and then fails to IMPORT (h100-targeted build / missing cc>=8 features / CUDA version
      mismatch). I expect the import to FAIL more often than not.
  P3. If the import SUCCEEDS, the remaining blocker is only the quantisation choice, and the next
      step is a W4A16 (GPTQ/AutoRound) model with vLLM pinned to float16 (T4 has no bfloat16).

Usage: push this directory as a Kaggle *script* kernel with the wheelhouse dataset attached.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

print("=" * 78)
print("STEP 1 -- what hardware did we actually get?")
print("=" * 78)
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=120)
    print(out.stdout.strip() or out.stderr.strip())
except Exception as exc:  # noqa: BLE001
    print("nvidia-smi failed:", exc)
print("KAGGLE_GPU_TYPE env:", os.environ.get("KAGGLE_GPU_TYPE"))
print("machine_shape is NOT trustable here -- F43.2 measured it being silently ignored.")

print()
print("=" * 78)
print("STEP 2 -- what does the wheelhouse actually contain?")
print("=" * 78)
wheels = []
for pat in ("/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/"):
    for d in sorted(glob.glob(pat)):
        found = glob.glob(os.path.join(d, "*.whl"))
        if found:
            wheels.append((d.rstrip("/"), found))
if not wheels:
    print("!! no .whl files found anywhere under /kaggle/input -- list the tree:")
    for pat in ("/kaggle/input/*", "/kaggle/input/*/*"):
        for p in sorted(glob.glob(pat))[:60]:
            print("   ", p)
else:
    for d, found in wheels:
        print(f"  dir {d}: {len(found)} wheel(s)")
        names = sorted(os.path.basename(w) for w in found)
        for n in names[:40]:
            print("     ", n)
        if len(names) > 40:
            print(f"      ... and {len(names) - 40} more")
        # vllm is the one that matters
        vllm = [n for n in names if n.lower().startswith("vllm")]
        print(f"  -> vllm wheels: {vllm or 'NONE'}")

print()
print("=" * 78)
print("STEP 3 -- can it be installed, and does it IMPORT on cc 7.5?")
print("=" * 78)
target = "/kaggle/working/_wh_env"
link_dirs = [d for d, _ in wheels]
cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
       "--target", target]
for d in link_dirs:
    cmd += ["--find-links", d]
cmd += ["vllm"]
print("install cmd:", " ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
print("install rc:", r.returncode)
print((r.stdout or "")[-1500:])
print((r.stderr or "")[-2500:])

os.environ["PYTHONPATH"] = target + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

print()
for mod in ("torch", "vllm", "transformers"):
    try:
        m = __import__(mod)
        print(f"import {mod:14s} OK   version {getattr(m, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"import {mod:14s} FAILED  {type(exc).__name__}: {str(exc)[:220]}")

print()
print("=" * 78)
print("STEP 4 -- if vllm imported, does it see the GPU?")
print("=" * 78)
try:
    import torch
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            print(f"  dev{i} {torch.cuda.get_device_name(i)} cc={cap[0]}.{cap[1]} "
                  f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    from vllm import LLM  # noqa: F401
    print("from vllm import LLM: OK")
except Exception as exc:  # noqa: BLE001
    print(f"vLLM GPU path FAILED  {type(exc).__name__}: {str(exc)[:300]}")

print()
print("=" * 78)
print("VERDICT OF THE PROBE (read the three import lines in STEP 3 and the block in STEP 4)")
print("=" * 78)
print("  vllm import OK + GPU visible  -> the blocker is ONLY the quantisation; next is a")
print("                                   W4A16 (GPTQ/AutoRound) 27B with float16 (T4 has no bf16).")
print("  vllm import FAILED            -> the h100 wheelhouse is unusable here; the Duck stack")
print("                                   needs a different serving route (e.g. llama.cpp with a")
print("                                   Q4_K_XL GGUF, which the TAAF source supports via")
print("                                   configs/inference.local.llama.json).")
