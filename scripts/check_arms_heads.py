"""Do the `raptor-knee-arms` checkpoints have TRANSFERABLE heads?

msft2's whole success rested on this: its warm start came from the same trainer, so the head
loaded 245/245 strict=True, and 4 epochs + SWA took it to 0.9069 -- our best member and the only
one that ever moved the rank. The ConvNeXt attempt failed precisely because its head was a
FOREIGN architecture (11 tensors, none matching), forcing a from-scratch head and landing at
0.8493.

`dreaddevelopment/raptor-knee-arms` ships three DIFFERENT backbones under the same naming scheme:
    raptor_ft_cnv2b336.pt    ConvNeXtV2-Base @336
    raptor_ft_coatnet384.pt  CoAtNet @384
    raptor_ft_effv2l480.pt   EfficientNetV2-L @480
If those are RaptorClassifier checkpoints, each is a NEW backbone family with a TRANSFERABLE head
-- exactly the msft2 recipe applied to a decorrelated architecture.

This compares the downloaded checkpoint's key set against `RaptorClassifier`'s parameter names,
read straight out of our own notebook, so the answer does not depend on anyone's claim.
"""

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
import json
import pathlib
import re

import torch

WORK = pathlib.Path(r"D:\kaggle\rsna-knee\work\arms_check")
NB = pathlib.Path(r"D:\kaggle\rsna-knee\kernels\kernel-0943cfmsft2m\rsna-knee-restructured-version-3.ipynb")

# --- 1. the parameter names RaptorClassifier would produce, from the notebook's own source ---
nb = json.loads(NB.read_text(encoding="utf-8"))
src = "\n".join("".join(c.get("source", "")) if isinstance(c.get("source"), list)
                else (c.get("source") or "") for c in nb["cells"])
m = re.search(r"class RaptorClassifier\(nn\.Module\):(.*?)(?=\nclass |\ndef |\Z)", src, re.S)
cls = m.group(1) if m else ""
print("RaptorClassifier source found:", bool(cls))
print(cls[:1200] if cls else "!! not found")
print()

# --- 2. what the arms checkpoints actually carry ---
for name in sorted(WORK.glob("*.pt")):
    try:
        o = torch.load(name, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"{name.name}: load FAILED {type(e).__name__}: {str(e)[:120]}")
        continue
    sd = o.get("state_dict") if isinstance(o, dict) else None
    if sd is None and hasattr(o, "state_dict"):
        sd = o.state_dict()
    print("=" * 78)
    print(f"{name.name}   {name.stat().st_size/1048576:.1f} MB")
    if isinstance(o, dict):
        for k in ("cfg", "fold", "tag", "arch", "backbone"):
            if k in o:
                v = o[k]
                print(f"   {k} = {str(v)[:200]}")
    if sd is None:
        print("   no state_dict found; top-level keys:", list(o)[:15] if isinstance(o, dict) else type(o))
        continue
    keys = list(sd.keys())
    print(f"   {len(keys)} tensors")
    print("   first 6:", keys[:6])
    print("   last 4 :", keys[-4:])
    # head-ish tensors: anything not obviously backbone
    head = [k for k in keys if not k.startswith("enc.")]
    print(f"   non-backbone (head/pool) tensors: {len(head)}")
    print("     ", head[:14])
