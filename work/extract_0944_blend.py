"""Extract the exact blend mechanism and the checkpoint dataset from the 0.944 notebook.

The claim to verify: it is the community stack UNCHANGED plus one independent reader, combined by a
single global rank blend, with the reader's weight as the only free parameter, AND a safe fallback that
keeps the stack's submission if the reader fails. If that is the mechanism, it is adoptable on top of
OUR arm without touching anything else -- which would make it the first genuinely new lever since msft2.
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

NB = pathlib.Path("work/newpub_1006/rsna-knee-stack-2-5d-convnext-mil-lb-0-944/"
                  "rsna-knee-stack-2-5d-convnext-mil-lb-0-944.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8", errors="replace"))
cells = nb["cells"]


def src(c):
    s = c.get("source")
    return "".join(s) if isinstance(s, list) else (s or "")


code = [src(c) for c in cells if c.get("cell_type") == "code"]
allcode = "\n\n# ---- next cell ----\n\n".join(code)
print("code cells: %d, total chars: %d" % (len(code), len(allcode)))
print()

print("=== cells that mention the blend / OWN_W / fallback ===")
for i, c in enumerate(code):
    if re.search(r"OWN_W|fallback|blend|0\.30|0\.70", c):
        print("-" * 90)
        print("CELL %d (%d chars)" % (i, len(c)))
        print(c[:2600])
        print()

print("=== checkpoint dataset references ===")
for m in sorted(set(re.findall(r"/kaggle/input/datasets/[A-Za-z0-9_\-]+/[A-Za-z0-9_\-]+", allcode))):
    print("   ", m)
print()
print("=== kernel-metadata dataset/model sources ===")
km = pathlib.Path("work/newpub_1006/rsna-knee-stack-2-5d-convnext-mil-lb-0-944/"
                  "kernel-metadata.json")
if km.is_file():
    md = json.loads(km.read_text(encoding="utf-8"))
    for k in ("dataset_sources", "model_sources", "competition_sources",
              "enable_gpu", "enable_tpu", "enable_internet", "machine_shape"):
        print("   %-20s %s" % (k + ":", md.get(k)))
