import json, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.loads(pathlib.Path("kernels/kernel-0943cfm2eff/rsna-knee-restructured-version-3.ipynb").read_text(encoding="utf-8", errors="replace"))
lines = "".join(nb["cells"][50].get("source", [])).split("\n")
s = next(k for k,l in enumerate(lines) if "_EFF_SLOT_NAMES" in l)
e = next(k for k,l in enumerate(lines) if "pinned: rules=" in l)
for k in range(s-30, e+3):
    if 0 <= k < len(lines): print("%4d| %s" % (k, lines[k]))
