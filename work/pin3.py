import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.loads(pathlib.Path("kernels/kernel-0943cfm2eff/rsna-knee-restructured-version-3.ipynb").read_text(encoding="utf-8", errors="replace"))
for i, c in enumerate(nb["cells"]):
    t = "".join(c.get("source", []))
    if "_EFF_PIN" in t or "frozen-corpus pin" in t:
        lines = t.split("\n")
        for k, l in enumerate(lines):
            if "_EFF_PIN" in l or "frozen-corpus pin" in l or "_EFF_SLOT_NAMES" in l:
                print("[%2d] %s" % (i, l[:165]))
