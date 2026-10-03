import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.loads(pathlib.Path("kernels/kernel-0943cfm2eff/rsna-knee-restructured-version-3.ipynb").read_text(encoding="utf-8", errors="replace"))
lines = "".join(nb["cells"][50].get("source", [])).split("\n")
for k, l in enumerate(lines):
    if "_EFF_PIN" in l or "frozen-corpus pin" in l or "pin failed" in l:
        print("%4d| %s" % (k, l))
print("\n---- full pin region ----")
try:
    s = next(k for k,l in enumerate(lines) if "_EFF_PIN" in l)
    for k in range(max(0,s-6), min(len(lines), s+30)): print("%4d| %s" % (k, lines[k]))
except StopIteration:
    print("no _EFF_PIN found")
