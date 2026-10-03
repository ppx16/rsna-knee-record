import json, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.loads(pathlib.Path("kernels/kernel-0943cfm2eff/rsna-knee-restructured-version-3.ipynb").read_text(encoding="utf-8", errors="replace"))
t = "".join(nb["cells"][50].get("source", []))
lines = t.split("\n")
# print from the slot-names block through the pin
start = next(k for k,l in enumerate(lines) if "_EFF_SLOT_NAMES" in l)
end   = next(k for k,l in enumerate(lines) if 'pinned: rules=' in l) + 2
for k in range(start, min(end, start+80)):
    print("%4d| %s" % (k, lines[k]))
