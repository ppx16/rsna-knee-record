import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.loads(pathlib.Path("kernels/kernel-0943cfm2eff/rsna-knee-restructured-version-3.ipynb").read_text(encoding="utf-8", errors="replace"))
for i, c in enumerate(nb["cells"]):
    t = "".join(c.get("source", []))
    for pat in [r"[^\n]*_EFF_SLOT_NAMES\s*=[^\n]*", r"[^\n]*_EFF_N_SLOT\s*=[^\n]*",
                r"[^\n]*RULES_LEGACY\s*=[^\n]*", r"[^\n]*_EFF_META_SLOTS\s*=[^\n]*",
                r"[^\n]*_EFF_SLOT_IDX\s*=[^\n]*"]:
        for m in re.finditer(pat, t):
            print("[%2d] %s" % (i, m.group(0).strip()[:170]))
print("\n---- re-pin block (where RULES etc are set) ----")
for i, c in enumerate(nb["cells"]):
    t = "".join(c.get("source", []))
    if "RULES = " in t or "RULES=" in t:
        for m in re.finditer(r"[^\n]*(RULES|CACHE_SLICES|SLICE_BAND|CROP_MM|IMG)\s*=[^\n]*", t):
            s = m.group(0).strip()
            if "==" not in s: print("[%2d] %s" % (i, s[:150]))
