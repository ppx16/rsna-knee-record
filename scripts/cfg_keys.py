import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.loads(pathlib.Path("kernels/kernel-0943cfm/rsna-knee-restructured-version-3.ipynb").read_text(encoding="utf-8", errors="replace"))
cfg = "".join(nb["cells"][2].get("source", []))
print("==== RUN config keys (cell 2) ====")
# lines like  "name": value,
keys = []
for m in re.finditer(r'^\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*([^,\n]+)', cfg, re.M):
    k, v = m.group(1), m.group(2).strip()
    if k in ("coatnet_w",): 
        keys.append((k, v)); continue
    print("   %-28s %s" % (k, v[:70]))
    keys.append((k, v))
print("\ntotal top-level keys: %d" % len(keys))
# nested coatnet_w
m = re.search(r'"coatnet_w"\s*:\s*\{([^}]*)\}', cfg, re.S)
if m:
    print("\n   coatnet_w (per-label): %s" % " ".join(m.group(1).split())[:220])
