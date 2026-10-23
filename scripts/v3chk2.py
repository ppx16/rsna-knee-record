import json, sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.load(open(r"kernels/kernel-0943cfm2eff/rsna-knee-restructured-version-3.ipynb", encoding="utf-8"))
t2 = "".join(nb["cells"][2].get("source", []))
for m in re.finditer(r"[^\n]*effb0mil[^\n]*", t2):
    print("  %s" % m.group(0).strip())
