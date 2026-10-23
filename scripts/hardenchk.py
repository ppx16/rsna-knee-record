import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb=json.load(open(r"kernels/kernel-0943cfm2eff/rsna-knee-restructured-version-3.ipynb",encoding="utf-8"))
t2="".join(nb["cells"][2].get("source",[])); t50="".join(nb["cells"][50].get("source",[]))
import re
print("cell2 budget:", re.search(r'"effb0mil_budget_s"[^\n]*', t2).group(0).strip())
print("cell50 pre-flight guard:", "test cache build (pre-flight)" in t50)
print("cell50 build timing    :", "_eff_build_t0" in t50)
