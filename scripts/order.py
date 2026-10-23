import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.loads(pathlib.Path("kernels/kernel-0943cfm2eff/rsna-knee-restructured-version-3.ipynb").read_text(encoding="utf-8", errors="replace"))
lines = "".join(nb["cells"][50].get("source", [])).split("\n")
keys = ["_eff_check_budget(", "build_cache", "_gate_t0", "for _f in", "to_input", "_EFF_GATE_BUDGET_S", "rsna_deadline"]
for k, l in enumerate(lines):
    for key in keys:
        if key in l:
            print("%4d| %s" % (k, l.strip()[:150])); break
