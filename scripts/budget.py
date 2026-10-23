import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.loads(pathlib.Path("kernels/kernel-0943cfm2eff/rsna-knee-restructured-version-3.ipynb").read_text(encoding="utf-8", errors="replace"))
t = "".join(nb["cells"][50].get("source", []))
print("==== budgets as coded ====")
for pat in [r"[^\n]*_EFF_BUDGET_S\s*=[^\n]*", r"[^\n]*_EFF_RESERVE_S\s*=[^\n]*",
            r"[^\n]*_EFF_GATE_BUDGET_S\s*=[^\n]*", r"[^\n]*effb0mil_budget_s[^\n]*",
            r"[^\n]*_eff_check_budget\([^\n]*"]:
    for m in re.finditer(pat, t):
        print("   %s" % m.group(0).strip()[:150])
print("\n==== where the budget is checked ====")
for m in re.finditer(r"[^\n]*_eff_check_budget\([^\n]*", t):
    print("   %s" % m.group(0).strip()[:140])
print("\n==== inference / cache-build call sites ====")
for k in ["build_cache", "to_input", "forward", "no_grad", "inference_mode"]:
    print("   %-16s %d" % (k, t.count(k)))
