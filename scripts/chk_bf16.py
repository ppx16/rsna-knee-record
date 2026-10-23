import json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.load(open(r"kernels\kernel-0943cfm\rsna-knee-restructured-version-3.ipynb", encoding="utf-8"))
pats = {
 "autocast":      r".*autocast.*",
 "bfloat16/bf16": r".*(bfloat16|bfloat|bf16).*",
 "AMP_PREF":      r".*AMP_PREF.*",
 "raptor rank":   r".*(rankpct|rank\(method|_raptor\.csv|raptor_k_eval).*",
}
for name, p in pats.items():
    print("==== %s ====" % name)
    seen = 0
    for i, c in enumerate(nb["cells"]):
        t = "".join(c.get("source", []))
        for m in re.finditer(p, t, re.I):
            line = m.group(0).strip()
            if len(line) > 6:
                print("  [%3d] %s" % (i, line[:130])); seen += 1
                if seen > 12: break
        if seen > 12: break
