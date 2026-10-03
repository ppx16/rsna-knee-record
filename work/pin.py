import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.loads(pathlib.Path("kernels/kernel-0943cfm2eff/rsna-knee-restructured-version-3.ipynb").read_text(encoding="utf-8", errors="replace"))
for i, c in enumerate(nb["cells"]):
    t = "".join(c.get("source", []))
    if "frozen-corpus pin failed" in t:
        for m in re.finditer(r"[^\n]*frozen-corpus pin failed[^\n]*", t):
            print("cell %d: %s" % (i, m.group(0).strip()[:300]))
        # show the few lines before it
        lines = t.split("\n")
        for k, l in enumerate(lines):
            if "frozen-corpus pin failed" in l:
                print("---- context ----")
                for x in lines[max(0,k-16):k+4]:
                    print("   %s" % x[:170])
                break
