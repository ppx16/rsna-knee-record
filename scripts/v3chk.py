import json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.load(open(r"kernels/kernel-0943cfm2eff/rsna-knee-restructured-version-3.ipynb", encoding="utf-8"))
t2 = "".join(nb["cells"][2].get("source", []))
m = re.search(r'"effb0mil_w"\s*:\s*float\(os\.environ\.get\("RSNA_EFFB0MIL_W"\s*,\s*"([0-9.]+)"\)\)', t2)
print("v3 default effb0mil_w =", m.group(1) if m else "NOT FOUND")
ok = bad = 0
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    try: compile("".join(c.get("source", [])), "<c>", "exec"); ok += 1
    except SyntaxError as e: bad += 1; print("SyntaxError:", e)
print("cells=%d  code_ok=%d  bad=%d" % (len(nb["cells"]), ok, bad))
