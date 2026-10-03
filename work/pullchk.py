import json, sys, re, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ps = list(pathlib.Path("work/pulled_v3").glob("*.ipynb"))
if not ps: print("no notebook pulled"); raise SystemExit
nb = json.loads(ps[0].read_text(encoding="utf-8", errors="replace"))
t2 = "".join(nb["cells"][2].get("source", []))
m = re.search(r'"effb0mil_w"[^\n]*', t2)
print("PUSHED v3 default:", m.group(0).strip() if m else "NOT FOUND")
t = " ".join("".join(c.get("source",[])) for c in nb["cells"])
print("has per-term pin (_EFF_PIN_TESTS):", "_EFF_PIN_TESTS" in t)
print("cells:", len(nb["cells"]))
