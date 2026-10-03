import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
b = pathlib.Path("work/eff_log.json").read_bytes()
for enc in ("utf-16","utf-16-le","utf-8"):
    try:
        t = b.decode(enc)
        if "effb0" in t.lower() or "raptor" in t.lower(): break
    except Exception: pass
print("decoded with", enc, "chars:", len(t))
i = t.find("[")
j = json.loads(t[i:]) if i >= 0 else []
txt = "".join(e.get("data","") for e in j)
print("events:", len(j), "text chars:", len(txt))
for k in ["effb0","GATE","ACCEPT","ABANDON","raptor","coat-arm","COMPLETE"]:
    print("  %-10s %d" % (k, len(re.findall(k, txt, re.I))))
print("\n---- effb0 / gate lines ----")
for m in re.findall(r"[^\n]*(?:effb0|gate|accept|abandon|adjudicator)[^\n]*", txt, re.I)[:25]:
    print("   %s" % m.strip()[:180])
