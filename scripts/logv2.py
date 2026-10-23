import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
b = pathlib.Path("work/eff_log_v2.json").read_bytes()
t = None
for enc in ("utf-16","utf-16-le","utf-8"):
    try:
        x = b.decode(enc)
        if "effb0" in x.lower(): t = x; break
    except Exception: pass
i = t.find("["); j = json.loads(t[i:])
txt = "".join(e.get("data","") for e in j)
print("events=%d chars=%d" % (len(j), len(txt)))
pats = [r"[^\n]*CONVENTION GATE[^\n]*", r"[^\n]*GATE[^\n]*", r"[^\n]*effb0\][^\n]*",
        r"[^\n]*LEG FAILED[^\n]*", r"[^\n]*Error[^\n]*", r"[^\n]*Exception[^\n]*"]
seen = set()
for p in pats:
    for m in re.findall(p, txt, re.I):
        s = m.strip()[:200]
        if s not in seen:
            seen.add(s); print("  | %s" % s)
