import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
b = pathlib.Path("work/eff_log_v3.json").read_bytes()
t = None
for enc in ("utf-16","utf-16-le","utf-8"):
    try:
        x = b.decode(enc)
        if "effb0" in x.lower(): t = x; break
    except Exception: pass
i = t.find("["); j = json.loads(t[i:])
txt = "".join(e.get("data","") for e in j)
print("events=%d chars=%d" % (len(j), len(txt)))
seen=set()
for pat in [r"[^\n]*\[effb0\][^\n]*", r"[^\n]*pin (PASS|FAIL)[^\n]*", r"[^\n]*GATE[^\n]*",
            r"[^\n]*LEG FAILED[^\n]*", r"[^\n]*OutOfTime[^\n]*", r"[^\n]*fold \d done[^\n]*",
            r"[^\n]*FAILED[^\n]*", r"[^\n]*Error[^\n]*"]:
    for m in re.findall(pat, txt, re.I):
        s = m.strip()[:200]
        if s not in seen:
            seen.add(s); print("  | %s" % s)
