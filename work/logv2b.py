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
for m in re.finditer(r"[^\n]*frozen-corpus pin failed[^\n]*", txt):
    print("FULL: %s" % m.group(0).strip())
print()
# the assert site itself
for m in re.finditer(r"[^\n]*(_EFF_PIN|pin|assert)[^\n]*", txt):
    s = m.group(0).strip()
    if "pin" in s.lower() and len(s) > 20: print("  | %s" % s[:240])
