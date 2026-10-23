import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
raw = pathlib.Path("work/eff_log.json").read_text(encoding="utf-8", errors="replace")
txt = raw
try:
    j = json.loads(raw)
    txt = "".join(e.get("data","") for e in j) if isinstance(j, list) else json.dumps(j)
except Exception:
    pass
print("log chars:", len(txt))
for pat in [r".*effb0mil.*", r".*effb0.*", r".*GATE.*", r".*ACCEPT.*", r".*ABANDON.*", r".*NOT AN ADJUDICATOR.*"]:
    hits = re.findall(pat, txt, re.I)
    print("\n=== %s  (%d) ===" % (pat, len(hits)))
    for h in hits[:12]:
        print("   %s" % h.strip()[:170])
