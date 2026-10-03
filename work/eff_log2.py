import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
raw = pathlib.Path("work/eff_log.json").read_text(encoding="utf-8", errors="replace")
txt = raw
try:
    j = json.loads(raw); txt = "".join(e.get("data","") for e in j) if isinstance(j, list) else json.dumps(j)
except Exception as e:
    print("json parse failed:", str(e)[:80])
print("chars:", len(txt))
print("\n---- HEAD ----"); print(txt[:700])
print("\n---- TAIL ----"); print(txt[-900:])
for k in ["raptor","dinov2","coat-arm","COMPLETE","rsna_event","pipeline_stage","Cell","cell"]:
    print("  %-14s %d" % (k, len(re.findall(k, txt, re.I))))
