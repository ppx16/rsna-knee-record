import csv, json, sys, pathlib, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = pathlib.Path("work/newds")
rows = list(csv.DictReader((d/"competition_intelligence_panel.csv").open(encoding="utf-8", errors="replace")))
mine = [r for r in rows if "rsna-knee" in (r.get("competition_slug") or "")]
last = mine[-1]
print("==== metric_details_json (last scrape) ====")
print((last.get("metric_details_json") or "")[:1200])
print("\n==== any tie-related text anywhere in the panel ====")
seen=set()
for r in mine:
    for k,v in r.items():
        if v and re.search(r"tie|break|earliest|submission time", str(v), re.I):
            s = "%s = %s" % (k, str(v)[:150])
            if s not in seen: seen.add(s); print("   " + s)
