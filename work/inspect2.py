import json, sys, pathlib, csv
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = pathlib.Path("work/newds")
for n in ["ft_cache.json","ft_cache_index.json"]:
    p = d / n
    if not p.exists(): print("== %s MISSING" % n); continue
    print("== %s  (%d bytes)" % (n, p.stat().st_size))
    try:
        o = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        if isinstance(o, dict):
            print("   dict keys:", list(o)[:22])
            for k in list(o)[:8]:
                v = o[k]
                print("     %-22s %s" % (k, str(v)[:110]))
        elif isinstance(o, list):
            print("   list len", len(o), "first:", str(o[0])[:200] if o else "")
    except Exception as e:
        print("   not json:", str(e)[:80]); print("   head:", p.read_text(encoding="utf-8",errors="replace")[:250])
for n in ["manifest.csv","manifest_processed.csv","failed_studies.csv"]:
    p = d / n
    if not p.exists(): continue
    rows = list(csv.reader(p.open(encoding="utf-8", errors="replace")))
    print("== %s  %d rows" % (n, len(rows)))
    print("   header:", " | ".join(x[:34] for x in rows[0][:12]))
    if len(rows) > 1: print("   row1  :", " | ".join(x[:26] for x in rows[1][:12]))
