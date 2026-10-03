import csv, sys, pathlib, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = pathlib.Path("work/newds")
rows = list(csv.DictReader((d/"manifest_processed.csv").open(encoding="utf-8", errors="replace")))
print("rows:", len(rows), " studies:", len({r["StudyInstanceUID"] for r in rows}))
print("estado counts:", collections.Counter(r["estado"] for r in rows).most_common(6))
c = collections.Counter(r["descripcion"] for r in rows)
print("\ntop 24 series descriptions:")
for k, v in c.most_common(24): print("   %-40s %5d" % (k[:40], v))
