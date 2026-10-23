import sys, csv, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = pathlib.Path("work/newds/data_dictionary.csv")
rows = list(csv.reader(p.open(encoding="utf-8", errors="replace")))
print("rows:", len(rows))
for r in rows[:60]:
    print("  " + " | ".join(x[:44] for x in r[:4]))
