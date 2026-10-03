import csv, sys, pathlib, collections, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rows = list(csv.DictReader(pathlib.Path("work/newds/manifest_processed.csv").open(encoding="utf-8", errors="replace")))
by = collections.defaultdict(list)
for r in rows: by[r["StudyInstanceUID"]].append(r)
n = len(by)
dummy = sum(1 for v in by.values() if any("dummy" in (x["descripcion"] or "").lower() for x in v))
per = collections.Counter(len(v) for v in by.values())
print("studies: %d   with a DummySeriesDesc: %d (%.1f%%)" % (n, dummy, 100*dummy/n))
print("series per study: min %d  median %d  max %d" % (min(per), sorted(per.elements())[n//2], max(per)))
SAG = re.compile(r"sag", re.I); FS = re.compile(r"fs|spair|spir|fatsat|fat_sat", re.I)
PD = re.compile(r"pd|dp|pdw|proton", re.I)
T2 = re.compile(r"t2", re.I)
def has(v, rx1, rx2):
    return sum(1 for x in v if rx1.search(x["descripcion"] or "") and rx2.search(x["descripcion"] or ""))
stats = collections.Counter()
for v in by.values():
    s = has(v, SAG, FS); t = has(v, SAG, T2)
    stats["sag+FS present" if s else "NO sag+FS"] += 1
    stats["sag+T2 present" if t else "NO sag+T2"] += 1
    stats["has ANY sagittal" if any(SAG.search(x["descripcion"] or "") for x in v) else "NO sagittal at all"] += 1
for k in sorted(stats): print("   %-22s %5d  (%.1f%%)" % (k, stats[k], 100*stats[k]/n))
