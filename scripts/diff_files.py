import sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
def names(p):
    s = set()
    for l in open(p, encoding="utf-8"):
        if "\t" in l: s.add(l.split("\t")[0])
    return s
a = names("work/mcl_urls.txt"); b = names("work/eff_urls.txt")
print("0943cfm files: %d   0943cfm2eff files: %d" % (len(a), len(b)))
print("\n---- ONLY in 0943cfm2eff (i.e. produced by the leg) ----")
for n in sorted(b - a): print("   %s" % n)
print("\n---- ONLY in 0943cfm ----")
for n in sorted(a - b)[:15]: print("   %s" % n)
