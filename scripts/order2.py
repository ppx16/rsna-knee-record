import json, re, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.loads(pathlib.Path("kernels/kernel-0943cfm2eff/rsna-knee-restructured-version-3.ipynb").read_text(encoding="utf-8", errors="replace"))
lines = "".join(nb["cells"][50].get("source", [])).split("\n")
# enclosing def of line 389
d = None
for k in range(389, -1, -1):
    if lines[k].lstrip().startswith("def "):
        d = (k, lines[k].strip()); break
print("enclosing def of build_cache call:", d)
if d:
    name = re.match(r"def (\w+)", d[1]).group(1)
    print("name:", name)
    for k, l in enumerate(lines):
        if name + "(" in l and not l.lstrip().startswith("def "):
            print("  CALL at %4d| %s" % (k, l.strip()[:150]))
print("\n---- region 560-600 ----")
for k in range(560, 601):
    if k < len(lines): print("%4d| %s" % (k, lines[k][:150]))
