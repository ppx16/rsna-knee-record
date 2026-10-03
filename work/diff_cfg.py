import json, re, sys, pathlib, difflib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
def cfg(p):
    nb = json.loads(pathlib.Path(p).read_text(encoding="utf-8", errors="replace"))
    return "".join(nb["cells"][2].get("source", []))
A = cfg("downloads/research/fresh0924/maverickss26_rsna_knee_restructured_version_3/rsna-knee-restructured-version-3.ipynb")
B = cfg("kernels/kernel-0943cfm/rsna-knee-restructured-version-3.ipynb")
print("frontier cell2 chars=%d   ours=%d" % (len(A), len(B)))
d = list(difflib.unified_diff(A.split("\n"), B.split("\n"), "frontier", "ours", lineterm="", n=1))
print("diff lines: %d" % len(d))
for l in d[:60]:
    print("   %s" % l[:150])
