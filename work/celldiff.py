import json, sys, difflib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
a=json.load(open(r"kernels/kernel-0943cfm2eff/rsna-knee-restructured-version-3.ipynb",encoding="utf-8"))
b=json.load(open(r"work/pulled_v3/rsna-knee-0943cfm2eff.ipynb",encoding="utf-8"))
ta="".join(a["cells"][49].get("source",[])); tb="".join(b["cells"][49].get("source",[]))
print("local type=%s len=%d | pushed type=%s len=%d" % (a["cells"][49]["cell_type"], len(ta), b["cells"][49]["cell_type"], len(tb)))
for l in list(difflib.unified_diff(tb.split("\n"), ta.split("\n"), "pushed", "local", lineterm=""))[:24]:
    print("   %s" % l[:150])
