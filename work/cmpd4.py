import json, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
def meta(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8", errors="replace"))
A = meta("work/d4blend/kernel-metadata.json")                 # pjmathematician/rsna-knee-d4-blend
B = meta("kernels/kernel-0943s-2coat/kernel-metadata.json")   # ours, the 0.943 stack
for label, d in (("THEIRS (d4-blend, 0.946)", A), ("OURS (0943s-2coat, 0.943)", B)):
    ds = d.get("dataset_sources", [])
    print("== %s" % label)
    print("   dataset_sources: %d  (empty/private placeholders: %d)" % (len(ds), sum(1 for x in ds if not x)))
    for k in ("kernel_sources","model_sources","competition_sources","machine_shape","enable_gpu","enable_internet"):
        if k in d: print("   %-20s %s" % (k, d[k]))
a = [x for x in A.get("dataset_sources", []) if x]
b = [x for x in B.get("dataset_sources", []) if x]
print("\n---- ONLY in THEIRS (public, and we do NOT have it) ----")
for x in a:
    if x not in b: print("   %s" % x)
print("---- ONLY in OURS ----")
for x in b:
    if x not in a: print("   %s" % x)
print("\npublic datasets they mount that we also mount: %d / %d" % (sum(1 for x in a if x in b), len(a)))
print("private placeholders in theirs: %d   (we have %d)" % (sum(1 for x in A['dataset_sources'] if not x),
                                                              sum(1 for x in B['dataset_sources'] if not x)))
