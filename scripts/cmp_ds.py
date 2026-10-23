import json, sys, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for d in ["kernel-0943cf","kernel-0943cfm","kernel-0943cfm2eff"]:
    p = pathlib.Path("kernels")/d/"kernel-metadata.json"
    if not p.exists(): print(d, "MISSING"); continue
    m = json.loads(p.read_text(encoding="utf-8"))
    ds = m.get("dataset_sources", [])
    print("== %-22s %d dataset_sources" % (d, len(ds)))
    for s in ds: print("     %s" % s)
