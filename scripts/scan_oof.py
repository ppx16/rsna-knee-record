import sys, re, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
seen, hits = set(), []
for page in range(1, 9):
    try:
        ds = api.dataset_list(search="rsna-knee", page=page, sort_by="updated")
    except Exception as e:
        print("page %d failed: %s" % (page, str(e)[:90])); break
    if not ds: break
    for d in ds:
        ref = str(d.ref)
        if ref in seen: continue
        seen.add(ref)
print("datasets seen:", len(seen))
OOF = re.compile(r"(oof|pred|prob|logit|valid)", re.I)
for ref in sorted(seen):
    try:
        fs = api.dataset_list_files(ref).files
    except Exception:
        continue
    names = [f.name for f in fs]
    off = [n for n in names if OOF.search(n) and n.lower().endswith((".npy",".csv",".npz",".parquet"))]
    if off:
        hits.append((ref, len(names), off[:6]))
print("\n==== datasets publishing OOF/prediction files ====")
for ref, n, off in hits:
    print("  %-58s %3d files" % (ref, n))
    for o in off: print("        %s" % o)
