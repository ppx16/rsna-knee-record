import sys, re, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
t0=time.time()
seen={}
for page in (1,2):
    try:
        ds = api.dataset_list(search="rsna-knee", page=page, sort_by="updated")
    except Exception as e:
        print("page %d failed: %s" % (page, str(e)[:90])); break
    for d in ds:
        seen[str(d.ref)] = str(getattr(d, "lastUpdated", "") or "")
print("datasets listed: %d  (%.1fs)" % (len(seen), time.time()-t0))
items = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)[:28]
print("\n---- newest 28 by lastUpdated ----")
for ref, when in items:
    print("  %-58s %s" % (ref, when[:19]))
