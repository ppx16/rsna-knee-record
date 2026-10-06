"""Stage 2: for every username above us, list their PUBLIC kernels and datasets.

Turns "356 teams beat us" into "here is what those people actually published".
Slow (one API call per user), so run it in the background and cache to disk.
"""

# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded
# `print("... \u26a0 ...")` raises UnicodeEncodeError at that line and kills the script MID-REPORT, while
# everything printed earlier is already on screen -- so the run looks complete. Self-contained on purpose:
# it imports `sys` itself and depends on nothing else in this tree.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream: leave it alone, prints still degrade
    pass
# ----------------------------------------------------------------------------------------------------
import json
import pathlib
import time
import traceback

from kaggle.api.kaggle_api_extended import KaggleApi

WORK = pathlib.Path(r"D:\kaggle\rsna-knee\work")
above = json.loads((WORK / "above_us.json").read_text(encoding="utf-8"))

users = set()
for t in above["above"]:
    for u in (t.get("members") or "").split(","):
        u = u.strip()
        if u:
            users.add(u)
users = sorted(users)
print(f"usernames to probe: {len(users)}", flush=True)

api = KaggleApi()
api.authenticate()

out_path = WORK / "above_us_public.json"
cache = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
print(f"already cached: {len(cache)}", flush=True)


def sw(fn, *a, **k):
    for attempt in range(3):
        try:
            return fn(*a, **k)
        except Exception as e:
            if attempt == 2:
                return {"__error": f"{type(e).__name__}: {str(e)[:160]}"}
            time.sleep(3 + 3 * attempt)


for i, u in enumerate(users):
    if u in cache:
        continue
    ks = sw(api.kernels_list, user=u, page_size=100)
    ds = sw(api.dataset_list, user=u)
    rec = {}
    if isinstance(ks, dict):
        rec["kernels_error"] = ks["__error"]
        ks = []
    if isinstance(ds, dict):
        rec["datasets_error"] = ds["__error"]
        ds = []
    rec["kernels"] = [{"ref": getattr(k, "ref", None), "title": getattr(k, "title", None),
                       "votes": getattr(k, "total_votes", None),
                       "lastRun": str(getattr(k, "last_run_time", ""))} for k in ks]
    rec["datasets"] = [{"ref": getattr(d, "ref", None), "title": getattr(d, "title", None),
                        "votes": getattr(d, "vote_count", None),
                        "size": getattr(d, "total_bytes", None)} for d in ds]
    cache[u] = rec
    if (i + 1) % 10 == 0:
        out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        nk = sum(len(v.get("kernels", [])) for v in cache.values())
        nd = sum(len(v.get("datasets", [])) for v in cache.values())
        print(f"  [{i+1}/{len(users)}] kernels={nk} datasets={nd}", flush=True)
    time.sleep(0.4)

out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

nk = sum(len(v.get("kernels", [])) for v in cache.values())
nd = sum(len(v.get("datasets", [])) for v in cache.values())
print(f"\nDONE. users probed={len(cache)} public kernels={nk} public datasets={nd}")
owners_k = sorted({u for u, v in cache.items() if v.get("kernels")})
owners_d = sorted({u for u, v in cache.items() if v.get("datasets")})
print(f"users above us who publish at least one kernel : {len(owners_k)}")
print(f"users above us who publish at least one dataset: {len(owners_d)}")
print("\nkernel-publishing users:", owners_k[:60])
print("\ndataset-publishing users:", owners_d[:60])
