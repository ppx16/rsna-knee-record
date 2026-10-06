"""Close the Repair-v1 question properly, then look at the CoAt datasets the search surfaced.

Part 1: does ANY file in the three mounted CoAt datasets contain 'repair'? The search found no dataset
named for it, but the manifest could be shipped INSIDE one of them -- so check the listings rather than
trusting the slug search.

Part 2: the search surfaced several CoAt checkpoints we have never catalogued. Each is either a new
member (a lever) or a repackage of something we mount (not one). The file listing distinguishes them.
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
import pathlib
import time

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

print("=== Part 1: 'repair' anywhere inside the three mounted CoAt datasets? ===")
for ref in ("mattiaangeli/rsna-knee-coat-resgated-ep10-top3",
            "mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2",
            "mattiaangeli/rsna-knee-coatnet-global96-top3"):
    try:
        fl = api.dataset_list_files(ref)
        files = getattr(fl, "files", None) or fl or []
    except Exception as e:  # noqa: BLE001
        print("   %-50s FAILED %s" % (ref, str(e)[:60]))
        continue
    names = [getattr(f, "name", "?") for f in files]
    hits = [n for n in names if "repair" in n.lower()]
    print("   %-50s %2d files | repair hits: %s" % (ref, len(names), hits or "NONE"))
    print("        %s" % sorted(names)[:20])

print()
print("=== Part 2: the CoAt checkpoints the search surfaced ===")
NEW = ["tonylica/rsna-knee-bend-dinov3-0917-repro-assets",
       "kawaharataishi/rsna-knee-coatnet-distilled-checkpoint",
       "haideptry/rsna-knee-coatnet-25d-masked-checkpoint",
       "tannguyenngoc/rsna-knee-coatnet-a100-best",
       "myominhtet/rsna-knee-coat5fold"]
for ref in NEW:
    print("-" * 90)
    print(ref)
    try:
        got = api.dataset_list(search=ref.split("/", 1)[1]) or []
        me = [d for d in got if getattr(d, "ref", "") == ref] or got
        d = me[0] if me else None
        if d is not None:
            print("   title   :", str(getattr(d, "title", ""))[:80])
            print("   subtitle:", str(getattr(d, "subtitle", ""))[:110])
            print("   license :", getattr(d, "licenseName", "?"), "| bytes:", getattr(d, "totalBytes", "?"))
    except Exception as e:  # noqa: BLE001
        print("   list failed:", str(e)[:90])
    for _ in range(3):
        try:
            fl = api.dataset_list_files(ref)
            files = getattr(fl, "files", None) or fl or []
            print("   %d file(s):" % len(files))
            for f in files[:22]:
                nm = getattr(f, "name", "?")
                sz = getattr(f, "totalBytes", 0) or 0
                try:
                    sz = "%.1f MB" % (float(sz) / 1e6)
                except Exception:
                    sz = "?"
                print("      %-11s %s" % (sz, nm))
            if len(files) > 22:
                print("      ... %d more" % (len(files) - 22))
            break
        except Exception as e:  # noqa: BLE001
            print("   files retry: %s" % str(e)[:70])
            time.sleep(6)
