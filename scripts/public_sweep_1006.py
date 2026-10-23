"""Fresh public sweep: new competition kernels and discussion topics since the 2026-10-04 sweep.

The last exhaustive public sweep was 2026-10-04. Two days have passed and public artifacts are the one
input that can change without us doing anything, so re-scan before proposing any strategy.
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
import datetime as dt

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
now = dt.datetime.now(dt.timezone.utc)

api = KaggleApi()
api.authenticate()

print("=== competition kernels, newest first (public, this competition) ===")
seen = 0
for sort in ("dateRun", "voteCount"):
    try:
        ks = api.kernels_list(competition=COMP, sort_by=sort, page_size=25)
    except Exception as e:  # noqa: BLE001
        print("  list(%s) FAILED: %s" % (sort, str(e)[:110]))
        continue
    print("  -- sort_by=%s: %d rows --" % (sort, len(ks or [])))
    for k in (ks or []):
        lr = str(getattr(k, "last_run_time", ""))[:19]
        print("     %-58s %-19s votes=%s" % (str(getattr(k, "ref", ""))[:58], lr,
                                             getattr(k, "total_votes", "?")))
        seen += 1
    if seen > 40:
        break
print()

print("=== discussion topics, newest first ===")
try:
    tp = api.competition_list_topics(COMP, sort_by="recent", page_size=25)
    tl = getattr(tp, "topics", None) or tp or []
    print("  %d topics" % len(tl))
    for t in tl:
        print("     %-8s %-19s %s" % (getattr(t, "id", "?"),
                                      str(getattr(t, "last_post_date", ""))[:19],
                                      str(getattr(t, "title", ""))[:78]))
except Exception as e:  # noqa: BLE001
    print("  topics FAILED: %s: %s" % (type(e).__name__, str(e)[:140]))
