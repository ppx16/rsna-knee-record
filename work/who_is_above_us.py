"""Map the teams above us to usernames, then look for their public footprints.

The technique that found the M448 gap: take the teams that score higher than us, find their
PUBLIC notebooks, and diff their mounted `dataset_sources` / `model_sources` against ours.
Anything they mount that we do not is a candidate explanation.

Stage 1 (this file): who is above us, and which of them publish anything at all.
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
import csv
import io
import json
import pathlib
import zipfile
from collections import Counter

LB = sorted(pathlib.Path(r"D:\kaggle\rsna-knee\work\lb_live3").glob("*.zip"))[-1]
with zipfile.ZipFile(LB) as z:
    rows = list(csv.DictReader(io.StringIO(z.read(z.namelist()[0]).decode("utf-8-sig", "replace"))))

n = len(rows)
me = [r for r in rows if "ppxl16" in (r.get("TeamMemberUserNames", "") + r.get("TeamName", "")).lower()]
my_score = float(me[0]["Score"]) if me else 0.943
my_rank = int(me[0]["Rank"]) if me else None
print(f"board: {n} teams | us: rank {my_rank}, score {my_score}\n")

above = [r for r in rows if float(r["Score"]) > my_score]
print(f"teams strictly ABOVE us: {len(above)}")

# how many distinct users are up there, and how concentrated
users = []
for r in above:
    for u in (r.get("TeamMemberUserNames") or "").split(","):
        u = u.strip()
        if u:
            users.append(u)
print(f"distinct usernames above us: {len(set(users))}")
print(f"teams above us that are solo (1 member): "
      f"{sum(1 for r in above if len((r.get('TeamMemberUserNames') or '').split(',')) == 1)}")

print("\n--- score histogram above us ---")
c = Counter(r["Score"] for r in above)
for s in sorted(c, reverse=True):
    print(f"  {s}: {c[s]}")

print("\n--- the TOP 40 teams (the frontier) ---")
top = sorted(rows, key=lambda r: int(r["Rank"]))[:40]
for r in top:
    subs = r.get("SubmissionCount")
    members = (r.get("TeamMemberUserNames") or "").strip()
    print(f"  rank {r['Rank']:>4}  {r['Score']}  subs={subs:<4} {(r.get('TeamName') or '')[:30]:<30} [{members[:60]}]")

out = pathlib.Path(r"D:\kaggle\rsna-knee\work\above_us.json")
out.write_text(json.dumps({
    "snapshot": LB.name,
    "board_size": n,
    "our_rank": my_rank,
    "our_score": my_score,
    "above": [{"rank": int(r["Rank"]), "score": r["Score"],
               "team": r.get("TeamName"), "members": r.get("TeamMemberUserNames"),
               "subs": r.get("SubmissionCount")} for r in above],
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nwrote {out}")
