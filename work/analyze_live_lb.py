"""Read the live RSNA leaderboard download and answer the only questions that matter:
where are we NOW, how far has the field moved, and what score does a medal need TODAY.

The previous sheets reasoned from a 2026-10-02 scrape and a 2026-10-01 snapshot; the user is
right that those numbers are stale while the board is live.
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
import zipfile
import pathlib

WORK = pathlib.Path(r"D:\kaggle\rsna-knee\work\lb_live")
zp = WORK / "rsna-knee-abnormality-detection.zip"

with zipfile.ZipFile(zp) as z:
    names = z.namelist()
    print("zip contains:", names)
    raw = z.read(names[0]).decode("utf-8", "replace")

rows = list(csv.DictReader(io.StringIO(raw)))
print(f"columns: {list(rows[0].keys())}")
print(f"teams on the board: {len(rows)}")

def score(r):
    for k in ("Score", "score"):
        if k in r and r[k] not in ("", None):
            try:
                return float(r[k])
            except ValueError:
                pass
    return None

teams = []
for r in rows:
    s = score(r)
    name = r.get("TeamName") or r.get("teamName") or ""
    tid = r.get("TeamId") or r.get("teamId") or ""
    if s is not None:
        teams.append((s, name, tid))

# Kaggle ranks ties densely from the top; the file is already in rank order
teams_sorted = sorted(teams, key=lambda t: -t[0])
n = len(teams_sorted)
print(f"scored teams: {n}")
print(f"top score: {teams_sorted[0][0]}")
print(f"bottom score: {teams_sorted[-1][0]}")

print("\n--- where are WE ---")
for i, (s, name, tid) in enumerate(teams_sorted, 1):
    if "ppxl16" in name.lower() or "ppxl16" in str(tid):
        print(f"  rank {i} / {n}   score {s}   team {name!r} (id {tid})")

print("\n--- score distribution around our zone ---")
from collections import Counter
c = Counter(s for s, _, _ in teams_sorted)
print(f"{'score':>7} {'teams':>6} {'cum':>6} {'cum%':>7}")
cum = 0
for s in sorted(c, reverse=True):
    cum += c[s]
    if s >= 0.930:
        print(f"{s:>7.3f} {c[s]:>6} {cum:>6} {100*cum/n:>6.2f}%")

print("\n--- medal cutoffs, if bronze = top 10% ---")
for pct, label in ((0.02, "gold-ish 2%"), (0.05, "silver 5%"), (0.10, "bronze 10%")):
    k = int(n * pct)
    print(f"  {label}: rank {k} -> score {teams_sorted[k-1][0]}")

print("\n--- how many teams sit at each score >= 0.938 ---")
for s in sorted({x for x, _, _ in teams if x >= 0.938}, reverse=True):
    print(f"  {s:.3f}: {c[s]} teams")

print("\n--- top 15 with submission dates ---")
for i, r in enumerate(rows[:15], 1):
    s = score(r)
    nm = (r.get("TeamName") or r.get("teamName") or "").strip()
    dt = r.get("SubmissionDate") or r.get("submissionDate") or ""
    print(f"  {i:>3}. {s}  {nm[:28]:<28} {dt}")
