"""v2: locate OUR team properly (usernames live in TeamMemberUserNames, not TeamName),
and use the file's own Rank column instead of a re-derived ordering."""

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
    raw = z.read(z.namelist()[0]).decode("utf-8-sig", "replace")

rows = list(csv.DictReader(io.StringIO(raw)))
n = len(rows)
print(f"teams: {n}")
print(f"columns: {list(rows[0].keys())}\n")

CANDIDATES = ("ppxl16",)
print("--- searching TeamMemberUserNames AND TeamName ---")
found = []
for r in rows:
    hay = (r.get("TeamMemberUserNames", "") + "|" + r.get("TeamName", "")).lower()
    if any(c in hay for c in CANDIDATES):
        found.append(r)
        print(f"  Rank={r['Rank']:<6} Score={r['Score']:<7} Team={r.get('TeamName')!r} "
              f"subs={r.get('SubmissionCount')} last={r.get('LastSubmissionDate')} "
              f"members={r.get('TeamMemberUserNames')!r}")
if not found:
    print("  NOT FOUND")
    # show a couple of rows verbatim so we can see what the columns really hold
    print("\n  first 3 rows verbatim:")
    for r in rows[:3]:
        print("   ", r)

print("\n--- the shape of the 0.94x band (real Rank column) ---")
from collections import defaultdict
band = defaultdict(list)
for r in rows:
    try:
        s = round(float(r["Score"]), 3)
    except (TypeError, ValueError):
        continue
    if s >= 0.938:
        band[s].append(int(r["Rank"]))
for s in sorted(band, reverse=True):
    rk = band[s]
    print(f"  {s:.3f}: {len(rk):>4} teams   rank {min(rk)} .. {max(rk)}")

print("\n--- medal cutoffs (bronze = top 10% of the board) ---")
for pct, label in ((0.002, "gold  (top 10 + 0.2%)"), (0.05, "silver 5%"), (0.10, "bronze 10%")):
    k = max(1, int(n * pct))
    r = next((x for x in rows if int(x["Rank"]) == k), None)
    if r is None:
        for x in rows:
            if int(x["Rank"]) >= k:
                r = x
                break
    print(f"  {label}: rank {k} -> score {r['Score']}")

print("\n--- exact rows at rank 490..505 (the bronze edge) ---")
for r in rows:
    rk = int(r["Rank"])
    if 490 <= rk <= 505:
        print(f"  rank {rk:>4}  {r['Score']:<7} {(r.get('TeamName') or '')[:34]:<34} {r.get('LastSubmissionDate')}")
