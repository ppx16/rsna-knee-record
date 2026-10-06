"""Live board -> our rank, the 0.943 block, and the in-block percentile instrument."""
import csv
import io
import subprocess
import datetime as dt

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
api = KaggleApi()
# ⚠️ RETRY THE AUTH CALL ITSELF, not just the import. This line authenticates ONCE at module scope, before
# any polling loop, so a transient api.kaggle.com SSLEOFError here is instant death -- and for a watcher that
# means the result it waited hours for is never read. See work/patch_auth_guard.py.
# Self-contained on purpose: it imports what it needs, so it does not depend on the host file's imports.
import sys as _auth_sys
import time as _auth_time

_auth_last = None
for _auth_try in range(6):
    try:
        api.authenticate()
        break
    except Exception as _auth_e:  # noqa: BLE001
        _auth_last = _auth_e
        _auth_sys.stderr.write("api.authenticate() attempt %d failed: %s\n"
                               % (_auth_try + 1, type(_auth_e).__name__))
        _auth_sys.stderr.flush()
        _auth_time.sleep(8)
else:
    _auth_sys.stderr.write("COULD NOT AUTHENTICATE after 6 attempts (%r) -- TRANSPORT failure, "
                           "not a result.\n" % (_auth_last,))
    raise SystemExit(2)

print("now (UTC):", dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"))

api.competition_leaderboard_download(COMP, path="work/lb_live", quiet=False)
import glob
import zipfile

zp = "work/lb_live/rsna-knee-abnormality-detection.zip"
with zipfile.ZipFile(zp) as z:
    name = z.namelist()[0]
    print("zip member:", name)
    raw = z.read(name).decode("utf-8")
rows = list(csv.DictReader(io.StringIO(raw)))
print("rows:", len(rows))
print("cols:", list(rows[0].keys()))

for r in rows[:2]:
    print("  head:", r)

# score column name differs by API version
def score_of(r):
    for k in ("Score", "score"):
        if k in r:
            return float(r[k])
    raise KeyError(r.keys())

for r in rows:
    r["_s"] = score_of(r)

rows.sort(key=lambda r: -r["_s"])
for i, r in enumerate(rows, 1):
    r["_rank"] = i

# our best submission's score
OUR = {"56814081": None, "56826372": None, "56845973": None}
for r in rows:
    sid = str(r.get("SubmissionId") or r.get("SubmissionID") or "")
    for k in OUR:
        if k in sid:
            OUR[k] = (r["_rank"], r["_s"])

print()
print("=== our measured submissions on the board ===")
for k, v in OUR.items():
    print(f"  {k}: {v}")

top = rows[0]["_s"]
print()
print(f"=== score bands ===")
bands = {}
for r in rows:
    bands.setdefault(r["_s"], []).append(r["_rank"])
for s in sorted(bands, reverse=True)[:6]:
    rk = bands[s]
    print(f"  {s}: n={len(rk):5d}  ranks {min(rk)}..{max(rk)}")

# the block containing us
print()
print("=== US (ppxl16) ===")
me = [r for r in rows if "ppxl16" in (r.get("TeamMemberUserNames", "") + "|" + r.get("TeamName", "")).lower()]
for r in me:
    print(f"  rank {r['_rank']}  score {r['_s']}  team={r.get('TeamName')!r}  "
          f"members={r.get('TeamMemberUserNames')!r}  last={r.get('LastSubmissionDate')}")
our_rank = me[0]["_rank"] if me else None

target = 0.943
blk = bands.get(target, [])
if blk and our_rank:
    lo, hi = min(blk), max(blk)
    pct = 100.0 * (our_rank - lo) / max(hi - lo, 1)
    print()
    print(f"=== the {target} block ===")
    print(f"  n={len(blk)}  ranks {lo}..{hi}  (width {hi-lo+1})")
    print(f"  our rank {our_rank} -> in-block percentile {pct:.2f}%  (lower is better)")
    print(f"  bronze line (top 10% of {len(rows)}) = {int(len(rows)*0.10)}")
    print(f"  margin to bronze = {int(len(rows)*0.10) - our_rank:+d}")
else:
    print()
    print("=== the 0.943 block ===")
    if blk:
        print(f"  n={len(blk)}  ranks {min(blk)}..{max(blk)}  (width {max(blk)-min(blk)+1})")
    print(f"  ours: {me}")

print()
print("=== top of board ===")
for r in rows[:5]:
    print(f"  rank {r['_rank']:3d}  {r['_s']}  {r.get('TeamName','?')}")

# ---- persist -------------------------------------------------------------
if our_rank and blk:
    import json
    import pathlib

    now = dt.datetime.now(dt.timezone.utc)
    iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    snap = {
        "at": iso,
        "n_teams": len(rows),
        "our_rank": our_rank,
        "our_score": me[0]["_s"],
        "bronze_line": int(len(rows) * 0.10),
        "margin": int(len(rows) * 0.10) - our_rank,
        "block": {"score": target, "n": len(blk), "lo": min(blk), "hi": max(blk)},
        "in_block_pct": round(100.0 * (our_rank - min(blk)) / max(max(blk) - min(blk), 1), 3),
        "top_score": rows[0]["_s"],
    }
    pathlib.Path("state").mkdir(exist_ok=True)
    with open(f"state/lb_snapshot_{iso.replace(':','')}.json", "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2)
    with open("state/rank_track.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": iso, "rank": our_rank,
                             "teams": len(rows), "in_block_pct": snap["in_block_pct"]}) + "\n")
    print()
    print("persisted:", f"state/lb_snapshot_{iso.replace(':','')}.json", "+ rank_track.jsonl")

