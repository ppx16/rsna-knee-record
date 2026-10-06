"""Watch submission 56872230 (the reader arm) to a score, and capture the AFTER snapshot.

⚠️ READ IT RIGHT. The prereg (`docs/PREREG-2026-10-06-reader-leg.md` §3) fixes the reading:
  * primary judgement = the IN-BLOCK PERCENTILE moving >= 3 points;
  * the MARGIN is reported alongside, because the bronze line is an absolute top-10 % line;
  * the displayed score is NOT the criterion -- the author's own pipeline scored 0.944/0.944/0.943,
    so landing on 0.943 is normal and is not a refutation.

The before-snapshot is read from state/submit_56872230.json, so the comparison can never be made
against a remembered number.
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
import json
import pathlib
import re
import subprocess
import time

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
KG = r"D:\kaggle\venv-kaggle\Scripts\kaggle.exe"
STATE = pathlib.Path("state")

# ⚠️ GENERALISED OVER ARMS -- AND OVER WHICH RULE APPLIES, because they are NOT judged the same way.
# The plain reader leg is read on the IN-BLOCK PERCENTILE. The TTA and w=0.15 arms are read on the DISPLAYED
# SCORE, because for them the question is whether they CROSS the bronze step, not how far they move a rank.
# A watcher that applied the wrong rule would write a confident wrong verdict -- which this file already did
# once, before the drift correction, so it is worth saying twice.
import argparse as _ap

_p = _ap.ArgumentParser()
_p.add_argument("--ref", default="56872230")
_p.add_argument("--arm", default="0943cfmsft2-rdr")
_p.add_argument("--rule", choices=("rank", "score"), default="rank",
                help="rank = in-block percentile (plain reader leg); score = threshold (TTA / w=0.15)")
_p.add_argument("--out", default="state/reader_arm_result.json")
_a = _p.parse_args()
REF = _a.ref
ARM = _a.arm
RULE = _a.rule
RESULT = pathlib.Path(_a.out)

before = json.loads((STATE / ("submit_%s.json" % REF)).read_text(encoding="utf-8"))["before_snapshot"]
print("BEFORE:", json.dumps(before), flush=True)
print("arm=%s ref=%s rule=%s -> %s" % (ARM, REF, RULE, RESULT), flush=True)

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
POLL = 900
for i in range(120):                       # up to 30 h
    try:
        subs = api.competition_submissions(COMP)
        row = next((s for s in subs if str(getattr(s, "ref", "")) == REF), None)
        st = str(getattr(row, "status", "")).split(".")[-1] if row else "?"
        if st and st != "PENDING":
            print("[%s] %s" % (time.strftime("%H:%M:%S"), st), flush=True)
            break
        if i % 4 == 0:
            print("[%s] still PENDING (poll %d)" % (time.strftime("%H:%M:%S"), i), flush=True)
    except Exception as e:  # noqa: BLE001
        print("poll transport error:", type(e).__name__, flush=True)
    time.sleep(POLL)

# score from the CLI table (the Python client returns publicScore=None for every row)
score = None
for _ in range(4):
    r = subprocess.run([KG, "competitions", "submissions", "-c", COMP],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    txt = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"^%s\s+\S+\s+\S+\s+.*?SubmissionStatus\.(\w+)\s+([\d.]+|)" % REF, txt, re.M)
    if m:
        score = m.group(2) or None
        break
    time.sleep(20)

# AFTER snapshot
subprocess.run(["python", "work/lb_snapshot.py"], capture_output=True, text=True)
snaps = sorted(STATE.glob("lb_snapshot_*.json"))
after = json.loads(snaps[-1].read_text(encoding="utf-8")) if snaps else None

rec = {"ref": REF, "arm": "0943cfmsft2-rdr", "public_score": score, "before": before, "after": after,
       "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


# ⚠️ THE RAW BEFORE/AFTER COMPARISON IS WRONG ON THIS BOARD -- and it was wrong here until now.
# The board drifts: teams keep passing our 0.943, so our rank FALLS with time even when our own score cannot
# change. In-block percentile therefore RISES on its own. Measured this session at +145..+240 ranks/day,
# which over an 8 h wait is ~65 ranks -- about +6.5 points in a 990-wide block. The rule this replaces would
# have written "WORSE" for an arm that did nothing at all, and the pre-registered branch for WORSE is
# 'the composition is harmful; drop the leg'. That is a confidently wrong verdict produced by drift alone,
# written into the machine-readable file a handover reads.
# So: age the BEFORE rank by the measured drift, re-express it in TODAY's block geometry, judge against THAT.
# collect_results.py was corrected the same way; this is the script that persists the answer to disk.
def _rate_per_hour(track, t0, t1):
    if not track.is_file():
        return None
    pts = []
    for ln in track.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
            t = dt.datetime.strptime(d["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d.get("rank"), int):
            pts.append((t, d["rank"]))
    pts.sort()
    for win in ([(t, r) for t, r in pts if t0 <= t <= t1], [(t, r) for t, r in pts if t <= t1]):
        if len(win) >= 2:
            span = (win[-1][0] - win[0][0]).total_seconds() / 3600.0
            if span >= 0.5:
                return (win[-1][1] - win[0][1]) / span
    return None


if before and after:
    raw = after["in_block_pct"] - before["in_block_pct"]
    rec["delta_in_block_pct_raw"] = round(raw, 3)
    rec["delta_margin"] = after["margin"] - before["margin"]
    try:
        t0 = dt.datetime.strptime(before["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        t1 = dt.datetime.strptime(after["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except Exception:  # noqa: BLE001
        t0 = t1 = None
    rate = _rate_per_hour(STATE / "rank_track.jsonl", t0, t1) if (t0 and t1) else None
    blk = after.get("block") or {}
    if rate is not None and t0 and t1 and blk.get("hi", 0) > blk.get("lo", 0):
        hrs = (t1 - t0).total_seconds() / 3600.0
        adj_rank = before["our_rank"] + rate * hrs
        adj_pct = (adj_rank - blk["lo"]) / float(blk["hi"] - blk["lo"]) * 100.0
        d = after["in_block_pct"] - adj_pct
        rec["drift_ranks_per_hour"] = round(rate, 2)
        rec["drift_adjusted_before_pct"] = round(adj_pct, 3)
        rec["delta_in_block_pct_drift_adjusted"] = round(d, 3)
        rec["rank_verdict"] = ("REAL GAIN (>=3 points better)" if d <= -3 else
                               "no measurable change" if abs(d) < 3 else "WORSE")
        rec["rank_verdict_basis"] = "drift-adjusted"
    else:
        rec["rank_verdict"] = ("REAL GAIN (>=3 points better)" if raw <= -3 else
                               "no measurable change" if abs(raw) < 3 else "WORSE")
        rec["rank_verdict_basis"] = "RAW -- drift could NOT be estimated, so this is WEAK by construction"

# ---- the PRIMARY verdict, which is NOT the same question for every arm -------------------------------
# The rank block above answers 'did this move us WITHIN the 0.943 band?'. For the TTA and w=0.15 arms the
# registered question is 'did this CROSS the bronze step?', and the rank cannot answer it at all: the board
# carries our BEST submission, so anything at or below 0.943 leaves the rank exactly where drift put it.
try:
    _s = float(score) if score else None
except Exception:  # noqa: BLE001
    _s = None
if RULE == "rank":
    rec["verdict"] = rec.get("rank_verdict", "NOT READ")
    rec["verdict_rule"] = "in-block percentile vs the drift-adjusted bar (prereg: reader leg)"
    if _s is not None and _s <= 0.943:
        rec["instrument_blind"] = (
            "public score %.3f is not above our existing best 0.943, so the board cannot show this arm at "
            "all -- any rank movement recorded above is DRIFT, and the pre-registered '+/-3 -> pair "
            "unchanged' branch holds by construction rather than by measurement" % _s)
else:
    rec["verdict_rule"] = "DISPLAYED SCORE vs the bronze step (prereg: TTA / w=0.15)"
    if _s is None:
        rec["verdict"] = "NOT READ (no public score yet) -- never record this as 'worse'"
    elif _s >= 0.945:
        rec["verdict"] = "ABOVE the measured public ceiling (0.944) -- record the anomaly, do not rush"
    elif _s == 0.944:
        rec["verdict"] = ("*** CROSSES THE BRONZE LINE *** (ranks 319-504, entirely inside) -> becomes "
                          "PRIMARY, second stays 56613303, and STOP spending slots")
    elif _s == 0.943:
        rec["verdict"] = ("INSTRUMENT IS BLIND to this arm (the board carries our BEST): a tie at 0.943 "
                          "cannot be told apart from 56814081 by rank -> pair unchanged, by construction")
    else:
        rec["verdict"] = "not our best (<0.943) -> pair unchanged, and the reader-variant line ends here"
RESULT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(json.dumps(rec, indent=2))
