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
REF = "56872230"
KG = r"D:\kaggle\venv-kaggle\Scripts\kaggle.exe"
STATE = pathlib.Path("state")
RESULT = STATE / "reader_arm_result.json"

before = json.loads((STATE / "submit_56872230.json").read_text(encoding="utf-8"))["before_snapshot"]
print("BEFORE:", json.dumps(before), flush=True)

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
if before and after:
    d_pct = after["in_block_pct"] - before["in_block_pct"]
    rec["delta_in_block_pct"] = round(d_pct, 3)
    rec["delta_margin"] = after["margin"] - before["margin"]
    rec["verdict"] = ("REAL GAIN (>=3 points better)" if d_pct <= -3 else
                      "no measurable change" if abs(d_pct) < 3 else "WORSE")
RESULT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(json.dumps(rec, indent=2))
