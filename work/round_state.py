"""Round start: what is true right now?"""

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

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
KG = r"D:\kaggle\venv-kaggle\Scripts\kaggle.exe"
now = dt.datetime.now(dt.timezone.utc)
print("NOW (UTC): %s   |  %.1f days to the 2026-10-22 deadline"
      % (now.strftime("%Y-%m-%d %H:%M:%SZ"),
         (dt.datetime(2026, 10, 22, 23, 59, tzinfo=dt.timezone.utc) - now).total_seconds() / 86400))

r = subprocess.run([KG, "quota"], capture_output=True, text=True, encoding="utf-8", errors="replace")
print((r.stdout or r.stderr or "").strip())

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

subs = api.competition_submissions(COMP)
today = [s for s in subs if str(getattr(s, "date", "")).startswith(now.strftime("%Y-%m-%d"))]
print("submitted today: %d / 5" % len(today))

# score of the reader arm, via the CLI table (the Python client returns publicScore=None)
txt = ""
for _ in range(3):
    r = subprocess.run([KG, "competitions", "submissions", "-c", COMP],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    txt = (r.stdout or "") + (r.stderr or "")
    if "SSLError" not in txt:
        break

print()
print("=== newest 8 submissions with scores ===")
rows = []
for ln in txt.splitlines():
    ln = ln.strip()
    m = re.match(r"^(\d{8})\s+\S+\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S*\s+(.*)$", ln)
    if not m:
        continue
    ref, date, rest = m.groups()
    sm = re.search(r"SubmissionStatus\.(\w+)\s+([\d.]+|)", rest)
    rows.append((date, ref, (sm.group(2) or "-") if sm else "?", sm.group(1) if sm else "?"))
for d, ref, sc, st in rows[:8]:
    print("   %s  %-9s %-7s %s" % (d, ref, sc, st))

print()
snaps = sorted(pathlib.Path("state").glob("lb_snapshot_*.json"))
last = json.loads(snaps[-1].read_text(encoding="utf-8"))
print("=== last recorded board snapshot (%s) ===" % last["at"])
print("   teams %s | rank %s | line %s | margin %s | in-block %.3f%% | top %s"
      % (last["n_teams"], last["our_rank"], last["bronze_line"], last["margin"],
         last["in_block_pct"], last["top_score"]))
