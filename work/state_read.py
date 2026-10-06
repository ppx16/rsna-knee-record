"""Full state read before any strategy talk: clock, quota, submissions, board, kernels."""
import datetime as dt
import json
import pathlib
import subprocess

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
KG = r"D:\kaggle\venv-kaggle\Scripts\kaggle.exe"

now = dt.datetime.now(dt.timezone.utc)
print("NOW (UTC): %s" % now.strftime("%Y-%m-%d %H:%M:%SZ"))
print("deadline  : 2026-10-22 23:59Z  -> %.1f days left" % ((dt.datetime(2026, 10, 22, 23, 59, tzinfo=dt.timezone.utc) - now).total_seconds() / 86400))
print()

print("=== quota ===")
r = subprocess.run([KG, "quota"], capture_output=True, text=True, encoding="utf-8", errors="replace")
print((r.stdout or r.stderr or "").strip())
print()

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

print("=== submissions in the current UTC day (%s) ===" % now.strftime("%Y-%m-%d"))
subs = api.competition_submissions(COMP)
today = [s for s in subs if str(getattr(s, "date", "")).startswith(now.strftime("%Y-%m-%d"))]
print("  %d submitted today" % len(today))
for s in today:
    print("   ", getattr(s, "date", ""), getattr(s, "ref", ""),
          str(getattr(s, "status", "")).split(".")[-1])
print("  => remaining today: %d of 5" % (5 - len(today)))
print()

print("=== kernel statuses (top 6) ===")
ks = api.kernels_list(user="ppxl16", sort_by="dateRun", page_size=6)
for k in ks:
    st = "?"
    for _ in range(3):
        try:
            st = str(getattr(api.kernels_status(k.ref), "status", "")).split(".")[-1]
            break
        except Exception:
            pass
    print("   %-46s %s  %s" % (k.ref, str(getattr(k, "last_run_time", ""))[:19], st))
