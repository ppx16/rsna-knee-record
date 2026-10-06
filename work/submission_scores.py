"""Read the per-submission PUBLIC SCORES.

⚠️ `ApiSubmission.publicScore` comes back None from the Python client for every row (it did all
session), so the CLI table is the route. Run through the venv CLI: the PATH `kaggle` fails 6/6 on
the OAuth path.
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
import re
import subprocess

KG = r"D:\kaggle\venv-kaggle\Scripts\kaggle.exe"
COMP = "rsna-knee-abnormality-detection"

r = subprocess.run([KG, "competitions", "submissions", "-c", COMP],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
txt = (r.stdout or "") + (r.stderr or "")
if "SSLError" in txt or "Traceback" in txt:
    print("TRANSPORT ERROR -- retry")
    raise SystemExit(1)

rows = []
# ⚠️ FIRST VERSION TOOK THE FIRST `0.xxx` IN THE ROW AND GOT 56814081 WRONG: its description contains
# "gold58 0.9069", so the regex reported 0.9069 as the public score. The real value is 0.943. The
# score is the token immediately AFTER the status column -- anchor on that, never on "first number".
for ln in txt.splitlines():
    ln = ln.strip()
    if not ln or ln.lower().startswith(("ref", "warning")):
        continue
    m = re.match(r"^(\d{8})\s+(\S+)\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S*\s+(.*)$", ln)
    if not m:
        continue
    ref, fname, date, rest = m.groups()
    # ⚠️ the score groups must be OPTIONAL: a PENDING row ends right after `PENDING` with no trailing
    # whitespace, so the old `\s+([\d.]+|)` failed to match and the row reported `?` -- which reads like
    # a FAULT when it actually means "still being scored".
    sm = re.search(r"SubmissionStatus\.(\w+)(?:\s+([\d.]+))?(?:\s+([\d.]+))?", rest)
    status = sm.group(1) if sm else "?"
    score = (sm.group(2) or "-") if sm else "-"
    desc = rest[:rest.find("SubmissionStatus")] if "SubmissionStatus" in rest else rest
    rows.append((ref, date, score, status, desc.strip()[:52]))

print("%-10s %-20s %-7s %-9s %s" % ("ref", "date", "public", "status", "description"))
for ref, date, sc, status, desc in rows:
    print("%-10s %-20s %-7s %-9s %s" % (ref, date, sc, status, desc))

by_ref = {r[0]: r[2] for r in rows}
print()
print("=== scores for the arms in play ===")
for ref in ("56848247", "56845973", "56826372", "56814081", "56798408", "56791368",
            "56599932", "56613303", "56570151", "56570144", "56568520"):
    print("   %s -> %s" % (ref, by_ref.get(ref, "(not found)")))

