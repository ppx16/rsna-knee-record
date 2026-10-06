"""Assemble the CURRENT state from files, so a STATUS block can be written from evidence not memory.

This repository has already been damaged once by a STATUS.md whose prose had drifted ("the previous
session's 8 commits sat unpushed for 26 h and STATUS.md was actively misleading in the meantime"). The
DELTA stack has since grown past a dozen sections, so the top of the file needs one authoritative block —
and every number in it must come from a file, not from recollection.
"""
import datetime as dt
import glob
import json
import pathlib
import re
import subprocess

COMP = "rsna-knee-abnormality-detection"
KG = r"D:\kaggle\venv-kaggle\Scripts\kaggle.exe"
STATE = pathlib.Path("state")

now = dt.datetime.now(dt.timezone.utc)
print("NOW %s | %.1f days to the 2026-10-22 deadline"
      % (now.strftime("%Y-%m-%dT%H:%M:%SZ"),
         (dt.datetime(2026, 10, 22, 23, 59, tzinfo=dt.timezone.utc) - now).total_seconds() / 86400))

r = subprocess.run([KG, "quota"], capture_output=True, text=True, encoding="utf-8", errors="replace")
q = {}
for ln in (r.stdout or "").splitlines():
    p = ln.split()
    if len(p) >= 4 and p[0] in ("GPU", "TPU"):
        q[p[0]] = (p[1], p[2], p[3], p[4] if len(p) > 4 else "")
print("QUOTA", q)

# submission scores from the CLI table (the Python client returns publicScore=None)
txt = ""
for _ in range(3):
    r = subprocess.run([KG, "competitions", "submissions", "-c", COMP],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    txt = (r.stdout or "") + (r.stderr or "")
    if "SSLError" not in txt:
        break
rows = {}
for ln in txt.splitlines():
    m = re.match(r"^(\d{8})\s+\S+\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S*\s+(.*)$", ln.strip())
    if not m:
        continue
    ref, date, rest = m.groups()
    sm = re.search(r"SubmissionStatus\.(\w+)\s+([\d.]+|)", rest)
    rows[ref] = (date, (sm.group(2) or "-") if sm else "?", sm.group(1) if sm else "?")
today = sum(1 for v in rows.values() if v[0].startswith(now.strftime("%Y-%m-%d")))
print("SUBMISSIONS today=%d/5  total visible=%d" % (today, len(rows)))
for ref in ("56872230", "56848247", "56826372", "56814081", "56613303", "56599932"):
    print("   %s -> %s" % (ref, rows.get(ref, "ABSENT")))

snaps = sorted(STATE.glob("lb_snapshot_*.json"))
last = json.loads(snaps[-1].read_text(encoding="utf-8"))
print("BOARD %s teams=%s rank=%s line=%s margin=%s in-block=%.3f%% top=%s"
      % (last["at"], last["n_teams"], last["our_rank"], last["bronze_line"], last["margin"],
         last["in_block_pct"], last["top_score"]))

print("PENDING MEASUREMENTS")
g = STATE / "gate_distilled_result.json"
print("   reader leg 56872230 :", rows.get("56872230", ("?",))[1:])
print("   distilled gate      :", "result written" if g.is_file() else "not yet")

print("ARTEFACTS")
for pat in ("docs/PREREG-2026-10-06-reader-leg.md", "docs/REVISION-2026-10-06c-second-slot-reverts-to-cflat.md",
            "docs/VERIFY-2026-10-06-reader-leg-is-clean.md", "docs/READ-2026-10-06e-stackpp-reader-changes.md",
            "docs/CHECK-2026-10-06f-lixin-checkpoint-closed.md",
            "docs/EXTRACT-2026-10-06g-tta-variant-source-ready.md"):
    print("   %-64s %s" % (pat, "present" if pathlib.Path(pat).is_file() else "MISSING"))
