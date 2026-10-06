"""Round start: what is true right now?"""
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
api.authenticate()

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
