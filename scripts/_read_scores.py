"""Fetch the RSNA submissions table and read the SCORE column safely.

The CLI renders each submission as a multi-line block whose description is our own long free text, so a
truncated row shows no score at all -- which is exactly how a real result gets missed. This runs the CLI to a
file, then associates each `SubmissionStatus.<STATE> <score>` field with the nearest PRECEDING ref.
"""
import os
import pathlib
import re
import subprocess
import time

KG = r"D:\kaggle\venv-kaggle\Scripts\kaggle.exe"
COMP = "rsna-knee-abnormality-detection"
OUT = pathlib.Path(os.environ["TEMP"]) / "subs_fetch.txt"

text = ""
for _ in range(4):
    r = subprocess.run([KG, "competitions", "submissions", "-c", COMP],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = (r.stdout or "") + (r.stderr or "")
    OUT.write_text(text, encoding="utf-8")
    if "SSLError" not in text and "Traceback" not in text and len(text) > 500:
        break
    time.sleep(8)

lines = text.splitlines()
print("fetched lines:", len(lines), "| bytes:", len(text))

# walk the file: remember the most recent ref seen, attach any status field that follows it
cur = None
found = {}
for ln in lines:
    m = re.match(r"^(\d{8})\s+(\S+)\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", ln.strip())
    if m:
        cur = m.group(1)
    sm = re.search(r"SubmissionStatus\.(\w+)\s+([\d.]*)", ln)
    if sm and cur:
        found[cur] = (sm.group(1), sm.group(2) or None)

print("submissions with a parsed status:", len(found))
for ref in ("56884048", "56878092", "56876044", "56872230", "56814081", "56613303", "56848247"):
    st, sc = found.get(ref, ("?", None))
    print("  %s : status=%-9s public=%s" % (ref, st, sc if sc else "-"))
print()
print("ALL parsed scores (top 12 rows):")
for ref, (st, sc) in list(found.items())[:12]:
    print("   %s  %-9s %s" % (ref, st, sc if sc else "-"))
