"""Recover BOTH outstanding results from nothing but this repository. Run this if the watchers died.

WHY THIS EXISTS. The two measurements in flight are watched by background jobs
(`work/watch_rdr_submission.py`, `work/watch_distilled_gate.py`). Those jobs are **session-scoped**: if the
session ends before a measurement lands, the job is gone and its `state/*.json` result was never written.
The results are still recoverable -- the score is on Kaggle's submission table and the gate's number is in
its kernel log -- but only if someone knows where to look, and "remember to run the right script" is not a
handover.

So this is the no-watcher path, in one command:

    python work/collect_results.py

It prints, for each outstanding measurement, the raw evidence and the pre-registered verdict. It writes
nothing it did not read.

⚠️ It NEVER reports a verdict it could not earn. A transport failure is reported as `UNASKED`, a missing
score as `PENDING`, and an unreadable log as `UNREAD` -- because a probe that cannot tell "absent" from "I
failed to ask" is worse than no probe, which this repository has learned four separate times.
"""
from __future__ import annotations

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

api = KaggleApi()
api.authenticate()
now = dt.datetime.now(dt.timezone.utc)
print("collected at %s   (%.1f days to the 2026-10-22 deadline)\n"
      % (now.strftime("%Y-%m-%dT%H:%M:%SZ"),
         (dt.datetime(2026, 10, 22, 23, 59, tzinfo=dt.timezone.utc) - now).total_seconds() / 86400))


def cli(args, tries=3):
    for _ in range(tries):
        r = subprocess.run([KG, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = (r.stdout or "") + (r.stderr or "")
        if "SSLError" not in out and "Traceback" not in out:
            return out
        time.sleep(6)
    return None


# ---------------------------------------------------------------- the reader leg's score
print("=" * 92)
print("MEASUREMENT 1 -- reader leg, submission 56872230")
txt = cli(["competitions", "submissions", "-c", COMP])
score = status = None
if txt is None:
    print("   score: UNASKED (transport failed on every attempt)")
else:
    for ln in txt.splitlines():
        m = re.match(r"^56872230\s+\S+\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S*\s+(.*)$", ln.strip())
        if m:
            # ⚠️ the score group must be OPTIONAL. A PENDING row ends right after `PENDING` with no
            # trailing whitespace, so requiring `\s+([\d.]+)` fails to match at all -- and the row then
            # reports status `?`, which reads like a FAULT when it actually means "still being scored".
            sm = re.search(r"SubmissionStatus\.(\w+)(?:\s+([\d.]+))?", m.group(2))
            status = sm.group(1) if sm else "?"
            score = (sm.group(2) or None) if sm else None
            print("   submitted: %s   status: %s   public: %s" % (m.group(1), status, score or "-"))
            break
    else:
        print("   the submission is ABSENT from the table (never submitted, or removed)")

before = json.loads((STATE / "submit_56872230.json").read_text(encoding="utf-8"))["before_snapshot"]
print("   BEFORE snapshot: rank %s, line %s, margin %s, in-block %.3f%%"
      % (before["our_rank"], before["bronze_line"], before["margin"], before["in_block_pct"]))
print("   READ RULE (fixed in advance): in-block percentile vs %.3f%%, margin vs %s."
      % (before["in_block_pct"], before["margin"]))
print("      >= 3 points better  -> real gain; switch the primary and fire the pre-registered TTA arm")
print("      within +/-3 points  -> no measurable change; the pair stays 56814081 + 56613303")
print("      worse               -> the composition is harmful; drop the leg")
print("   ⚠️ The DISPLAYED SCORE is not the criterion: the author's own pipeline scored 0.944/0.944/0.943.")

# ---------------------------------------------------------------- the gate's gold AUC
print()
print("=" * 92)
print("MEASUREMENT 2 -- distilled zero-shot gate, kernel ppxl16/rsna-knee-msft2-gate-distilled")
st = None
try:
    st = str(getattr(api.kernels_status("ppxl16/rsna-knee-msft2-gate-distilled"), "status", ""))
except Exception:  # noqa: BLE001
    st = None
print("   kernel status: %s" % ("UNASKED (transport)" if st is None else st))
g = STATE / "gate_distilled_result.json"
if g.is_file():
    rec = json.loads(g.read_text(encoding="utf-8"))
    print("   watcher file: state/gate_distilled_result.json")
    print("   gold-58 macro-AUC: %s" % rec.get("gold_macro_auc"))
    print("   decision (from the watcher's pre-registered rule): %s" % rec.get("decision"))
else:
    print("   watcher file: ABSENT -- the watcher did not survive. Recover the log with:")
    print("      python work/read_finished_runs.py        (fetches and greps the kernel log)")
    print("   or read it directly:  kaggle kernels output ppxl16/rsna-knee-msft2-gate-distilled")
    print("   and look for the line  [zeroshot] zero training steps: GOLD macro-AUC <N>")
print("   READ RULE (fixed in advance): >=0.92 geometry matches + as strong as claimed -> wire as a member")
print("      0.90-0.92 usable but no better than msft2 (0.9069) | 0.85-0.90 below the A167 gate")
print("      <0.85 geometry mismatch -> reject permanently")

# ---------------------------------------------------------------- where the state lives
print()
print("=" * 92)
print("AUTHORITATIVE STATE: the `CURRENT STATE` block at the top of STATUS.md.")
print("Final pair: 56814081 (primary) + 56613303 (second) -- BROWSER-ONLY, and six other submissions")
print("also display 0.942-0.943, so pick by DESCRIPTION TEXT, never by score.")
