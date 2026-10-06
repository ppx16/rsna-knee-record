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
import sys
import time

# ⚠️ THIS SCRIPT PRINTS EMOJI, AND THIS CONSOLE IS GBK.
# On Windows the default stdout encoding here is cp936/GBK, which cannot encode `⚠️`; the first print that
# contains one raises UnicodeEncodeError and KILLS THE SCRIPT MID-REPORT. That is exactly what happened:
# measurement 1 printed, then the traceback swallowed measurements 2 and 3 -- so the recovery path was
# silently reporting only one third of what it exists to report, while looking like it had run.
# Fix at the source (all prints) rather than deleting the emoji one at a time.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream (e.g. redirected): fall back to plain prints
    pass

# ⚠️ THE RETRY MUST WRAP THE IMPORT, NOT JUST THE CALLS.
# `kaggle/__init__.py` line 8 calls `api.authenticate()` at import time, so a transient api.kaggle.com
# SSLEOFError surfaces as a failure of the `from kaggle... import KaggleApi` statement itself. The first
# version of this file imported at module top with no retry, and died with a full traceback on the very
# error it exists to survive -- a recovery path that cannot recover is worse than none.
KaggleApi = None
for _attempt in range(6):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi as _K

        KaggleApi = _K
        break
    except Exception as _e:  # noqa: BLE001  (SSLEOFError and friends, raised from the import)
        print("auth import attempt %d failed: %s" % (_attempt + 1, type(_e).__name__), file=sys.stderr)
        time.sleep(8)

if KaggleApi is None:
    print("COULD NOT AUTHENTICATE after 6 attempts -- this is a TRANSPORT failure, not a result.")
    print("Re-run `python work/collect_results.py`; nothing about the measurements has been learned.")
    raise SystemExit(2)

COMP = "rsna-knee-abnormality-detection"
KG = r"D:\kaggle\venv-kaggle\Scripts\kaggle.exe"
STATE = pathlib.Path("state")

api = KaggleApi()
# ⚠️ THE IMPORT IS NOT THE ONLY PLACE THIS AUTHENTICATES. The loop above guards `from kaggle... import`,
# but the SDK authenticates AGAIN here -- and this call was left bare. On 2026-10-06 it died with exactly
# the SSLEOFError this file exists to survive, at THIS line, AFTER the import retry had already succeeded.
# A recovery path with two authentication points and one guard has no guard.
_last = None
for _try in range(6):
    try:
        api.authenticate()
        break
    except Exception as _e:  # noqa: BLE001
        _last = _e
        print("api.authenticate() attempt %d failed: %s" % (_try + 1, type(_e).__name__), file=sys.stderr)
        time.sleep(8)
else:
    print("COULD NOT AUTHENTICATE after 6 attempts (%r) -- TRANSPORT, not a result." % (_last,))
    print("Re-run `python work/collect_results.py`; nothing about the measurements has been learned.")
    raise SystemExit(2)
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
print("      ⚠️ Those are the values AS REGISTERED. They are NOT the bar to compare against: drift has moved")
print("         them, and the drift-adjusted bar is printed below. Do not stop reading at this line.")
print("      >= 3 points better  -> real gain; switch the primary and fire the pre-registered TTA arm")
print("      within +/-3 points  -> no measurable change; the pair stays 56814081 + 56613303")
print("      worse               -> the composition is harmful; drop the leg")
print("   ⚠️ The DISPLAYED SCORE is not the criterion: the author's own pipeline scored 0.944/0.944/0.943.")

# ---------------------------------------------------------------- WHAT THE RANK INSTRUMENT CAN AND CANNOT SHOW
# ⚠️ THE PUBLIC BOARD SHOWS OUR **BEST** SUBMISSION, NOT OUR NEWEST. So the rank moves ONLY if this arm
# becomes our new best. That splits the outcome three ways, and two of them move the rank by nothing:
#
#   score > 0.943  -> the best changed, the rank moves, the in-block read above is meaningful.
#   score = 0.943  -> the board CANNOT distinguish this arm from the best we already have. Any rank movement
#                     seen then is DRIFT, not this arm. The pre-registered '+/-3 points -> pair unchanged'
#                     branch applies -- but by CONSTRUCTION rather than by measurement. And 0.943 is a normal
#                     author-side outcome (0.944/0.944/0.943), so it refutes nothing either.
#   score < 0.943  -> not our best; the rank stays wherever drift puts it. Drop the leg.
#
# Writing this down because 'the percentile did not move' reads like 'no effect' when it can equally mean
# 'the instrument is blind to this by design'. This board has already produced one sub-resolution trap --
# 990 teams inside a single displayed value -- and this is the same trap wearing different clothes.
print()
if score is None:
    print("   ⚠️ No public score yet, so nothing can be compared. When it arrives: the board shows our BEST")
    print("      submission, so a score of 0.943 or lower will NOT move the rank at all, and any movement")
    print("      seen at that point is drift rather than this arm.")
else:
    try:
        _s = float(score)
    except Exception:  # noqa: BLE001
        _s = None
    if _s is None:
        print("   public score %r is not a number -- do not interpret it" % (score,))
    elif _s > 0.943:
        print("   score %.3f > 0.943 -> THIS ARM IS THE NEW BEST; the rank read above is meaningful." % _s)
        print("      ⚠️ 0.944 is the bronze step: ranks 319-504, entirely inside the line.")
        print("      See docs/TARGET-2026-10-06j-one-displayed-step-is-bronze.md")
    elif _s == 0.943:
        print("   score 0.943 -> *** THE INSTRUMENT IS BLIND TO THIS ARM. *** The board shows our BEST")
        print("      submission, so a tie at 0.943 cannot be told apart from 56814081 by rank, and any rank")
        print("      movement is DRIFT. The pre-registered '+/-3 points -> pair unchanged' branch therefore")
        print("      holds BY CONSTRUCTION, not by measurement -- and 0.943 is a normal author-side result")
        print("      (0.944/0.944/0.943), so it is not a refutation of the composition either.")
    else:
        print("   score %.3f < 0.943 -> not our best; the rank will not move for this arm. Drop the leg;" % _s)
        print("      the final pair is unchanged.")

# ---------------------------------------------------------------- DRIFT: the BEFORE baseline is going stale
# ⚠️ WHY THIS IS HERE. The rule above compares a NEW rank against a BEFORE snapshot -- but the board does not
# hold still. Teams keep passing our 0.943, so our rank FALLS with time even though our own score cannot
# change. Measured this session, the rate is ACCELERATING: 4.1 -> 6.6 -> 10.2 ranks/h, and the margin to the
# bronze line went -29 (10-05 16:05) -> -136 (10-06 08:05). We are not hovering at the bronze boundary; we
# are 136 ranks BEHIND it and losing ~170 ranks/day.
#
# A raw comparison against a hours-old baseline therefore UNDERSTATES a real gain, and can report a genuine
# improvement as "no measurable change" -- or, worse, let drift alone look like the arm made things worse.
# So print the DRIFT-ADJUSTED baseline: what the old best would read if re-measured now.


def _drift_rates(track: pathlib.Path):
    """(window_hours, ranks_per_hour) for a recent burst and for a robust longer window."""
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
    if len(pts) < 2:
        return None
    pts.sort()
    last_t, last_r = pts[-1]
    out = []
    for hours in (3.0, 48.0):
        base = None
        for t, r in pts[:-1]:
            if (last_t - t).total_seconds() / 3600.0 <= hours:
                base = (t, r)
                break
        if base is None or base[0] == last_t:
            continue
        span = (last_t - base[0]).total_seconds() / 3600.0
        if span < 1.0:
            continue
        out.append((span, (last_r - base[1]) / span))
    return (last_t, last_r, out) if out else None


_d = _drift_rates(STATE / "rank_track.jsonl")
_snaps = sorted(STATE.glob("lb_snapshot_*.json"))
_before_t = None
if _d:
    _last_t, _last_r, _rates = _d
    # locate the BEFORE snapshot's own timestamp by matching its rank in the track
    for _t, _r in [(dt.datetime.strptime(json.loads(l)["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc),
                    json.loads(l)["rank"])
                   for l in (STATE / "rank_track.jsonl").read_text(encoding="utf-8").splitlines()
                   if l.strip() and '"rank"' in l and "teams" in l]:
        if _r == before["our_rank"]:
            _before_t = _t
    print()
    print("   DRIFT (this baseline does NOT hold still):")
    for _span, _rate in _rates:
        print("      over the last %5.1f h: %+6.2f ranks/h  (%+.0f ranks/day)"
              % (_span, _rate, _rate * 24))
    if _before_t is not None:
        _dt_h = (_last_t - _before_t).total_seconds() / 3600.0
        _recent = _rates[0][1]
        _adj_rank = before["our_rank"] + _recent * _dt_h
        _blk = json.loads(_snaps[-1].read_text(encoding="utf-8"))["block"] if _snaps else None
        print("      the BEFORE snapshot is %.1f h old -> at %+.2f ranks/h the old best would now read"
              % (_dt_h, _recent))
        print("      rank ~%.0f (observed then: %d)" % (_adj_rank, before["our_rank"]))
        if _blk and _blk.get("hi", 0) > _blk.get("lo", 0):
            _adj_pct = (_adj_rank - _blk["lo"]) / float(_blk["hi"] - _blk["lo"]) * 100.0
            print("      = %.3f%% in-block against TODAY's block %d..%d, vs the %.3f%% recorded before"
                  % (_adj_pct, _blk["lo"], _blk["hi"], before["in_block_pct"]))
            print("      => compare the new arm against ~%.3f%%, NOT %.3f%%; the gap between them is drift"
                  % (_adj_pct, before["in_block_pct"]))
            print("         that this submission did not cause.")
        print("      ⚠️ the rate is an ESTIMATE from rank_track.jsonl and it is accelerating; if it moved")
        print("         during the wait, re-take a snapshot and recompute rather than trusting this number.")
    else:
        print("      ⚠️ could not match the BEFORE rank in rank_track.jsonl -> adjustment NOT applied.")
else:
    print()
    print("   DRIFT: not estimable (state/rank_track.jsonl missing or has <2 usable points).")
    print("      => the raw BEFORE comparison stands, and any verdict is weaker for it.")

# ---------------------------------------------------------------- the reader ARMS judged on the SCORE
# ⚠️ SAME INSTRUMENT, DIFFERENT QUESTION. The plain reader leg above is judged on how far the IN-BLOCK
# PERCENTILE moves. These are judged on whether they CROSS THE BRONZE STEP -- because 0.944 lands at ranks
# 319-504, entirely inside the line, while 0.943 is a ~990-wide block that straddles it. So the criterion
# here is the DISPLAYED SCORE.
# Written as a LOOP rather than a copy per arm, deliberately: a third copy is exactly where a registered
# rule quietly stops matching the arm it was registered for, which is the failure this repository keeps
# recording. Adding an arm is now one tuple.
SCORE_ARMS = [
    ("56876044", "TTA reader arm (w=0.30 + TTA)", "08:50Z", "18:50-19:50Z",
     "docs/PREREG-2026-10-06n-tta-reader-arm.md"),
    ("56878092", "reader leg at w=0.15", "10:23Z", "20:23-21:23Z",
     "docs/PREREG-2026-10-06o-reader-w015.md"),
    ("56884048", "reader leg on the COMMUNITY stack (0943 + reader, w=0.30)", "15:1xZ", "~01:1xZ next day",
     "docs/PREREG-2026-10-06r-reader-on-community-stack.md"),
]
for _i, (_ref, _label, _fired, _expect, _prereg) in enumerate(SCORE_ARMS, start=3):
    print()
    print("=" * 92)
    print("MEASUREMENT %d -- %s, submission %s" % (_i, _label, _ref))
    _sc = None
    _txt = cli(["competitions", "submissions", "-c", COMP])
    if _txt is None:
        print("   score: UNASKED (transport failed on every attempt)")
    else:
        for ln in _txt.splitlines():
            m = re.match(r"^%s\s+\S+\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\S*\s+(.*)$" % _ref, ln.strip())
            if m:
                sm = re.search(r"SubmissionStatus\.(\w+)(?:\s+([\d.]+))?", m.group(2))
                _sc = (sm.group(2) or None) if sm else None
                print("   submitted: %s   status: %s   public: %s"
                      % (m.group(1), sm.group(1) if sm else "?", _sc or "-"))
                break
        else:
            print("   the submission is ABSENT from the table (never submitted, or removed)")
    _rec = STATE / ("submit_%s.json" % _ref)
    if _rec.is_file():
        _b = json.loads(_rec.read_text(encoding="utf-8")).get("before_snapshot") or {}
        print("   BEFORE (%s): rank %s, line %s, margin %s, in-block %s%%"
              % (_fired, _b.get("our_rank"), _b.get("bronze_line"), _b.get("margin"),
                 _b.get("in_block_pct")))
    print("   expected score ~%s (10-11 h, FACTS A7)" % _expect)
    print("   READ RULE (preregistered, SCORE-based -- a threshold question, not a rank increment):")
    print("      0.944    *** CROSSES THE BRONZE LINE *** (ranks 319-504, entirely inside) -> becomes")
    print("               primary, second stays 56613303, and STOP spending slots")
    print("      0.943    invisible to the rank instrument (the board carries our BEST) -> pair unchanged")
    print("      <=0.942  not our best -> pair unchanged, and the reader-variant line ends here")
    print("      prereg  : %s" % _prereg)

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
    # ⚠️ "the file is absent" does NOT mean "the watcher died". It is written only when a watcher reaches a
    # terminal kernel state, so a still-running watcher looks identical to a dead one from here. The first
    # version printed "the watcher did not survive", which is an inference this script cannot make --
    # exactly the "I failed to ask" vs "absent" confusion this repository has hit four times.
    print("   watcher file: NOT WRITTEN YET (a running watcher writes it only on a terminal state,")
    print("                 so this is indistinguishable here from a dead watcher -- check both)")
    print("   either way the log is recoverable directly:")
    print("      python work/read_finished_runs.py        (fetches and greps the kernel log)")
    print("   or:  kaggle kernels output ppxl16/rsna-knee-msft2-gate-distilled")
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
