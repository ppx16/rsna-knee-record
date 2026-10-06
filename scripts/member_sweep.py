"""P1.1 -- sweep EVERY arm we hold through the honest member-addition gate, and rank them.

WHY THIS IS THE FIRST MOVE (docs/REVIEW-2026-09-27.md, Phase 1)
    The public leaderboard is 0.959 and we are 0.943, a gap of 0.016 = 2x the noise band.  The gap
    is NOT a missing checkpoint: kernel-0943 already mounts essentially the whole published model
    zoo.  The one mechanism this repo has ever MEASURED working is adding a decorrelated member of
    our own: cn0 (0.8769, i.e. BELOW the frontier's best arm) gave +0.0028, CI [+0.0011, +0.0043]
    on a parent-quality base (A131/A137).

    So before training anything, ask the cheap question: of all the arms we already hold but do not
    ship, does any add on the same gate?  This costs no quota at all.

WHAT IT DOES
    Enumerates every base `raptor_oof_<tag>_fold<N>.npz` under downloads/{cv,rent,a5lite_*}, then
    runs `member_add_test.py --strong` for each -- the parent-quality variant, where the base gets
    the published Rad heads AND every blend weight is chosen on one half and scored on the other.
    Then it ranks by the honest delta.

    ⚠️ It SHELLS OUT rather than importing, on purpose: member_add_test.py's main() is the audited,
    already-trusted implementation of this gate (it is what predicted cn0).  Re-implementing its
    blend logic here to save a few seconds of data loading would create a second implementation
    that could silently disagree with the first -- which is precisely the failure this repo keeps
    recording.  A subprocess cannot drift.

Usage:
    python scripts/member_sweep.py                 # every arm, strong base
    python scripts/member_sweep.py --weak          # the cheaper 0.917 base instead
    python scripts/member_sweep.py --only cn0@f0,r3360@f0
    python scripts/member_sweep.py --selftest      # prove the parser on synthetic output
"""
from __future__ import annotations


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
import argparse
import glob
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable
TEST = ROOT / "scripts" / "member_add_test.py"

# base npz only: tags are alnum, and the epoch/swa/ens variants are trailing groups we skip
ARM_RE = re.compile(r"raptor_oof_([A-Za-z0-9]+?)(?:_(ep\d+|epens|swa))?_fold(\d+)\.npz$")
GLOBS = ("downloads/cv/**/raptor_oof_*.npz",
         "downloads/rent/**/raptor_oof_*.npz",
         "downloads/a5lite_*/raptor_oof_*.npz")

DELTA_RE = re.compile(r"base\+member\s*:\s*([0-9.]+)\s+delta\s+([+-][0-9.]+)")
CI_RE = re.compile(r"95%\s*CI\s*\[([+-][0-9.]+),\s*([+-][0-9.]+)\]\s+P\(delta>0\)=([0-9.]+)")


def arms():
    """tag@fN -> path, for every base OOF we hold."""
    out = {}
    for g in GLOBS:
        for p in glob.glob(str(ROOT / g), recursive=True):
            m = ARM_RE.search(os.path.basename(p))
            if not m or m.group(2):
                continue
            out["%s@f%s" % (m.group(1), m.group(3))] = p
    return out


def parse(text):
    """-> (member_added_score, delta, lo, hi, p) or None.  Loud about a missing field."""
    d = DELTA_RE.search(text)
    c = CI_RE.search(text)
    if not d or not c:
        return None
    return (float(d.group(1)), float(d.group(2)), float(c.group(1)), float(c.group(2)),
            float(c.group(3)))


def run(arm, weak, boot):
    cmd = [PY, str(TEST), "--arm", arm, "--boot", str(boot)]
    if not weak:
        cmd.append("--strong")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def selftest():
    good = "  honest strong base+member : 0.9201   delta +0.0028\n  95% CI [+0.0011, +0.0043]  P(delta>0)=0.97"
    got = parse(good)
    assert got is not None and abs(got[1] - 0.0028) < 1e-9, got
    assert parse("no fields here") is None, "a missing delta must return None, not a guess"
    print("SELFTEST OK: parser reads the gate's output and refuses to guess when fields are absent")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weak", action="store_true", help="use the 0.917 base, not the strong one")
    ap.add_argument("--only", default="", help="comma list of arms")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    all_arms = arms()
    if a.only:
        want = [t.strip() for t in a.only.split(",") if t.strip()]
        missing = [t for t in want if t not in all_arms]
        if missing:
            raise SystemExit("!! not found: %s\n   available: %s"
                             % (missing, ", ".join(sorted(all_arms))))
        all_arms = {t: all_arms[t] for t in want}
    if not all_arms:
        raise SystemExit("!! no arms found -- check the GLOBS in this file")
    print("arms to test: %d   base=%s" % (len(all_arms), "0.917 (weak)" if a.weak else "parent-quality (strong)"))

    rows, failed = [], []
    for i, arm in enumerate(sorted(all_arms), 1):
        rc, out = run(arm, a.weak, a.boot)
        got = parse(out)
        if rc != 0 or got is None:
            failed.append(arm)
            tail = [l for l in out.strip().splitlines() if l.strip()][-1:] or ["(no output)"]
            print("[%2d/%2d] %-12s FAILED  %s" % (i, len(all_arms), arm, tail[0][:96]))
            continue
        score, delta, lo, hi, p = got
        rows.append((delta, arm, score, lo, hi, p))
        print("[%2d/%2d] %-12s +%.4f  delta %+.4f  CI [%+.4f, %+.4f]  P=%.2f"
              % (i, len(all_arms), arm, score, delta, lo, hi, p))

    print("\n" + "=" * 92)
    print("RANKED BY HONEST DELTA (does adding this arm help a parent-quality base?)")
    print("%-14s %-9s %-9s %-22s %s" % ("arm", "added", "delta", "95% CI", "verdict"))
    for delta, arm, score, lo, hi, p in sorted(rows, reverse=True):
        if lo > 0:
            v = "PASS  (CI excludes 0)"
        elif hi < 0:
            v = "HURTS (CI excludes 0)"
        else:
            v = "no signal"
        print("%-14s %-9.4f %-+9.4f [%+.4f, %+.4f]      %s" % (arm, score, delta, lo, hi, v))
    print("=" * 92)
    print("GATE: only 'PASS' arms are candidates to integrate.  A 'no signal' arm is NOT a rejection")
    print("of the family -- it is one fold-0 arm on one instrument; it just does not earn a slot.")
    if failed:
        print("⚠️ %d arm(s) could not be scored: %s" % (len(failed), ", ".join(failed)))
    print("\nreference: cn0@f0 is the arm that shipped (it is not in downloads/{cv,rent} -- it was")
    print("built on the rented box), and the published bar is +0.0028 CI [+0.0011, +0.0043].")
    return 0


if __name__ == "__main__":
    sys.exit(main())
