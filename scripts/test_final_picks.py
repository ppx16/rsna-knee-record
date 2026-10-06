"""Exercise report_final_picks() on synthetic submission histories.

Why this exists: HANDOFF 53 reversed the hedge preference in status.py, but the
live history at the time still had `halfway-routing` PENDING, so the changed
branch was never taken -- "the fix ran" and "the fix was verified" are different
claims (HANDOFF 52.3).  These cases pin the decision down offline.

report_final_picks() only prints, so calling it with fake rows cannot touch the
real ledger.  Keep it that way: if it ever grows a side effect, this test must
stop calling it directly.

Usage:  python scripts/test_final_picks.py
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
import contextlib
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import status  # noqa: E402

CANON = status.BASELINE_MARKER          # marks the canonical 0.941 recipe


def row(ref, score, desc):
    return {"ref": ref, "publicScore": score, "description": desc}


def picks(rows):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        status.report_final_picks(rows)
    out = buf.getvalue()
    offence = hedge = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("offence"):
            offence = s.split()[2]
        if s.startswith("hedge") and "ref" in s:
            hedge = s.split()[2]
    return offence, hedge, out


CASES = []

# 1. THE CHANGED BRANCH: both hedges available, halfway scored 0.940.
#    Before HANDOFF 53 this picked halfway; it must now pick parent.
CASES.append((
    "both hedges scored, halfway >= 0.940",
    [row("111", 0.941, CANON),
     row("222", 0.940, "ppxl16/rsna-knee-ens-halfway-routing"),
     row("333", 0.939, "ppxl16/rsna-knee-ens-parent-routing")],
    "333",
    "offence must stay canonical 111",
))

# 2. halfway at 0.941 - even a top-scoring half-hedge must not displace the
#    fully independent one, since the offence already carries the routing.
CASES.append((
    "halfway scores 0.941 (ties the offence)",
    [row("111", 0.941, CANON),
     row("222", 0.941, "ppxl16/rsna-knee-ens-halfway-routing"),
     row("333", 0.939, "ppxl16/rsna-knee-ens-parent-routing")],
    "333",
    "a half-hedge scoring high is still not independent",
))

# 3. parent absent -> halfway is the fallback, NOT a same-score variant.
CASES.append((
    "only halfway available",
    [row("111", 0.941, CANON),
     row("222", 0.940, "ppxl16/rsna-knee-ens-halfway-routing")],
    "222",
    "fallback branch must still fire",
))

# 4. neither routing hedge -> LAST RESORT same-score variant.
CASES.append((
    "no routing hedge at all",
    [row("111", 0.941, CANON),
     row("444", 0.941, "ppxl16/rsna-knee-ens-exclude-fracture")],
    "444",
    "last resort must still fire",
))

# 5. offence promotion: a variant >= PROMOTE_MARGIN above canonical wins.
CASES.append((
    "variant leads by exactly PROMOTE_MARGIN",
    [row("111", 0.941, CANON),
     row("555", 0.943, "ppxl16/rsna-knee-ens-some-variant"),
     row("333", 0.939, "ppxl16/rsna-knee-ens-parent-routing")],
    "333",
    "offence must promote to 555",
))

# 6. offence promotion withheld: a variant inside the margin must NOT win.
CASES.append((
    "variant leads by less than PROMOTE_MARGIN",
    [row("111", 0.941, CANON),
     row("555", 0.942, "ppxl16/rsna-knee-ens-some-variant"),
     row("333", 0.939, "ppxl16/rsna-knee-ens-parent-routing")],
    "333",
    "offence must stay canonical 111",
))


def main():
    failures = 0
    for name, rows, want_hedge, note in CASES:
        offence, hedge, out = picks(rows)
        ok = hedge == want_hedge
        failures += 0 if ok else 1
        print("%-4s %-44s hedge=%-6s want=%-6s  (%s)"
              % ("PASS" if ok else "FAIL", name, hedge, want_hedge, note))
        if not ok:
            print("     --- output was ---")
            for line in out.splitlines():
                print("     " + line)

    # 7. offence promotion, checked explicitly (case 5 wants offence 555).
    offence, _, _ = picks(CASES[4][1])
    ok = offence == "555"
    failures += 0 if ok else 1
    print("%-4s %-44s offence=%s want=555" % ("PASS" if ok else "FAIL",
                                              "promotion actually promotes", offence))
    offence, _, _ = picks(CASES[5][1])
    ok = offence == "111"
    failures += 0 if ok else 1
    print("%-4s %-44s offence=%s want=111" % ("PASS" if ok else "FAIL",
                                              "promotion correctly withheld", offence))

    # 8. empty history must not crash and must name nothing.
    _, hedge, out = picks([])
    ok = hedge is None and "none scored yet" in out
    failures += 0 if ok else 1
    print("%-4s %-44s hedge=%s" % ("PASS" if ok else "FAIL", "empty history", hedge))

    print("\n%d case(s) failed" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
