"""What score actually reaches the bronze line? Read the raw public board, not a remembered number.

WHY. The rank instrument is sub-resolution: the 0.943 block holds ~990 teams in one displayed value, so
"we need a bit more than 0.943" is not a plan. The board is ordered, so the question has an exact answer --
how many teams sit at each score immediately above us, and what the smallest score is whose LAST member is
still inside the top 10 %.

⚠️ The bronze line is a FRACTION of the field (top 10 %), so it moves as teams join. This prints the line it
used, and at what score the cut actually falls, so the conclusion can be re-derived rather than trusted.

Usage:
    python work/target_score.py
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
import csv
import io
import pathlib
import sys
import time
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
OUT = pathlib.Path("work/lb_live")


def main() -> int:
    api = KaggleApi()
    _last = None
    for _t in range(6):
        try:
            api.authenticate()
            break
        except Exception as e:  # noqa: BLE001
            _last = e
            time.sleep(8)
    else:
        print("COULD NOT AUTHENTICATE (%r) -- TRANSPORT, not a result." % (_last,), file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    api.competition_leaderboard_download(COMP, path=str(OUT), quiet=True)
    zp = OUT / (COMP + ".zip")
    if not zp.is_file():
        cands = list(OUT.glob("*.zip"))
        if not cands:
            print("no leaderboard zip in %s -- NOT a result" % OUT, file=sys.stderr)
            return 2
        zp = cands[-1]
    with zipfile.ZipFile(zp) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        raw = z.read(name).decode("utf-8-sig", errors="replace")
    rows = list(csv.DictReader(io.StringIO(raw)))
    print("board rows: %d   (%s)" % (len(rows), name))

    recs = []
    for r in rows:
        try:
            recs.append((int(r["Rank"]), float(r["Score"])))
        except Exception:  # noqa: BLE001
            continue
    recs.sort()
    n_teams = len(recs)
    bronze_line = max(1, int(round(n_teams * 0.10)))

    ours = [x for x in recs if x[1] == 0.943]
    our_rank = next((rk for rk, sc in recs if sc <= 0.943), recs[-1][0])

    print()
    print("teams              : %d" % n_teams)
    print("bronze line (top10%%): rank <= %d" % bronze_line)
    print("our score display  : 0.943")
    print("first rank at 0.943: %d" % (ours[0][0] if ours else -1))
    print("last  rank at 0.943: %d   (block width %d)" % (ours[-1][0] if ours else -1, len(ours)))

    # score band structure from the top of our block upward
    print()
    print("=== score bands from 0.960 down to 0.938 ===")
    print("  score    n     rank range        cum-teams-above")
    bands = {}
    for rk, sc in recs:
        bands.setdefault(sc, []).append(rk)
    above = 0
    for sc in sorted(bands, reverse=True):
        if sc > 0.960 or sc < 0.938:
            continue
        rks = bands[sc]
        print("  %.3f  %5d   %6d .. %-6d  %6d" % (sc, len(rks), rks[0], rks[-1], above))
        above += len(rks)

    # the actual answer: smallest displayed score whose last member is still inside the line
    print()
    print("=== what it takes ===")
    ok = None
    for sc in sorted(bands, reverse=True):
        if bands[sc][-1] <= bronze_line:
            ok = sc
    if ok is None:
        print("  no score band ends inside the line -- the cut falls mid-band")
    else:
        print("  the LOWEST score whose last member is still inside rank %d is %.3f" % (bronze_line, ok))
    # and the first band that is fully outside
    out_bands = sorted(sc for sc in bands if bands[sc][0] > bronze_line)
    if out_bands:
        worst_in = max((sc for sc in bands if bands[sc][0] <= bronze_line), default=None)
        print("  the highest score that is entirely OUTSIDE the line is %.3f" % max(out_bands))
        if worst_in is not None:
            print("  the lowest score that is entirely INSIDE  the line is %.3f" % min(
                sc for sc in bands if bands[sc][-1] <= bronze_line))
    print()
    print("  ⚠️ ranks, not scores, decide the medal. The displayed score has 0.001 resolution while the")
    print("     board resolves finer, so two teams on 0.943 can sit 990 ranks apart.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
