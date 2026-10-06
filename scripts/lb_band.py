"""Analyse the public leaderboard band structure around our rank.

Purpose (HANDOFF 51): the public LB shows only 3 decimals, so ~200 teams
display the same score as us while occupying 200 distinct ranks.  The silver
line falls *inside* that band.  Final placement is decided on the PRIVATE
split, so the question "how safe is our silver?" is really the question
"how wide is our tie band, and where does the medal line cut it?".

This script answers the second half exactly, from a downloaded leaderboard
CSV.  It cannot answer the first half (public->private reordering is not
observable from our side) -- see HANDOFF 51.3 and 53.

Usage:
    python scripts/lb_band.py downloads/lb/<file>.zip
    python scripts/lb_band.py <path-to-leaderboard.csv>
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
import argparse
import csv
import io
import sys
import zipfile
from collections import OrderedDict

OUR_TEAM = "p pxl16"
MEDALS = OrderedDict([("gold", 17), ("silver", 188), ("bronze", 366)])
# "Displays the same 3-decimal score" means "true scores lie in a window this
# wide".  Used as the band's score extent when converting a true gain into
# band positions -- see the caveat printed at that step.
ROUNDING_WINDOW = 0.001


def load_rows(path):
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if len(names) != 1:
                raise SystemExit("expected exactly 1 member in zip, got %r" % names)
            raw = z.read(names[0])
            label = names[0]
    else:
        with open(path, "rb") as fh:
            raw = fh.read()
        label = path
    text = raw.decode("utf-8-sig", "replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    return rows, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    rows, label = load_rows(args.path)
    print("source : %s" % label)
    print("teams  : %d" % len(rows))

    for r in rows:
        r["_rank"] = int(r["Rank"])
        r["_score"] = float(r["Score"])
    ranks = [r["_rank"] for r in rows]
    if ranks != list(range(1, len(rows) + 1)):
        print("WARNING: Rank column is not a clean 1..N sequence")

    me = [r for r in rows if r["TeamName"].strip() == OUR_TEAM]
    if not me:
        me = [r for r in rows if "pxl16" in r["TeamName"]]
    if not me:
        raise SystemExit("could not find team %r in the leaderboard" % OUR_TEAM)
    me = me[0]
    my_rank, my_score = me["_rank"], me["_score"]
    print("\n=== US ===")
    print("  rank %d / %d   score %.3f   submissions %s"
          % (my_rank, len(rows), my_score, me.get("SubmissionCount", "?")))

    # ---- the tie band: every team displaying the same score as us ----
    band = [r for r in rows if r["_score"] == my_score]
    band_ranks = sorted(r["_rank"] for r in band)
    top, bot = band_ranks[0], band_ranks[-1]
    pos = band_ranks.index(my_rank) + 1
    print("\n=== TIE BAND (displayed score %.3f) ===" % my_score)
    print("  teams in band : %d" % len(band))
    print("  rank span     : %d .. %d  (width %d)" % (top, bot, bot - top + 1))
    print("  our position  : %d of %d  (%.0f%% from the top)"
          % (pos, len(band), 100.0 * (pos - 1) / max(1, len(band) - 1)))

    print("\n=== WHERE THE MEDAL LINES FALL RELATIVE TO THE BAND ===")
    for name, line in MEDALS.items():
        if line < top:
            verdict = ("band is entirely BELOW the %s line -- no team in this "
                       "band wins %s" % (name, name))
        elif line > bot:
            verdict = ("band is entirely INSIDE %s -- every team in this band "
                       "wins %s" % (name, name))
        else:
            above = sum(1 for rk in band_ranks if rk <= line)
            verdict = ("line cuts the band: %d/%d of the band is above it "
                       "(we are #%d of the band, so %s)"
                       % (above, len(band), pos,
                          "SAFE" if my_rank <= line else "BELOW"))
        print("  %-6s top %-4d -> %s" % (name, line, verdict))

    # ---- density: how the board is populated near us ----
    print("\n=== SCORE -> RANK DENSITY (top of the board) ===")
    scores = sorted({r["_score"] for r in rows}, reverse=True)
    print("  %-7s %-6s %-6s %s" % ("score", "first", "last", "teams"))
    for s in scores:
        if s < my_score - 0.004:
            break
        grp = [r["_rank"] for r in rows if r["_score"] == s]
        mark = "  <== us" if s == my_score else ""
        print("  %-7.3f %-6d %-6d %-5d%s"
              % (s, min(grp), max(grp), len(grp), mark))

    # ---- risk, stated honestly (HANDOFF 51) ----
    # Our position is not "rank 152"; it is "we are #pos of len(band) inside a
    # band that the medal line cuts at #above".  Everything about how safe we
    # are follows from those three numbers.
    print("\n=== RISK, STATED HONESTLY (HANDOFF 51) ===")
    for name, line in MEDALS.items():
        if not (top <= line <= bot):
            continue
        above = sum(1 for rk in band_ranks if rk <= line)
        p_random = 100.0 * above / len(band)
        print("  the %s line (top %d) cuts our band at #%d of %d"
              % (name, line, above, len(band)))
        print("    if the private split PRESERVES the band order : SAFE by %d places"
              % (above - pos))
        print("    if it RE-ORDERS the band at random            : P(stay inside) = %.0f%%"
              % p_random)
        print("    the truth is between those two, and is NOT observable from our side")

    # ---- what a small TRUE gain is worth, inside the band ----
    # WARNING: do not compute this from displayed-score thresholds.  Every team
    # here displays the same 3 decimals, so a threshold test cannot see any move
    # smaller than 0.001 and would report a misleadingly small number.
    print("\n=== VALUE OF A TRUE GAIN, MEASURED INSIDE THE BAND ===")
    dens = len(band) / ROUNDING_WINDOW
    print("  band holds %d teams inside a %.3f rounding window" % (len(band), ROUNDING_WINDOW))
    print("  => %.0f band positions per +1.000 of score (%.0f per +0.001)"
          % (dens, dens * ROUNDING_WINDOW))
    for delta in (0.0002, 0.0005, 0.001):
        print("    +%.4f  ->  ~%.0f band positions (capped at %d)"
              % (delta, min(delta * dens, len(band)), len(band)))
    print("  ASSUMPTION: band members are spread across the window.  If they are")
    print("  really clustered the value is LOWER (0 if all identical).  Upper bound.")

    print("\nNOTE: this is the PUBLIC board.  Final placement is decided on the")
    print("      PRIVATE split, which re-orders teams inside a tie band in a way")
    print("      we cannot observe (HANDOFF 51.3).  A gain measured on the public")
    print("      board can also be partly selection bias (HANDOFF 53.3).  Treat the")
    print("      band span, not our single rank, as the honest description of where")
    print("      we stand.")


if __name__ == "__main__":
    sys.exit(main())
