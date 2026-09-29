"""Read the EFFICIENCY track leaderboard -- the second track this project forgot it was in.

WHY THIS EXISTS (FACTS A164 / HANDOFF 221)
    The competition has a `$18,000` Efficiency Track (3 places, "accuracy-per-second") and the SAME
    submission counts for both tracks.  This repo recorded it on 2026-09-13 (HANDOFF 72.6), flagged
    it unverified, and never mentioned it again -- so the entire project optimised the main LB only.
    We sit at **efficiency rank 3343 / 4363** while our main-LB rank is 430 / 4479.

    ⚠️ AND WE CANNOT READ OUR SCORED RUNTIME.  The efficiency metric is computed on Kaggle's HIDDEN
    rerun (1322 studies); the log we can download is from the 3 visible studies and ends with
    `COMPLETE: submission.csv, 3 studies` (~273 s, mostly fixed overhead).  So the runtime input to
    the metric is not observable to us.

    ⇒ The board itself is therefore the instrument: it is the only signal that responds to runtime.
    Re-read it after changes and watch our EfficiencyRank and DateSubmitted.

WHAT `DateSubmitted` TELLS US, AND WHY IT MATTERS
    The board carries one row per team with a DateSubmitted.  Ours read `Sat Sep 26 05:24:23`,
    which matches submission `56570144` (0943f) -- NOT our then-latest.  So the board tracks ONE
    specific submission and we do not yet know WHICH rule picks it (best score? selected? last?).
    That question decides whether selecting a FASTER arm at the deadline can move our efficiency
    rank, so this script reports the date prominently; re-running it after new submissions resolves
    the rule empirically.

TRAPS ALREADY PAID FOR
    * Our team name on the board is **`p pxl16`** (with a space).  Searching for `ppxl16` finds
      NOTHING and it is easy to conclude "we are not on the board" -- that happened once.
    * The publisher notebook (`ryanholbrook/rsna-knee-abnormalities-efficiency-lb`) just copies
      Kaggle's own `leaderboard.csv`; the CSV is reachable through `kernels_output`, not by scraping
      the overview page (which is JS-rendered and returns an empty shell).
    * `kernels_output(file_pattern=...)` takes a REGEX, not a glob.

Usage:
    python scripts/eff_rank.py                 # fetch the board and report our row
    python scripts/eff_rank.py --csv PATH      # use a local copy
    python scripts/eff_rank.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import subprocess
import sys

# ⚠️ REQUIRED, not cosmetic. This script prints "⇒" (U+21D2) in its closing recommendation, and a
# Windows console defaults stdout to the LOCALE codec (GBK/cp936 on a Chinese Windows), so the run
# died with
#     UnicodeEncodeError: 'gbk' codec can't encode character '\u21d2'
# AFTER printing the standings and BEFORE printing what to do about them -- i.e. it silently ate the
# only actionable line. Third instance of this same class of bug in this workspace; the other two
# were in the ARC starter's slim_framework.py and build_notebook.py (both fixed with encoding="utf-8"
# on the file read/write side). Pipe the output anywhere and it is fine; it only breaks on a console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
PUBLISHER = "ryanholbrook/rsna-knee-abnormalities-efficiency-lb"
CACHE = ROOT / "downloads" / "research" / "lb"
OUR_TEAM = "p pxl16"
# the submission the board's DateSubmitted matched on 2026-09-28, for the "which rule?" question.
# ⚠️ The date string carries the YEAR ("Sat Sep 26 05:24:23 2026"), so a key without it silently
# fails to map -- which is what happened on the first run.
KNOWN = {"Sat Sep 26 05:24:23 2026": "56570144 (0943f) -- our then-best score, NOT our latest",
         "Sun Sep 27 06:07:33 2026": "56599932 (0943cf) -- a later joint-best",
         "Sun Sep 27 15:58:47 2026": "56613303 (0943cflat) -- the flattened-routing arm"}


def fetch():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    CACHE.mkdir(parents=True, exist_ok=True)
    fs, _ = api.kernels_output(PUBLISHER, path=str(CACHE),
                               file_pattern=r"full_leaderboard\.csv$", quiet=True)
    got = [pathlib.Path(x) for x in (fs or []) if x.endswith(".csv")]
    if not got:
        raise SystemExit("!! the publisher notebook returned no CSV")
    return got[0]


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not (r.get("PublicScore") or "").strip():
                continue
            rows.append({"rank": int(r["EfficiencyRank"]), "team": (r.get("TeamName") or "").strip(),
                         "score": float(r["PublicScore"]), "date": (r.get("DateSubmitted") or "").strip()})
    if not rows:
        raise SystemExit("!! no parseable rows")
    return sorted(rows, key=lambda r: r["rank"])


def ours(rows, team=OUR_TEAM):
    """Find OUR row.

    ⚠️ Match on the SQUEEZED-ZH-LOWER form.  The board writes `p pxl16`; searching for the literal
    'ppxl16' returns nothing and nearly produced the wrong conclusion 'we are not on the board'.
    """
    key = re.sub(r"\s+", "", team).lower()
    return next((r for r in rows if re.sub(r"\s+", "", r["team"]).lower() == key), None)


def report(rows):
    n = len(rows)
    me = ours(rows)
    print("efficiency board : %d teams" % n)
    if me is None:
        print("!! our row was NOT found -- check OUR_TEAM (%r) before concluding anything" % OUR_TEAM)
        return 1
    pct = 100.0 * me["rank"] / n
    print("US               : eff#%d / %d  (worse than %.1f%% of teams)  score %.3f" % (
        me["rank"], n, pct, me["score"]))
    print("DateSubmitted    : %s   (%s)" % (me["date"], KNOWN.get(me["date"], "unmapped -- "
                                              "this is how the tracking rule gets resolved")))
    # how the board relates to score -- the whole point is that it barely does
    same = [r for r in rows if abs(r["score"] - me["score"]) < 1e-9]
    print("teams at our score: %d, best efficiency rank among them: eff#%d"
          % (len(same), min(r["rank"] for r in same)))
    top = rows[0]
    print("eff#1            : %s (score %.3f)" % (top["team"], top["score"]))
    print("\n⇒ the rank to move is the EFFICIENCY one; the main LB is worth +0.001, this is worth "
          "thousands of places, and the same submission counts for both.")
    return 0


def score(auc, runtime_s, benchmark=0.5, max_auc=0.959):
    """The published efficiency score (lower is better).  VERIFIED from the competition page
    (2026-09-28, user-supplied screenshots of Overview -> Efficiency Prize Evaluation):

        Efficiency = AUC / (Benchmark - max AUC) + RuntimeSeconds / 32400

    `Benchmark` is the score of `sample_submission.csv`, and we MEASURED that file: every value is
    0.5, one distinct value, so a constant prediction scores AUC = 0.5 exactly.  That makes the
    denominator 0.5 - 0.959 = -0.459, i.e. NEGATIVE, so the first term is negative and grows in
    magnitude with AUC -- minimising the score wants a HIGHER AUC and a SHORTER runtime, as
    expected, and the exchange rate between them is what matters:

        1 s of runtime  = 1/32400 score      = 1.417e-5 AUC  (divide by 0.459)
        1 minute        = 0.00085 AUC
        1 hour          = 0.0510 AUC

    ⚠️ That exchange rate is the whole decision.  An earlier reading of this file claimed the track
    was "a runtime contest, not an accuracy contest" -- inferred from rank-vs-score alone, and WRONG
    in that form.  AUC enters linearly and cannot be bought back by speed alone, but the rate is
    brutal in the other direction too: at 1 hour = 0.051 AUC, **three hours of runtime are worth
    0.153 AUC -- nearly ten times the entire 0.016 gap between us and the leaderboard top.**
    So both terms matter and they are commensurable; neither may be waved away.
    """
    d = benchmark - max_auc                     # negative
    return auc / d + runtime_s / 32400.0


def whatif(our_auc, max_auc=0.959, benchmark=0.5):
    """The numbers that decide whether a speedup is worth engineering."""
    d = abs(benchmark - max_auc)
    per_hour_auc = 3600.0 / 32400.0 * d
    print("efficiency score  = AUC/(Benchmark-maxAUC) + Runtime/32400")
    print("Benchmark (sample_submission) = %.3f  [MEASURED: the file is a constant 0.5]" % benchmark)
    print("max AUC (private, unknown; public is %.3f)" % max_auc)
    print("=> the AUC term spans only %.4f of score across the ENTIRE AUC range 0..1"
          % (1.0 / d))
    print("=> exchange rate: 1 hour of runtime  ==  %.4f AUC" % per_hour_auc)
    print("                  1 minute           ==  %.5f AUC" % (per_hour_auc / 60))
    print()
    print("%-28s %s" % ("our AUC / runtime", "efficiency score (lower is better)"))
    for auc in (our_auc, our_auc + 0.001, 0.959):
        for rt in (600, 1800, 3600, 10800, 21600):
            print("   AUC %.3f  runtime %5.1f h   ->   %+.4f" % (auc, rt / 3600.0,
                                                                 score(auc, rt, benchmark, max_auc)))
    print()
    print("⚠️ our own runtime is STILL UNMEASURED -- the metric is computed on Kaggle's hidden")
    print("   rerun (1322 studies) and the log we can download only covers the 3 visible studies.")
    print("   ⇒ treat the runtime column as a scenario, not a reading.")


def selftest():
    rows = [{"rank": 1, "team": "A", "score": 0.958, "date": "x"},
            {"rank": 14, "team": "Stephen Schott", "score": 0.943, "date": "x"},
            {"rank": 3343, "team": "p pxl16", "score": 0.943, "date": "Sat Sep 26 05:24:23"},
            {"rank": 4000, "team": "Z", "score": 0.500, "date": "x"}]
    me = ours(rows)
    assert me is not None and me["rank"] == 3343, me
    assert ours([{"rank": 1, "team": "ppxl16", "score": 1.0, "date": "x"}]) is not None, \
        "the squeezed form must also match"
    assert ours(rows, team="nobody") is None
    same = [r for r in rows if abs(r["score"] - me["score"]) < 1e-9]
    assert min(r["rank"] for r in same) == 14, "a 0.943 team really does sit at eff#14"
    print("SELFTEST OK: finds our row through the SPACE in the name, and the 0.943/eff#14 case holds")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--whatif", type=float, default=None, metavar="OUR_AUC",
                    help="print the efficiency score for a grid of runtimes at this AUC")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.whatif is not None:
        whatif(a.whatif)
        return 0
    path = pathlib.Path(a.csv) if a.csv else fetch()
    print("board csv: %s" % path)
    return report(load(path))


if __name__ == "__main__":
    sys.exit(main())
