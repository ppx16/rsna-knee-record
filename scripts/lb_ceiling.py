"""Read the live public leaderboard and report where WE actually stand.

WHY THIS EXISTS
---------------
On 2026-09-27 this project discovered that it had spent roughly a week of strategy on a false
premise: "the public ceiling is ~0.941-0.943, so we hold it."  That number came from A87.1, which
read the *text of other competitors' claims* ("highest declared scores were all [0.941 SOTA]") off a
discussion thread.  The live leaderboard was 0.959, with 218 teams measurably above us and 24 at or
above 0.955 -- a gap of 0.016, twice the +-0.008 noise band, i.e. a real difference and not a lottery.

The leaderboard is a free, authenticated, ONE-COMMAND instrument that had never been read.  A
quantity that is cheaply and authoritatively measurable must never be carried as a remembered
belief.  This script is that command, so the failure cannot recur silently.

RUN IT whenever the plan depends on "where the ceiling is" -- at minimum weekly (P1.4 of
docs/REVIEW-2026-09-27.md), and always before writing anything that says "we hold the ceiling".

    python scripts/lb_ceiling.py                # download the live board and report
    python scripts/lb_ceiling.py --csv F.csv    # use a local copy instead
    python scripts/lb_ceiling.py --selftest     # prove the parser and the banding

NOTES / TRAPS (both cost me time)
    * The downloaded zip contains a filename with COLONS in it
      (`...-publicleaderboard-2026-09-27T14:00:59.csv`), which Windows cannot extract.  So this reads
      the zip in memory via `zipfile` and writes a sanitised copy; do NOT use Expand-Archive.
    * `KaggleApi.dataset_list()` in kaggle 2.2.4 takes `search=`/`page=`, NOT `competition=`/`page_size=`.
    * The score column is `Score` and the team column is `TeamName`; ranks in the file are already
      ordered but overlap on ties, so this recomputes rank from the score.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMP = "rsna-knee-abnormality-detection"
CACHE = ROOT / "downloads" / "research" / "lb"
BANDS = [0.958, 0.955, 0.950, 0.948, 0.946, 0.945, 0.944, 0.943, 0.940]


def parse(rows):
    """rows -> list of dicts with float 'score' and int 'subs'; validates the schema loudly."""
    if not rows:
        raise SystemExit("!! empty leaderboard")
    cols = set(rows[0])
    for need in ("Score", "TeamName"):
        if need not in cols:
            raise SystemExit("!! leaderboard has no %r column (has %s)" % (need, sorted(cols)))
    out = []
    for r in rows:
        try:
            s = float(r["Score"])
        except (TypeError, ValueError):
            continue
        try:
            n = int(r.get("SubmissionCount") or 0)
        except (TypeError, ValueError):
            n = 0
        out.append({"score": s, "subs": n, "team": (r.get("TeamName") or "").strip()})
    if not out:
        raise SystemExit("!! no parseable scores")
    return out


def load_csv(path: pathlib.Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return parse(list(csv.DictReader(f)))


def load_live():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    CACHE.mkdir(parents=True, exist_ok=True)
    api.competition_leaderboard_download(COMP, path=str(CACHE))
    zips = sorted(CACHE.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not zips:
        raise SystemExit("!! no leaderboard zip landed in %s" % CACHE)
    z = zipfile.ZipFile(zips[-1])
    inner = z.namelist()[0]
    text = z.read(inner).decode("utf-8-sig")
    # keep a Windows-safe copy so later runs (and other tools) do not need the zip
    safe = CACHE / "public_leaderboard_latest.csv"
    safe.write_text(text, encoding="utf-8")
    print("leaderboard: %s  (%s)" % (inner, zips[-1].name))
    return parse(list(csv.DictReader(io.StringIO(text))))


def our_best():
    """Our best submitted score, from the ledger -- the authority is the submission list."""
    p = ROOT / "state" / "lb_ground_truth.csv"
    if not p.is_file():
        return None, None
    best, ref = None, None
    with open(p, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                s = float(r.get("score") or "")
            except ValueError:
                continue
            if best is None or s > best:
                best, ref = s, r.get("ref")
    return best, ref


def report(rows, best):
    rows = sorted(rows, key=lambda r: -r["score"])
    n = len(rows)
    top = rows[0]["score"]
    print("\nteams scored : %d" % n)
    print("board max    : %.3f  (%s)" % (top, rows[0]["team"][:34]))
    if best is None:
        print("our best     : unknown (no ledger)")
    else:
        above = sum(1 for r in rows if r["score"] > best)
        at = sum(1 for r in rows if abs(r["score"] - best) < 1e-9)
        print("our best     : %.3f   -> %d teams strictly above, %d teams on exactly that score"
              % (best, above, at))
        print("                rank ~%d / %d  (top %.1f%%)" % (above + 1, n, 100.0 * (above + 1) / n))
        gap = top - best
        print("                gap to the board max: %+.3f  = %.1fx the +-0.008 noise band"
              % (gap, gap / 0.008))
        if gap > 0.008:
            print("                ⚠️ REAL GAP -- larger than the noise band. We are NOT at the ceiling.")
        else:
            print("                inside the noise band -- the board cannot separate us from the top here.")
    print("\n%-8s %-7s %-8s %s" % ("score", "teams", "median", "cumulative at-or-above"))
    for b in BANDS:
        band = [r for r in rows if b <= r["score"] < b + 0.001]
        cum = sum(1 for r in rows if r["score"] >= b)
        med = sorted(r["subs"] for r in band)
        med = med[len(med) // 2] if med else 0
        print("%-8.3f %-7d %-8d %d" % (b, len(band), med, cum))
    print("\n(median = median SubmissionCount in that 0.001 band; high medians above us mean an "
          "unknown share of any gap is public-LB hill-climbing, not recipe)")


def selftest():
    """Prove the parser and the banding fire, on synthetic rows -- a gate that never fires is not a gate."""
    rows = [{"Score": "0.959", "TeamName": "a", "SubmissionCount": "73"},
            {"Score": "0.943", "TeamName": "b", "SubmissionCount": "15"},
            {"Score": "0.943", "TeamName": "c", "SubmissionCount": "9"},
            {"Score": "garbage", "TeamName": "d", "SubmissionCount": "1"}]
    out = parse(rows)
    assert len(out) == 3, "unparseable score must be dropped, got %d" % len(out)
    assert out[1]["subs"] == 15, out[1]
    try:
        parse([{"Nope": "1"}])
    except SystemExit:
        pass
    else:
        raise AssertionError("a leaderboard with no Score column must fail loudly")
    best = 0.943
    above = sum(1 for r in out if r["score"] > best)
    assert above == 1, above
    print("SELFTEST OK: parser drops junk, validates the schema, and counts teams above us")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--best", type=float, default=None, help="override our score")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    rows = load_csv(pathlib.Path(a.csv)) if a.csv else load_live()
    best = a.best
    if best is None:
        best, _ref = our_best()
    report(rows, best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
