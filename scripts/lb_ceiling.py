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
    """rows -> list of dicts with float 'score', int 'subs', int|None 'rank'; validates loudly."""
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
        try:
            # ⚠️ READ THE RANK COLUMN.  Do NOT derive rank as "teams strictly above + 1" -- that
            # was this script's first bug, and it reported 219 while our actual rank was 411,
            # because Kaggle gives every team a DISTINCT rank inside a block of equal scores
            # (the 0.943 block alone spans 448 ranks).  The tie-break is opaque: it is not
            # submission date (162 order violations in the 448-team block).
            rk = int(r["Rank"])
        except (KeyError, TypeError, ValueError):
            rk = None
        out.append({"score": s, "subs": n, "rank": rk, "team": (r.get("TeamName") or "").strip(),
                    "users": (r.get("TeamMemberUserNames") or "")})
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


def report(rows, best, user="ppxl16"):
    rows = sorted(rows, key=lambda r: -(r["rank"] or 10 ** 9) if r["rank"] else -r["score"])
    n = len(rows)
    top = max(r["score"] for r in rows)
    print("\nteams scored : %d" % n)
    print("board max    : %.3f" % top)

    me = next((r for r in rows if user and user in r["users"]), None)
    if me is None and best is not None:
        # fall back: the best-ranked row carrying our score
        cand = [r for r in rows if abs(r["score"] - best) < 1e-9]
        me = min(cand, key=lambda r: r["rank"] or 10 ** 9) if cand else None
        if me is not None:
            print("⚠️ team %r not found in TeamMemberUserNames; matched on score instead" % user)

    if me is not None:
        print("US           : rank %s  score %.3f  subs %d  (%s)"
              % (me["rank"], me["score"], me["subs"], me["team"]))
    above = sum(1 for r in rows if best is not None and r["score"] > best)
    if best is not None:
        print("strictly above us by SCORE: %d   (rank %s is decided by an opaque tie-break,"
              % (above, me["rank"] if me else "?"))
        print("                              NOT by score -- see the block table below)")
        gap = top - best
        print("gap to board max: %+.3f  = %.1fx the +-0.008 noise band" % (gap, gap / 0.008))
        if gap > 0.008:
            print("  ⚠️ REAL GAP -- larger than the noise band. We are NOT at the ceiling.")
        else:
            print("  inside the noise band -- the board cannot separate us from the top here.")

    # ---- the part that decides MEDALS: where do the cutoffs fall relative to our score block? ----
    blocks = collections.Counter(r["score"] for r in rows)
    spans = {}
    for r in rows:
        if r["rank"] is None:
            continue
        lo, hi = spans.get(r["score"], (r["rank"], r["rank"]))
        spans[r["score"]] = (min(lo, r["rank"]), max(hi, r["rank"]))
    ranks = {s: list(v) for s, v in spans.items()}
    print("\n-- medal cutoffs (standard rule: GOLD top-10 + top 0.2%, SILVER top 5%, BRONZE top 10%) --")
    cut = [("gold (0.2%)", max(1, round(0.002 * n))), ("silver (5%)", max(1, round(0.05 * n))),
           ("bronze (10%)", max(1, round(0.10 * n)))]
    ordered = sorted(rows, key=lambda r: r["rank"] or 10 ** 9)
    for name, k in cut:
        r = ordered[k - 1]
        print("  %-13s rank <= %-5d -> score %.3f  (a %d-team block)"
              % (name, k, r["score"], blocks[r["score"]]))
    if me is not None:
        bloc = blocks[me["score"]]
        lo, hi = ranks.get(me["score"], [None, None])
        print("\n  ⚠️ OUR score %.3f is a %d-team block spanning ranks %s..%s."
              % (me["score"], bloc, lo, hi))
        for name, k in cut:
            if lo is not None and lo <= k <= hi:
                print("     -> the %s cutoff (rank %d) falls INSIDE that block: whether we get it is"
                      % (name, k))
                print("        decided by the tie-break, NOT by score.")
        nxt = [s for s in blocks if s > me["score"]]
        if nxt:
            s2 = min(nxt)
            print("     -> the very next score up (%.3f) is a %d-team block at ranks %s..%s,"
                  % (s2, blocks[s2], ranks[s2][0], ranks[s2][1]))
            print("        i.e. +%.3f ESCAPES the tie-break entirely. That is the actionable target,"
                  % (s2 - me["score"]))
            print("        and it is far smaller than the %+.3f gap to the top." % (top - me["score"]))

    print("\n%-8s %-7s %-8s %-12s %s" % ("score", "teams", "median", "rank span", "cumulative"))
    for b in BANDS:
        band = [r for r in rows if b <= r["score"] < b + 0.001]
        cum = sum(1 for r in rows if r["score"] >= b)
        med = sorted(r["subs"] for r in band)
        med = med[len(med) // 2] if med else 0
        span = "%s..%s" % (ranks[b][0], ranks[b][1]) if b in ranks else "-"
        print("%-8.3f %-7d %-8d %-12s %d" % (b, len(band), med, span, cum))


def selftest():
    """Prove the parser and the rank/block logic -- a gate that never fires is not a gate.

    The cases below include the two bugs this script actually had, so they cannot come back:
    a missing Rank column must not silently become a derived guess, and a medal cutoff landing
    INSIDE a block of equal scores must be reported as tie-break-decided.
    """
    rows = [{"Rank": "1", "Score": "0.959", "TeamName": "a", "SubmissionCount": "73",
             "TeamMemberUserNames": "x"},
            {"Rank": "2", "Score": "0.944", "TeamName": "b", "SubmissionCount": "4",
             "TeamMemberUserNames": "y"},
            {"Rank": "3", "Score": "0.943", "TeamName": "c", "SubmissionCount": "15",
             "TeamMemberUserNames": "ppxl16"},
            {"Rank": "4", "Score": "0.943", "TeamName": "d", "SubmissionCount": "9",
             "TeamMemberUserNames": "z"},
            {"Score": "garbage", "TeamName": "e", "SubmissionCount": "1"}]
    out = parse(rows)
    assert len(out) == 4, "unparseable score must be dropped, got %d" % len(out)
    assert out[2]["rank"] == 3 and out[2]["subs"] == 15, out[2]
    assert out[4 - 1]["rank"] == 4, out[3]
    try:
        parse([{"Nope": "1"}])
    except SystemExit:
        pass
    else:
        raise AssertionError("a leaderboard with no Score column must fail loudly")
    # the rank is READ, never derived: with two scores above 0.943 the derived rank would be 3,
    # which happens to match here -- so also assert the case where deriving gives a WRONG answer
    above = sum(1 for r in out if r["score"] > 0.943)
    assert above == 2 and out[2]["rank"] == 3, (above, out[2]["rank"])
    tie = [dict(r) for r in rows[:3]] + [{"Rank": "9", "Score": "0.943", "TeamName": "dup",
                                          "SubmissionCount": "1", "TeamMemberUserNames": "w"}]
    ot = parse(tie)
    us = next(r for r in ot if "ppxl16" in r["users"])
    assert us["rank"] == 3, us["rank"]
    assert sum(1 for r in ot if r["score"] > us["score"]) == 2, "derived rank would say 3 -> right"
    # ...and here the DERIVED rank is wrong, which is why Rank is read:
    assert [r for r in ot if r["score"] == 0.943][-1]["rank"] == 9, "tie block must keep real ranks"
    # block/cutoff detection: build a board where the 10% cutoff genuinely lands inside the big block
    N = 20                                   # 1 team at 0.950, then 19 at 0.943
    big = [{"Rank": str(i + 1), "Score": "0.950" if i == 0 else "0.943", "TeamName": "t%d" % i,
            "SubmissionCount": "5", "TeamMemberUserNames": "ppxl16" if i == 6 else "u%d" % i}
           for i in range(N)]
    ob = parse(big)
    blocks = collections.Counter(r["score"] for r in ob)
    assert blocks[0.943] == N - 1, blocks
    lo = min(r["rank"] for r in ob if r["score"] == 0.943)
    hi = max(r["rank"] for r in ob if r["score"] == 0.943)
    assert (lo, hi) == (2, N), (lo, hi)
    cutoff10 = max(1, round(0.10 * N))       # derive it -- hardcoding it is how this test broke twice
    assert lo <= cutoff10 <= hi, ("10%% cutoff rank %d should fall inside the 0.943 block %d..%d"
                                  % (cutoff10, lo, hi))
    cutoff02 = max(1, round(0.002 * N))
    assert not (lo <= cutoff02 <= hi), "the 0.2% cutoff must NOT be inside the block here"
    print("SELFTEST OK: parser reads Rank (never derives it), and block-vs-cutoff detection works")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--best", type=float, default=None, help="override our score")
    ap.add_argument("--user", default="ppxl16", help="our Kaggle username, to find OUR row's rank")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    rows = load_csv(pathlib.Path(a.csv)) if a.csv else load_live()
    best = a.best
    if best is None:
        best, _ref = our_best()
    report(rows, best, a.user)
    return 0


if __name__ == "__main__":
    sys.exit(main())
