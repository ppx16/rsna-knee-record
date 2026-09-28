"""Audit the READ-FIRST state files against LIVE sources -- mechanise the lesson, do not re-learn it.

WHY THIS EXISTS
    Three times in one week this project acted on a number that a cheap, authoritative source
    contradicted:

      * "the public ceiling is ~0.941" -- read off other competitors' discussion CLAIMS, while the
        leaderboard said 0.959 (FACTS A157);
      * our leaderboard rank printed as 219 -- DERIVED as 'teams above + 1' while the file already
        downloaded contained a Rank column saying 411 (A157);
      * a bare 400 from CreateCodeSubmission was read as "daily quota exhausted" -- the quota was
        fine, the call was missing `-f submission.csv` (A162).

    Each was a hand-maintained belief drifting away from a live source.  This script makes that
    class of drift LOUD: it extracts the claims the read-first files make and compares them with
    the ledger, the platform API and the leaderboard.

WHAT IT CHECKS
    1. ledger vs the submissions API  -- row count, and every scored row's value
    2. the best-score claim in STATUS.md vs the ledger
    3. the quota claims quoted in STATUS.md / docs/FINAL-SELECTION.md vs `kaggle quota`
    4. the submission-limit claim vs the live counter
    5. the leaderboard: OUR row by username, reading the Rank column (never deriving it)
    6. freshness of the auto-generated state/NOW.md
    7. (delegated) every FACTS citation and row structure, via check_facts_refs.py

NOT CHECKED, DELIBERATELY
    Prose, statuses written in words ("both QUEUED"), and historical docs.  A checker that tries to
    parse free text gets ignored; one that checks a short list of hard numbers gets used.

Usage:
    python scripts/audit_state.py            # report
    python scripts/audit_state.py --selftest # prove the mismatch detector fires
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATUS = ROOT / "STATUS.md"
FINAL = ROOT / "docs" / "FINAL-SELECTION.md"
LEDGER = ROOT / "state" / "lb_ground_truth.csv"
NOWMD = ROOT / "state" / "NOW.md"
PY = sys.executable
NOW_MAX_AGE_H = 48

# ⚠️ TWO ENVIRONMENTS, AND THEY ARE NOT INTERCHANGEABLE.
#   * `python scripts/*.py` runs under D:\python (numpy, sklearn, ...).
#   * the `kaggle` CLI **with the `quota` command** and the credentials lives in the venv.
# First version of this file used `sys.executable -m kaggle` for quota and therefore silently
# reported "could not read `kaggle quota`" -- a checker that cannot read its source is worse than
# no checker, because it looks like a pass.  Both paths are absolute and asserted below.
VENV = ROOT.parent / "venv-kaggle" / "Scripts"
KAGGLE_PY = VENV / "python.exe"
KAGGLE_EXE = VENV / "kaggle.exe"


def read(path):
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


class Findings:
    def __init__(self):
        self.rows = []

    def add(self, level, what, detail):
        self.rows.append((level, what, detail))

    def ok(self, what, detail=""):
        self.add("ok", what, detail)

    def warn(self, what, detail):
        self.add("WARN", what, detail)

    def bad(self, what, detail):
        self.add("STALE", what, detail)

    def report(self):
        n_bad = sum(1 for l, _, _ in self.rows if l == "STALE")
        for lv, what, detail in self.rows:
            print("%-6s %-42s %s" % (lv, what, detail))
        print("\n%d stale claim(s), %d warning(s)"
              % (n_bad, sum(1 for l, _, _ in self.rows if l == "WARN")))
        return 1 if n_bad else 0


# ------------------------------------------------------------------ individual checks
def ledger_rows():
    import csv
    if not LEDGER.is_file():
        return []
    with open(LEDGER, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def live_submissions():
    """ALL submissions, as dicts with 'ref' and 'score'.

    ⚠️ Do NOT use `api.competition_submissions()` for this: it defaults to `page_size=20` and does
    not auto-paginate, so it silently returned 20 of our 37 and this audit's first run reported a
    bogus 'ledger 37 vs API 20' mismatch.  `lb_ledger.submissions()` already reads the CLI's CSV,
    which returns everything -- reuse it rather than writing a second reader (the same reason
    member_sweep shells out instead of re-implementing the gate).
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import lb_ledger
    env = {**os.environ}
    if KAGGLE_EXE.is_file():
        env["KAGGLE_EXE"] = str(KAGGLE_EXE)
    old = os.environ.get("KAGGLE_EXE")
    os.environ["KAGGLE_EXE"] = env.get("KAGGLE_EXE", "kaggle")
    try:
        rows = lb_ledger.submissions()
    except Exception:
        return None
    finally:
        if old is None:
            os.environ.pop("KAGGLE_EXE", None)
        else:
            os.environ["KAGGLE_EXE"] = old
    out = []
    for r in rows:
        out.append({"ref": (r.get("ref") or "").strip(),
                    "score": (r.get("publicScore") or "").strip(),
                    "status": (r.get("status") or "").replace("SubmissionStatus.", "").strip()})
    return out


def check_ledger(f, api_rows):
    """The ledger is the authority for our scores; the live list is the authority for existence."""
    led = ledger_rows()
    f.ok("ledger rows", "%d rows" % len(led))
    if api_rows is None:
        f.warn("ledger vs live", "submission list unavailable -- skipped")
        return
    byref = {str(r.get("ref")): r for r in api_rows}
    if len(led) != len(api_rows):
        f.bad("ledger row count", "ledger %d vs live %d" % (len(led), len(api_rows)))
    else:
        f.ok("ledger row count matches the live list", str(len(led)))
    mism = []
    for r in led:
        a = byref.get(str(r.get("ref")))
        if not a:
            mism.append("%s not in the live list" % r.get("ref"))
            continue
        api_s, led_s = str(a.get("score") or ""), str(r.get("score") or "")
        if api_s and led_s:
            try:
                if abs(float(api_s) - float(led_s)) > 1e-9:
                    mism.append("%s ledger %s vs live %s" % (r.get("ref"), led_s, api_s))
            except ValueError:
                mism.append("%s unparseable (ledger %r live %r)" % (r.get("ref"), led_s, api_s))
    if mism:
        f.bad("ledger scores disagree with the live list", "; ".join(mism[:5]))
    else:
        f.ok("every ledger score matches the live list")


def check_best_score(f, api_rows):
    txt = read(STATUS)
    led = ledger_rows()
    best = None
    for r in led:
        try:
            v = float(r.get("score") or "")
        except ValueError:
            continue
        best = v if best is None else max(best, v)
    if best is None:
        f.warn("best-score claim", "no scored ledger rows")
        return
    m = re.search(r"Best ledger score.*?\*\*([01]\.\d{3})\*\*", txt, re.S)
    if not m:
        f.warn("best-score claim", "STATUS.md has no parseable best score")
        return
    if abs(float(m.group(1)) - best) > 1e-9:
        f.bad("STATUS best score", "claims %s, ledger says %.3f" % (m.group(1), best))
    else:
        f.ok("STATUS best score matches ledger", "%.3f" % best)


def quota_live():
    if not KAGGLE_PY.is_file():
        return {}
    r = subprocess.run([str(KAGGLE_PY), "-m", "kaggle", "quota"], capture_output=True, text=True,
                       cwd=str(ROOT))
    out = {}
    for line in ((r.stdout or "") + (r.stderr or "")).splitlines():
        m = re.match(r"\s*(GPU|TPU)\s+([\d.]+)h\s+([\d.]+)h\s+([\d.]+)h", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(4)))
    return out


def check_quota(f, live):
    if not live:
        f.warn("quota", "could not read `kaggle quota`")
        return
    for path, name in ((STATUS, "STATUS.md"), (FINAL, "FINAL-SELECTION.md")):
        txt = read(path)
        # ⚠️ NOT `GPU\s*...` -- the docs write "| GPU quota | **10.33 h / 30 h**", so there is
        # "quota | **" between the word GPU and the number.  My first regex missed it and the
        # selftest is what caught that (it asserted the regex matches the real phrasing).
        for m in re.finditer(r"GPU[^0-9\n]{0,24}([\d.]+)\s*h\s*/\s*30\s*h", txt):
            claimed = float(m.group(1))
            actual = live.get("GPU", (None, None))[0]
            if actual is None:
                continue
            if abs(claimed - actual) > 0.34:      # the counter moves in 20-min quanta (A121)
                f.bad("%s GPU quota" % name, "claims %.2f h, live %.2f h" % (claimed, actual))
            else:
                f.ok("%s GPU quota" % name, "%.2f h ~ live %.2f h" % (claimed, actual))
    for m in re.finditer(r"\*\*([\d.]+) h / 30 h\*\*", read(STATUS)):
        f.warn("STATUS quota phrasing", "found %s -- check it is the same number" % m.group(1))


def check_limits(f):
    if not KAGGLE_PY.is_file():
        f.warn("submission limits", "venv kaggle missing")
        return None
    r = subprocess.run([str(KAGGLE_PY), "-m", "kaggle", "competitions", "submission-limits",
                        "-c", "rsna-knee-abnormality-detection"],
                       capture_output=True, text=True, cwd=str(ROOT))
    got = {}
    for line in ((r.stdout or "") + (r.stderr or "")).splitlines():
        m = re.match(r"\s*(Submissions today|Lifetime submissions|Remaining today):\s*(\d+)", line)
        if m:
            got[m.group(1)] = int(m.group(2))
    if not got:
        f.warn("submission limits", "could not read the live counter")
        return None
    led = ledger_rows()
    if "Lifetime submissions" in got and len(led) != got["Lifetime submissions"]:
        f.warn("lifetime submissions", "ledger %d vs live %d" % (len(led), got["Lifetime submissions"]))
    else:
        f.ok("lifetime submissions matches the ledger", str(got.get("Lifetime submissions")))
    f.ok("slots remaining today", str(got.get("Remaining today")))
    return got


def check_leaderboard(f):
    """⚠️ The cached CSV must not be read as if it were live.

    First version read `public_leaderboard_latest.csv` unconditionally, so this audit reported
    rank 411 (yesterday's cache) while the live board said 430 -- i.e. the checker itself was
    carrying a remembered belief, which is the exact failure it exists to catch.  Refresh when the
    cache is older than LB_MAX_AGE_H, and always report the cache's age.
    """
    LB_MAX_AGE_H = 12
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import lb_ceiling
    except Exception as e:
        f.warn("leaderboard", "cannot import lb_ceiling (%s)" % type(e).__name__)
        return
    safe = ROOT / "downloads" / "research" / "lb" / "public_leaderboard_latest.csv"
    age_h = ((datetime.now(timezone.utc).timestamp() - safe.stat().st_mtime) / 3600.0
             if safe.is_file() else 1e9)
    rows = None
    if age_h > LB_MAX_AGE_H:
        try:
            rows = lb_ceiling.load_live()
            age_h = 0.0
        except Exception as e:
            f.warn("leaderboard refresh", "failed (%s); using a %.0f h old cache"
                   % (type(e).__name__, age_h))
    if rows is None:
        try:
            rows = lb_ceiling.load_csv(safe)
        except Exception as e:
            f.warn("leaderboard", "cannot read the cache (%s)" % type(e).__name__)
            return
    f.ok("leaderboard cache age", "%.1f h" % age_h)
    best, _ref = None, None
    for r in ledger_rows():
        try:
            v = float(r.get("score") or "")
        except ValueError:
            continue
        best = v if best is None else max(best, v)
    me = next((r for r in rows if "pxl16" in (r.get("team") or "").replace(" ", "").lower()), None)
    if me is None:
        f.warn("our leaderboard row", "our team not found in the board")
        return
    f.ok("our leaderboard row (Rank column, not derived)",
         "rank %s score %.3f" % (me["rank"], me["score"]))
    top = max(r["score"] for r in rows)
    gap = top - me["score"]
    f.ok("gap to board max", "%+.3f (%.1fx the +-0.008 noise band)" % (gap, gap / 0.008))
    # the block we sit in decides the medals -- report it, since it is the actionable number
    same = [r for r in rows if abs(r["score"] - me["score"]) < 1e-9]
    rk = [r["rank"] for r in same if r["rank"] is not None]
    if rk:
        f.ok("our score block", "%d teams at %.3f, ranks %d..%d (cutoffs inside it decide medals)"
             % (len(same), me["score"], min(rk), max(rk)))


def check_nowmd(f):
    """Freshness must be judged by CONTENT, not by mtime.

    ⚠️ The first version checked only the file's mtime, so a freshly regenerated NOW.md passed --
    while its body showed a GPU queue from 09-14 and a log from `director.py`, a component deleted
    in §164.  A timestamp proves the file was WRITTEN recently, not that what it says is current;
    that is the same "the gate does not look at the thing it certifies" trap this repo keeps hitting.
    """
    if not NOWMD.is_file():
        f.warn("state/NOW.md", "missing")
        return
    txt = NOWMD.read_text(encoding="utf-8", errors="replace")
    age_h = (datetime.now(timezone.utc).timestamp() - NOWMD.stat().st_mtime) / 3600.0
    if age_h > NOW_MAX_AGE_H:
        f.bad("state/NOW.md mtime", "generated %.0f h ago (limit %d h)" % (age_h, NOW_MAX_AGE_H))
    else:
        f.ok("state/NOW.md mtime", "%.0f h old" % age_h)
    # content freshness: the newest date the body mentions
    dates = re.findall(r"20\d\d-\d\d-\d\d", txt)
    if dates:
        newest = max(dates)
        try:
            nd = datetime.strptime(newest, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            behind = (datetime.now(timezone.utc) - nd).days
            if behind > 3:
                f.bad("state/NOW.md content", "newest date inside is %s (%d days behind)"
                      % (newest, behind))
            else:
                f.ok("state/NOW.md content", "newest date %s" % newest)
        except ValueError:
            f.warn("state/NOW.md content", "unparseable date %r" % newest)
    # and it must not present deleted machinery as live
    dead = [tok for tok in ("director.log", "cv_queue.py") if tok in txt]
    if dead:
        banner = "已过时" in txt or "obsolete" in txt.lower()
        if banner:
            f.ok("state/NOW.md dead-refs", "%s referenced but the file carries an obsolete banner"
                 % ", ".join(dead))
        else:
            f.bad("state/NOW.md dead-refs",
                  "references %s (deleted in 164) with NO obsolete banner" % ", ".join(dead))


def check_facts(f):
    r = subprocess.run([PY, str(ROOT / "scripts" / "check_facts_refs.py")],
                       capture_output=True, text=True, cwd=str(ROOT))
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()][-3:]
    if r.returncode != 0:
        f.bad("FACTS citations/rows", " | ".join(tail))
    else:
        f.ok("FACTS citations/rows", tail[-1][:70] if tail else "OK")


def selftest():
    """The detector must FIRE on a planted mismatch -- a checker that never fires is not a checker."""
    f = Findings()
    f.bad("planted", "this must be reported")
    assert sum(1 for l, _, _ in f.rows if l == "STALE") == 1
    assert f.report() == 1, "report() must return non-zero when something is stale"
    g = Findings()
    g.ok("clean", "nothing wrong")
    assert g.report() == 0, "report() must return zero when nothing is stale"
    # and the quota regex must actually match the phrasing the docs really use
    sample = "| GPU quota | **10.33 h / 30 h** used (**19.67 h left**)"
    m = re.search(r"GPU[^0-9\n]{0,24}([\d.]+)\s*h\s*/\s*30\s*h", sample)
    assert m and m.group(1) == "10.33", "quota regex does not match the real phrasing: %r" % sample
    print("SELFTEST OK: mismatch detector fires, clean report returns 0, quota regex matches the docs")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    from kaggle.api.kaggle_api_extended import KaggleApi
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print("WARN   kaggle auth failed (%s) -- live checks will be skipped" % type(e).__name__)
    f = Findings()
    check_ledger(f, live_submissions())
    check_best_score(f, None)
    check_quota(f, quota_live())
    check_limits(f)
    check_leaderboard(f)
    check_nowmd(f)
    check_facts(f)
    return f.report()


if __name__ == "__main__":
    sys.exit(main())
