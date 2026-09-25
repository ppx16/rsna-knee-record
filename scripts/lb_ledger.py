"""Regenerate state/lb_ground_truth.csv from the competition's own submission list.

WHY (found 2026-09-14)
    `state/lb_ground_truth.csv` is described by `cv_assemble.py` as "the recorded list of known
    LB differences" -- i.e. the project's LB ground truth and the ladder the CV is supposed to
    reproduce.  But NOTHING writes it: the only mentions in the whole tree are a comment in
    `cv_assemble.py` and a file-mtime line in `now.py`.  It has been maintained by hand.

    It was therefore STALE: mtime 2026-09-14 01:15 UTC, while the director had logged four more
    readings after that (pooloamax / poolefforig / poolfluidmax / arm-weights-alt, all 0.941),
    and three rows were still marked PENDING.  A stale ground-truth ledger is exactly the kind
    of thing a later round reads as "those probes never reported".

    The authoritative source is the competition's submission list, which `status.py` already
    uses.  So the ledger becomes GENERATED, like state/NOW.md -- it cannot go stale again.

Usage:
    python scripts/lb_ledger.py            # rewrite the ledger, print the diff summary
    python scripts/lb_ledger.py --show     # also print every row
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "state", "lb_ground_truth.csv")
COMPETITION = "rsna-knee-abnormality-detection"


def submissions():
    """The submission rows as dicts.  The CLI prints a version warning before the CSV, so the
    preamble is dropped by starting at the header line."""
    exe = os.environ.get("KAGGLE_EXE", "kaggle")
    p = subprocess.run([exe, "competitions", "submissions", "-c", COMPETITION, "-v"],
                       capture_output=True, text=True, timeout=600)
    lines = ((p.stdout or "") + (p.stderr or "")).splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("ref,")), None)
    if start is None:
        raise SystemExit("no CSV header in the submissions output:\n%s" % "\n".join(lines[:8]))
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args(argv)

    rows = submissions()
    # keep the historical column order so every existing reader keeps working
    out = []
    for r in rows:
        out.append({
            "date": (r.get("date") or "").strip(),
            "ref": (r.get("ref") or "").strip(),
            "desc": (r.get("description") or "").strip(),
            "score": (r.get("publicScore") or "").strip(),
            "status": (r.get("status") or "").replace("SubmissionStatus.", "").strip(),
        })
    out.sort(key=lambda r: r["date"])                       # oldest first, as before

    before = []
    if os.path.exists(LEDGER):
        with open(LEDGER, encoding="utf-8", newline="") as fh:
            before = list(csv.DictReader(fh))
    bmap = {r["ref"]: r for r in before}

    added = [r for r in out if r["ref"] not in bmap]
    changed = [r for r in out if r["ref"] in bmap
               and (bmap[r["ref"]].get("score") or "") != r["score"]]
    dropped = [r for r in before if r["ref"] not in {x["ref"] for x in out}]

    print("submissions from the API : %d" % len(out))
    print("ledger rows before       : %d" % len(before))
    print("new rows                 : %d %s" % (len(added), [r["ref"] for r in added]))
    print("score changed            : %d %s"
          % (len(changed), [(r["ref"], bmap[r["ref"]].get("score"), r["score"]) for r in changed]))
    if dropped:
        print("⚠️ rows in the ledger but NOT in the API: %s" % [r["ref"] for r in dropped])

    with open(LEDGER, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "ref", "desc", "score", "status"])
        w.writeheader()
        w.writerows(out)
    print("\nwrote %s (%d rows)" % (os.path.relpath(LEDGER, ROOT), len(out)))

    scored = [r for r in out if r["score"]]
    print("scored rows: %d   best: %s"
          % (len(scored), max(scored, key=lambda r: float(r["score"]))["score"] if scored else "-"))
    if a.show:
        print()
        for r in out:
            print("  %-10s %-6s %-9s %s" % (r["ref"], r["score"] or "-", r["status"],
                                             r["desc"][:52]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
