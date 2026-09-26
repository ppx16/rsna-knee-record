"""Wait for specific competition submissions to be scored, then print them.

Why a watcher and not a loop of manual checks: the hidden-set rerun takes tens of minutes, and the
submission list is a shared, rate-limited endpoint (FACTS A26).  Polling every 5 minutes is cheap and
the run ends on its own; polling by hand is neither.

The list is read through the API rather than by scraping the CLI's table, because the table's
description column contains spaces and its trailing score columns are empty while a row is PENDING --
i.e. exactly the two shapes a naive split gets wrong.

Usage:
    python scripts/watch_scores.py --ref 56568520 --ref 56568524
"""
import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", default="rsna-knee-abnormality-detection")
    ap.add_argument("--ref", action="append", required=True,
                    help="submission ref to wait for; repeat for several")
    ap.add_argument("--interval-min", type=float, default=5.0)
    ap.add_argument("--max-hours", type=float, default=6.0)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    wanted = list(dict.fromkeys(args.ref))
    pending = set(wanted)
    deadline = time.time() + args.max_hours * 3600

    while True:
        try:
            subs = api.competition_submissions(args.competition)
        except Exception as exc:
            print("!! list failed (%s: %s); retrying next cycle" % (type(exc).__name__, exc))
            subs = []

        by_ref = {}
        for s in subs:
            by_ref[str(getattr(s, "ref", ""))] = s

        for ref in wanted:
            s = by_ref.get(ref)
            if s is None:
                print("%s  not visible in the list yet" % ref)
                continue
            status = str(getattr(s, "status", "?"))
            pub = getattr(s, "public_score", None)
            prv = getattr(s, "private_score", None)
            printed = getattr(s, "date", "")
            print("%s  status=%-28s public=%-8s private=%-8s  %s"
                  % (ref, status, pub if pub not in (None, "") else "-",
                     prv if prv not in (None, "") else "-", printed))
            scored = status.endswith("COMPLETE") and pub not in (None, "", "None")
            if scored:
                pending.discard(ref)
            elif status.endswith("ERROR"):
                print("%s  !! submission ERRORED - this one is dead" % ref)
                pending.discard(ref)

        if not pending:
            print("ALL DONE")
            return 0
        if time.time() > deadline:
            print("timeout; still pending: %s" % sorted(pending))
            return 2
        print("-- waiting %s min; still pending: %s"
              % (args.interval_min, sorted(pending)))
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    raise SystemExit(main())
