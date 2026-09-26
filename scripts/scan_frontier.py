"""Scan the competition's public notebooks for score, and harvest the frontier's own ledger.

Two questions, one tool:
  1. Has anyone published a notebook that scores at or above our target?  The CLI's table does not
     show scores at all, so it cannot answer this -- the sort is by score but the column is missing,
     which makes the listing actively misleading (the top row looks like "the best" with no evidence).
  2. What does the frontier's own change-log say?  `mattiaangeli/...`'s notebook carries
     `metadata.rsna_optimization`, a versioned record of every predictive change with a stated
     reason.  It is NOT in kernel-metadata.json and NOT exposed by the kernel API -- it only exists
     inside the pulled .ipynb -- so it has to be pulled and read.

Usage:
    python scripts/scan_frontier.py                        # score scan only
    python scripts/scan_frontier.py --ledger               # also pull + print the ledger
    python scripts/scan_frontier.py --top 30 --target 0.943
"""
import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

LEDGER_NOTEBOOKS = [
    # Where the v38 `rsna_optimization` entry was found on 2026-09-25 -- the score-improvement log.
    "mattiaangeli/bend-the-knee-to-the-dinosaurs",
    # The 0.943-class notebook we build our arms from.  On 2026-09-26 its `rsna_revision` chain ends
    # at `v559d`, the SAME sha as our 2026-09-24 snapshot -- i.e. this lineage had not advanced.
    "mattiaangeli/bend-the-knee-to-speedy-raptors-the-original",
    "maverickss26/rsna-knee-restructured-version-3",
]


def scan(api, competition, top, target):
    ks = api.kernels_list(competition=competition, sort_by="scoreDescending", page_size=top)
    print("%-56s %-8s %-6s %-6s %s" % ("ref", "score", "votes", "views", "lastRun"))
    print("-" * 110)
    above = []
    for k in ks:
        score = getattr(k, "score", None)
        if score in (None, "", "None"):
            shown = "-"
        else:
            try:
                shown = "%.4f" % float(score)
            except Exception:
                shown = str(score)
        ref = getattr(k, "ref", "?")
        print("%-56s %-8s %-6s %-6s %s" % (
            ref[:56], shown, getattr(k, "total_votes", ""),
            getattr(k, "total_views", ""), str(getattr(k, "last_run_time", ""))[:16]))
        try:
            if score not in (None, "", "None") and float(score) >= target:
                above.append((float(score), ref, getattr(k, "title", "")))
        except Exception:
            pass
    print()
    if above:
        print("!! AT OR ABOVE the %.3f target:" % target)
        for score, ref, title in sorted(above, reverse=True):
            print("   %.4f  %s   %s" % (score, ref, title))
    else:
        print("nothing public at or above %.3f in the top %d" % (target, top))
    return above


def harvest_ledger(api, refs):
    for ref in refs:
        print()
        print("=" * 78)
        print("LEDGER: %s" % ref)
        print("=" * 78)
        with tempfile.TemporaryDirectory() as td:
            try:
                api.kernels_pull(ref, path=td, metadata=False, quiet=True)
            except Exception as exc:
                print("  pull failed: %s: %s" % (type(exc).__name__, exc))
                continue
            nbs = list(Path(td).rglob("*.ipynb"))
            if not nbs:
                print("  no .ipynb in the pull")
                continue
            nb = json.loads(nbs[0].read_text(encoding="utf-8", errors="replace"))
            meta = nb.get("metadata") or {}
            # The frontier's change-log has been seen under BOTH names: `rsna_optimization` (the
            # 2026-09-14 vintage we first recorded as v38) and `rsna_revision` (2026-09-26).  Look for
            # every key starting with `rsna_` rather than hard-coding one, so a third rename does not
            # silently read as "no ledger" -- the failure mode that would quietly hide the frontier's
            # current lever from us.
            found = {k: v for k, v in meta.items() if k.startswith("rsna_")}
            if not found:
                print("  no rsna_* key in metadata; keys = %s" % sorted(meta.keys()))
                continue
            for key, val in found.items():
                print()
                print("  --- metadata.%s ---" % key)
                print(json.dumps(val, indent=2, ensure_ascii=False)[:8000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", default="rsna-knee-abnormality-detection")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--target", type=float, default=0.943)
    ap.add_argument("--ledger", action="store_true", help="also pull and print the frontier ledger")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    scan(api, args.competition, args.top, args.target)
    if args.ledger:
        harvest_ledger(api, LEDGER_NOTEBOOKS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
