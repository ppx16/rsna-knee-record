"""Sweep the RSNA public surface for anything that could become a NEW ENSEMBLE MEMBER.

WHY THIS AND NOT SOMETHING ELSE
    The +0.001 that escapes the 599-team 0.943 block needs a member that is **strong AND
    decorrelated**, and A167 fixed the hard gate: **solo >= 0.90**, because below that a member
    costs more than its diversity buys. Our own training ceiling is ~0.877, so nothing we train from
    scratch can pass; the only routes are (a) fine-tune a published strong checkpoint (the msft2
    line, queued on TPU) or (b) find a published checkpoint that already passes.

    (b) is cheap to check and has not been re-checked recently, and the competition runs to 10-22 so
    people keep publishing. This lists the newest and the most-voted notebooks AND datasets, so a
    newly published weight file shows up rather than being missed by a stale sweep.

WHAT IS REPORTED
    For notebooks: last run time, votes, and whether the slug is already in our local tree, so the
    output is "what is NEW", not "what exists".
    For datasets: last updated, votes, size -- a checkpoint usually arrives as a dataset.

Usage:
    python scripts/scan_public_members.py
"""
from __future__ import annotations

import pathlib
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = pathlib.Path(__file__).resolve().parent.parent
KNOWN_MARKERS = ("rsna", "knee")


def have(slug: str) -> bool:
    """Is this slug already pulled anywhere in the local tree?"""
    tail = slug.split("/")[-1].lower()
    for base in (ROOT / "kernels", ROOT / "datasets", ROOT / "downloads"):
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if tail[:24] and tail[:24] in p.name.lower():
                return True
    return False


def main() -> int:
    api = KaggleApi()
    api.authenticate()
    for label, kw in (("NOTEBOOKS newest", dict(search="rsna knee", sort_by="dateRun", page_size=60)),
                      ("NOTEBOOKS top-voted", dict(search="rsna knee", sort_by="voteCount", page_size=40))):
        try:
            ks = api.kernels_list(**kw)
        except Exception as e:
            print("!! %s failed: %s: %s" % (label, type(e).__name__, e))
            continue
        print("\n=== %s (%d) ===" % (label, len(ks)))
        for k in ks:
            ref = getattr(k, "ref", "") or ""
            print("  %-19s v%-4s %-58s %s" % (str(getattr(k, "last_run_time", ""))[:19],
                                              getattr(k, "total_votes", ""), ref[:58],
                                              "HAVE" if have(ref) else ""))
    print("\n=== DATASETS newest (rsna knee) ===")
    try:
        ds = api.dataset_list(search="rsna knee", sort_by="updated", page_size=60)
        for d in ds:
            ref = getattr(d, "ref", "") or ""
            print("  %-19s v%-4s %-56s %-10s %s" % (str(getattr(d, "last_updated", ""))[:19],
                                                    getattr(d, "total_votes", ""), ref[:56],
                                                    getattr(d, "total_bytes", ""),
                                                    "HAVE" if have(ref) else ""))
    except Exception as e:
        print("!! dataset_list failed: %s: %s" % (type(e).__name__, e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
