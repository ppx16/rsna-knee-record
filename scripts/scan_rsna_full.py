"""Full RSNA surface sweep: notebooks, datasets, and the DISCUSSION forum.

WHY THE FORUM IS INCLUDED
    Every sweep so far has covered notebooks and datasets, and the discussion forum has never been
    read once -- yet for this competition it is where a `$77,000` recipe is most likely to be written
    down, and where a checkpoint's author explains the preprocessing we could not recover for exp085.
    A "public artifact" search that skips the forum is searching half the surface.

Usage:
    python work/scan_rsna_full.py > work/rsna_scan.txt 2>&1
"""
from __future__ import annotations

import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"


def notebooks(api):
    seen = set()
    for label, kw in (("newest", dict(sort_by="dateRun")), ("top-voted", dict(sort_by="voteCount"))):
        try:
            ks = api.kernels_list(search="rsna knee", page_size=40, **kw)
        except Exception as e:
            print("!! notebooks/%s: %s" % (label, type(e).__name__))
            continue
        fresh = [k for k in ks if (getattr(k, "ref", "") or "") not in seen]
        if not fresh:
            continue
        print("\n=== NOTEBOOKS %s (%d new) ===" % (label, len(fresh)))
        for k in fresh:
            seen.add(getattr(k, "ref", "") or "")
            print("  %-19s v%-5s %-56s %s" % (str(getattr(k, "last_run_time", ""))[:19],
                                              getattr(k, "total_votes", ""),
                                              (getattr(k, "ref", "") or "")[:56],
                                              (getattr(k, "title", "") or "")[:42]))


def datasets(api):
    try:
        ds = api.dataset_list(search="rsna knee", sort_by="updated")
    except Exception as e:
        print("!! datasets: %s" % type(e).__name__)
        return
    print("\n=== DATASETS newest (%d) ===" % len(ds))
    for d in ds[:30]:
        print("  %-19s v%-5s %-50s %-12s %s" % (str(getattr(d, "last_updated", ""))[:19],
                                                getattr(d, "total_votes", ""),
                                                (getattr(d, "ref", "") or "")[:50],
                                                getattr(d, "total_bytes", ""),
                                                (getattr(d, "title", "") or "")[:38]))


def discussions(api):
    """The forum. The classic Kaggle API has no documented method, so try every plausible one and
    report WHICH worked rather than silently printing nothing."""
    print("\n=== DISCUSSIONS ===")
    cands = [m for m in dir(api) if "discuss" in m.lower() or "forum" in m.lower() or "topic" in m.lower()]
    print("  candidate methods on the client: %s" % cands)
    try:
        from kagglesdk import KaggleClient  # noqa: F401
        import kagglesdk
        svc = [m for m in dir(kagglesdk) if "discuss" in m.lower()]
        print("  kagglesdk discussion modules: %s" % svc)
    except Exception as e:
        print("  kagglesdk probe failed: %s" % type(e).__name__)
    try:
        from kagglesdk.discussions.services.discussion_service import DiscussionServiceClient  # noqa
        print("  DiscussionServiceClient importable -- a forum read IS possible")
    except Exception as e:
        print("  DiscussionServiceClient not importable: %s: %s" % (type(e).__name__, str(e)[:120]))


def main() -> int:
    api = KaggleApi()
    api.authenticate()
    notebooks(api)
    datasets(api)
    discussions(api)
    return 0


if __name__ == "__main__":
    sys.exit(main())
