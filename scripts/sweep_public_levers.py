"""Sweep public notebooks for a lever we have not already classified.

WHY THIS IS THE RIGHT NEXT MOVE
-------------------------------
Every lever we can reach is closed by measurement, and the one knob with no coverage at all is the Rad
*weight* family (`rad_cal_w`, `rad_alpha`) -- selected on 58 gold labels, the least safe provenance,
and exactly what the frontier's own v38 changed (`Rad 88-feature calibration blend 0.40 to 0.00`).
A candidate exists but cannot be validated locally (A105), so before spending a submission slot on a
documented bet, check whether someone public has already tried it and said what happened.

WHAT IT LOOKS FOR
-----------------
Per notebook, its markdown, filtered for the terms that would carry a lever: calibration, alpha and
weight changes, exclusion lists, blend changes, and any stated public score.  The point is not to read
everything; it is to find the notebooks that talk about the knobs we care about.

⚠️ Rate discipline (FACTS A26): a burst of roughly a hundred API calls in an hour got the read paths
403-banned.  This pulls at most ~15 notebooks with a pause between them and reuses anything already on
disk, so it stays far below that.

Usage:
    python scripts/sweep_public_levers.py --top 16
"""
import argparse
import glob
import json
import os
import re
import sys
import time

# terms that would appear in prose about a lever
PAT = re.compile(
    r"rad_cal|rad_alpha|rad_second|calibrat|_RAD_EXCLUDE|exclude|blend|alpha|"
    r"weight|public score|0\.9[0-4][0-9]|fold|oof",
    re.I)
# where we have already pulled things, so we do not pay for them twice
LOCAL = ["downloads/research/frontier3/*/*.ipynb",
         "downloads/research/validation_survey/*/*.ipynb",
         "downloads/research/fresh0924/*/*.ipynb",
         "downloads/research/*/*.ipynb"]


def local_index():
    out = {}
    for pat in LOCAL:
        for p in glob.glob(pat):
            try:
                nb = json.loads(open(p, encoding="utf-8", errors="replace").read())
            except Exception:
                continue
            md = "\n".join("".join(c.get("source", []))
                           for c in nb.get("cells", []) if c.get("cell_type") == "markdown")
            out[os.path.basename(os.path.dirname(p))] = (p, md)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--pause", type=float, default=1.5)
    ap.add_argument("--competition", default="rsna-knee-abnormality-detection")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    ks = api.kernels_list(competition=args.competition, sort_by="scoreDescending",
                          page_size=args.top + 4)
    refs = [getattr(k, "ref", "") for k in ks][: args.top + 4]
    have = local_index()
    print("candidates: %d   already on disk: %d\n" % (len(refs), len(have)))

    tmp = os.path.join(os.environ.get("TEMP", "."), "leversweep")
    os.makedirs(tmp, exist_ok=True)
    seen = set()
    for ref in refs:
        if not ref or ref in seen:
            continue
        seen.add(ref)
        slug = ref.split("/")[-1]
        md = None
        if slug in have:
            md = have[slug][1]
            src = "local"
        else:
            d = os.path.join(tmp, slug)
            try:
                api.kernels_pull(ref, path=d, metadata=False, quiet=True)
            except Exception as exc:
                print("=== %-58s pull failed: %s" % (ref[:58], type(exc).__name__))
                continue
            nbs = glob.glob(os.path.join(d, "*.ipynb"))
            if not nbs:
                continue
            try:
                nb = json.loads(open(nbs[0], encoding="utf-8", errors="replace").read())
            except Exception:
                continue
            md = "\n".join("".join(c.get("source", []))
                           for c in nb.get("cells", []) if c.get("cell_type") == "markdown")
            src = "pulled"
            time.sleep(args.pause)
        if not md:
            continue
        hits = [l.strip() for l in md.split("\n") if len(l.strip()) > 30 and PAT.search(l)]
        # the terms we actually care about get their own pass
        hot = [l for l in hits if re.search(r"rad_cal|rad_alpha|rad_second|calibrat", l, re.I)]
        score = re.findall(r"0\.9[0-4][0-9]", md)
        print("=== %-58s [%s] md=%d lines, %d hits, %d calibration/alpha hits"
              % (ref[:58], src, len(md.split("\n")), len(hits), len(hot)))
        if score:
            print("    scores mentioned: %s" % sorted(set(score)))
        for l in hot[:6]:
            print("    HOT  " + l[:150])
        if not hot:
            for l in hits[:2]:
                print("    ..   " + l[:130])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
