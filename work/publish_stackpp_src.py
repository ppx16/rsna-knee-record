"""Publish the extracted Stack++ reader source as OUR dataset, so the TTA arm can mount it.

WHY A DATASET RATHER THAN AN EMBEDDED STRING. The three files are ~15 KB of Python. Embedding them as
literals inside a notebook cell means escaping every quote and newline, and a single escaping slip produces
a cell that compiles here and fails on Kaggle -- the silent-failure class this repository keeps paying for.
Shipping them as a mounted dataset keeps them as actual files, byte-identical to what was extracted and
verified (`docs/EXTRACT-2026-10-06g-...`).

⚠️ Hosting on OUR account matters for the same reason it did for the distilled checkpoint: the exp085 line
died when a weights dataset it mounted was made private by its owner, and `kernels_push` **silently dropped
the source while reporting success**.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "work" / "stackpp_src"
STAGE = ROOT / "work" / "stackpp_dataset"
SLUG = "rsna-knee-stackpp-reader-src"
FILES = ("preprocess.py", "knee.py", "infer.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    missing = [f for f in FILES if not (SRC / f).is_file()]
    if missing:
        raise SystemExit("!! missing from %s: %s -- run work/extract_stackpp_src.py first" % (SRC, missing))

    STAGE.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copyfile(SRC / f, STAGE / f)
        print("staged %-16s %6d bytes" % (f, (SRC / f).stat().st_size))

    meta = {
        "title": "RSNA Knee Stack++ reader source",
        "id": "ppxl16/%s" % SLUG,
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            "The three inference source files of the Stack++ reader variant, extracted verbatim from "
            "romantamrazov/rsna-knee-dinosaur-v5's %%writefile cells by work/extract_stackpp_src.py. "
            "Re-hosted under our own account so a kernel can mount them without depending on a third "
            "party's notebook or dataset remaining available. Their changes relative to "
            "goodpjw2008/rsna-knee-2-5d-convnext-reader (which our reader leg currently runs) are: "
            "header-first DICOM ordering with pixels decoded only for usable slices; inference slice "
            "centres aligned to the training-bin centres; overlapping windows; a slice-grid TTA mode set; "
            "and fold x TTA aggregation in RANK space via _rank_members instead of a probability mean. "
            "Verified: all three parse under ast, define their required symbols, and contain ZERO "
            "references to the leaderboard."
        ),
    }
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote dataset-metadata.json -> %s" % meta["id"])

    if not a.push:
        print("staged only (pass --push)")
        return 0
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        r = api.dataset_create_new(str(STAGE), public=False, quiet=False, dir_mode="skip")
        print("dataset created:", getattr(r, "ref", None) or r)
    except Exception as e:  # noqa: BLE001
        print("create failed: %s: %s" % (type(e).__name__, str(e)[:400]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
