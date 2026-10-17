"""Read prvsiyan's public-notebook scorecard and separate VERIFIED scores from CLAIMS.

Why this matters: the competition's public surface is full of titles like "0.951+", "0.954",
"≈955", "0.957 Candidate". We have repeatedly had to spend a GPU run + a submission slot to
find out what a notebook actually scores (the fitted-0.954 experiment cost exactly that and
returned 0.950). This dataset already did that work, and its `score_label` column says which
rows are real:
    V = verified (the displayed score was actually observed)
    P = the score is a TITLE CLAIM only

So this replaces "submit it and see" with "read the scorecard".

Source (downloaded into scratch/pscores/):
    kaggle datasets download -d prvsiyan/rsna-knee-2026-public-notebook-scores-and-assets

Usage: python work/scorecard.py [--all]
"""
from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "scratch" / "pscores" / "rsna-knee-2026-public-notebook-scores-and-assets.zip"


def rows() -> list[dict]:
    z = zipfile.ZipFile(ZIP)
    raw = z.read("public_notebooks.csv").decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(io.StringIO(raw)))


def label(r: dict) -> str:
    return (r.get("score_label") or "").strip().upper()


def main(argv: list[str]) -> int:
    if not ZIP.is_file():
        print(f"missing {ZIP} -- download the dataset first")
        return 2
    rs = rows()
    verified = [r for r in rs if label(r) == "V"]
    claims = [r for r in rs if label(r) == "P"]
    other = [r for r in rs if label(r) not in ("V", "P")]
    print(f"rows={len(rs)}  verified={len(verified)}  claims={len(claims)}  other={len(other)}")

    print("\n=== VERIFIED scores, descending -- the real public ceiling ===")
    print(f"{'score':>6}  {'OAI?':<16} {'ref':<50} weight_origin")
    print("-" * 150)
    for r in sorted(verified, key=lambda x: -float(x.get("displayed_score") or 0)):
        oai = str(r.get("uses_oai_derived_checkpoint"))[:15]
        print(f"{str(r.get('displayed_score')):>6}  {oai:<16} {r.get('notebook_ref','')[:50]:<50} "
              f"{str(r.get('weight_origin'))[:44]}")

    print("\n=== CLAIMS only -- titles, NOT achievements ===")
    for r in sorted(claims, key=lambda x: -float(x.get("displayed_score") or 0)):
        print(f"{str(r.get('displayed_score')):>6}  {r.get('notebook_ref','')[:64]}")
        n = str(r.get("notes") or "")
        if n:
            print(f"          note: {n[:150]}")

    if "--all" in argv:
        print("\n=== every row ===")
        for r in sorted(rs, key=lambda x: -float(x.get("displayed_score") or 0)):
            print(f"{str(r.get('displayed_score')):>6} [{label(r) or '-'}] "
                  f"{r.get('notebook_ref','')[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
