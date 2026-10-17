"""Verify a pushed kernel's OUTPUT FILES before we spend a submission on it.

Motivation: a code-competition kernel can reach COMPLETE while a leg silently produced
degenerate or misaligned predictions. Ranking inside the 0.950 block is decided by the
UNROUNDED score (C113), so a half-broken blend is not a neutral mistake -- it wastes one
of the five daily submission slots and 10-11 h of latency.

For each expected CSV this checks:
  * it exists and is non-empty
  * 12 findings (+ StudyInstanceUID), no NaN, no constant column
  * values inside [0, 1]
  * StudyInstanceUID set/order matches submission.csv
  * the legs are not near-duplicates of each other (Spearman), i.e. the blend is real

Usage:
    python work/verify_run_outputs.py <kernel-slug> [expected.csv ...]
"""
from __future__ import annotations

import glob
import io
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch" / "kout"
TRANSPORT = ("Max retries", "SSLError", "UNEXPECTED_EOF", "Timeout", "timed out",
             "Connection aborted", "ConnectionReset", "Read timed out")

# Default expectations per kernel (files written into /kaggle/working).
EXPECT = {
    "ppxl16/rsna-knee-blendgold-restored": ["raptor_raw.csv", "ours_sub.csv"],
    "ppxl16/rsna-knee-fitted-0954": ["_c224.csv", "_sota_0949.csv", "_own.csv"],
    "ppxl16/rsna-knee-triple-reader-0951": ["_c224.csv", "_sota_0949.csv", "_own.csv"],
    "ppxl16/rsna-knee-apex-0949": ["_sota_0949.csv", "_own.csv"],
}


def pull_outputs(slug: str) -> Path | None:
    d = OUT / slug.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    for a in range(4):
        r = subprocess.run(["kaggle", "kernels", "output", slug, "-p", str(d)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=1800)
        o = (r.stdout or "") + (r.stderr or "")
        if r.returncode == 0 and not any(t in o for t in TRANSPORT):
            return d
        time.sleep(15 * (a + 1))
    print("!! could not download outputs (transport) -- NOT a verdict")
    return None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    slug = argv[1]
    wanted = argv[2:] or EXPECT.get(slug, [])
    d = pull_outputs(slug)
    if d is None:
        return 1

    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        print("pandas/numpy unavailable")
        return 1

    files = {Path(p).name: Path(p) for p in glob.glob(str(d / "**" / "*.csv"), recursive=True)}
    print(f"outputs in {d.name}: {sorted(files)}")

    sub = files.get("submission.csv")
    if sub is None:
        print("\nFAIL: no submission.csv produced -- the run did NOT deliver a submission")
        logs = list(glob.glob(str(d / "*.log")))
        for lg in logs:
            print(f"  (log present: {Path(lg).name}; inspect it for the traceback)")
        return 1

    ref = pd.read_csv(sub, dtype={"StudyInstanceUID": str})
    labels = [c for c in ref.columns if c != "StudyInstanceUID"]
    print(f"\nsubmission.csv: {len(ref)} rows x {len(labels)} findings")
    ok = True
    if len(labels) != 12:
        print(f"FAIL: expected 12 findings, got {len(labels)}")
        ok = False
    if ref[labels].isna().any().any():
        print("FAIL: NaN in submission.csv")
        ok = False
    const = [c for c in labels if ref[c].nunique() <= 1]
    if const:
        print(f"FAIL: constant column(s) in submission.csv: {const}")
        ok = False
    if not ((ref[labels].values >= 0) & (ref[labels].values <= 1)).all():
        print("FAIL: submission.csv has values outside [0,1]")
        ok = False

    legs: dict[str, "pd.DataFrame"] = {}
    print("\n--- legs ---")
    for name in wanted:
        p = files.get(name)
        if p is None:
            print(f"  MISSING {name}  <== a leg did not produce output")
            ok = False
            continue
        df = pd.read_csv(p, dtype={"StudyInstanceUID": str})
        n_missing = [c for c in labels if c not in df.columns]
        aligned = (len(df) == len(ref)) and df["StudyInstanceUID"].tolist() == ref["StudyInstanceUID"].tolist()
        nan = bool(df[[c for c in labels if c in df.columns]].isna().any().any()) if not n_missing else True
        cst = [c for c in labels if c in df.columns and df[c].nunique() <= 1]
        rng_ok = bool(((df[[c for c in labels if c in df.columns]].values >= 0) &
                       (df[[c for c in labels if c in df.columns]].values <= 1)).all()) if not n_missing else False
        print(f"  {name:16s} rows={len(df):5d} missing_cols={n_missing} aligned={aligned} "
              f"nan={nan} constant={len(cst)} in[0,1]={rng_ok}")
        if n_missing or not aligned or nan or cst or not rng_ok:
            ok = False
        if not n_missing:
            legs[name] = df[labels]

    if len(legs) >= 2:
        print("\n--- leg diversity (mean Spearman over the 12 findings) ---")
        names = list(legs)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                vals = [legs[a][c].corr(legs[b][c], method="spearman") for c in labels]
                m = float(np.nanmean(vals))
                flag = "  <== near-duplicate legs (blend adds little)" if m > 0.95 else ""
                print(f"  {a} vs {b}: {m:.3f}{flag}")

    print("\nVERDICT:", "PASS -- safe to submit" if ok else "FAIL -- do NOT submit this run")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
