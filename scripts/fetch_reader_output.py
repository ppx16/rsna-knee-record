"""Fetch the reader's own output from the preview run and check it is a real, differentiated model.

`0.0000 of label cells moved` on a 3-study preview could be a mathematical necessity (re-ranking three
items always yields {1/3, 2/3, 1}) or could mean the reader produced near-copies of the stack. The only
way to tell is to look at the two tables. 12 labels all preserving order has probability ~(1/3)^12 if
the reader were random, so it needs explaining before a 7-hour submission is spent.
"""

# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded
# `print("... \u26a0 ...")` raises UnicodeEncodeError at that line and kills the script MID-REPORT, while
# everything printed earlier is already on screen -- so the run looks complete. Self-contained on purpose:
# it imports `sys` itself and depends on nothing else in this tree.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream: leave it alone, prints still degrade
    pass
# ----------------------------------------------------------------------------------------------------
import json
import pathlib
import time

import numpy as np
import pandas as pd

from kaggle.api.kaggle_api_extended import KaggleApi

REF = "ppxl16/rsna-knee-0943cfmsft2-rdr"
OUT = pathlib.Path("work/klog_rdr_out")
OUT.mkdir(parents=True, exist_ok=True)

api = KaggleApi()
api.authenticate()

names = ["_own.csv", "submission.csv", "_stack_submission.csv"]
for pat in names:
    for _ in range(3):
        try:
            res = api.kernels_output(REF, path=str(OUT), file_pattern=pat, force=True, quiet=True)
            print("pattern %-22s -> %s" % (pat, (len(res[0]), res[1][:50]) if isinstance(res, tuple) else res))
            break
        except Exception as e:  # noqa: BLE001
            print("pattern %-22s attempt failed: %s: %s" % (pat, type(e).__name__, str(e)[:90]))
            time.sleep(8)

files = sorted(p for p in OUT.rglob("*") if p.is_file())
print("\nfiles on disk: %d" % len(files))
for p in files:
    print("   %8d  %s" % (p.stat().st_size, p.relative_to(OUT)))

def load(name):
    h = sorted(OUT.rglob(name))
    if not h:
        return None
    try:
        return pd.read_csv(h[0], dtype={"StudyInstanceUID": str})
    except Exception as e:  # noqa: BLE001
        print("   read %s failed: %s" % (name, e))
        return None

own, sub = load("_own.csv"), load("submission.csv")
for nm, d in (("_own.csv (reader)", own), ("submission.csv (blended)", sub)):
    if d is None:
        print("\n%s: NOT AVAILABLE" % nm)
        continue
    lab = [c for c in d.columns if c != "StudyInstanceUID"]
    print("\n%s: shape=%s" % (nm, d.shape))
    print("   per-label nunique:", {c: int(d[c].nunique()) for c in lab})
    print("   per-label min/max:")
    for c in lab[:4]:
        print("      %-20s %.5f .. %.5f   %s" % (c, d[c].min(), d[c].max(),
                                                 np.round(d[c].to_numpy(), 4).tolist()))
if own is not None and sub is not None:
    lab = [c for c in sub.columns if c != "StudyInstanceUID"]
    m = own.merge(sub, on="StudyInstanceUID", suffixes=("_own", "_sub"))
    print("\n=== order comparison (rank of each study within each table) ===")
    for c in lab:
        ro = m[c + "_own"].rank().to_numpy()
        rs = m[c + "_sub"].rank().to_numpy()
        print("   %-20s reader ranks %s | stack ranks %s | same order: %s"
              % (c, ro.astype(int).tolist(), rs.astype(int).tolist(), np.array_equal(ro, rs)))
