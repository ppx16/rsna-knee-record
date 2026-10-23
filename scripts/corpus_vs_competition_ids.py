"""Is `dreaddevelopment/knee-raptor-corpus` a preprocessed rendering of the COMPETITION's own train set?

If yes, then "more data" cannot mean the competition data -- we already consume all of it -- and any
headroom must come from genuinely external patients. If no, there is unused competition data sitting
right there, which would be a much cheaper axis. Either answer changes the plan, so check it rather
than infer it from the fact that both counts read 4407.
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
import csv
import numpy as np

train = [r["StudyInstanceUID"] for r in
         csv.DictReader(open("data/raw/train.csv", encoding="utf-8", errors="replace"))]
comp = set(train)
print("competition train.csv      : %d rows, %d unique" % (len(train), len(comp)))

all_ids = np.load(r"D:\kaggle\corpus\data\knee-raptor-corpus\all_ids.npy", allow_pickle=True)
extra = np.load(r"D:\kaggle\corpus\data\knee-raptor-corpus-ext\extra_ids.npy", allow_pickle=True)
A = {str(x) for x in all_ids.reshape(-1)}
E = {str(x) for x in extra.reshape(-1)}
print("corpus all_ids.npy         : %d" % len(A))
print("corpus extra_ids.npy       : %d" % len(E))
C = A | E
print("corpus total (union)       : %d" % len(C))
print("corpus internal overlap    : %d (all_ids n extra)" % len(A & E))
print()
print("competition NOT in corpus  : %d" % len(comp - C))
print("corpus NOT in competition  : %d" % len(C - comp))
print("identical sets             : %s" % (comp == C))
if comp != C:
    print("  examples comp-only:", sorted(comp - C)[:3])
    print("  examples corpus-only:", sorted(C - comp)[:3])
