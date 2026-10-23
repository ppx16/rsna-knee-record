"""Place the candidate member on OUR one-convention table, before wiring anything.

The model card claims 0.8251 on the **58-gold** set -- a set whose bootstrap CI is +/-0.021, so that
number cannot gate anything. This computes the same statistic on the full 4407-study OOF against the
label source the card says it trained on (`report_labels_v5.csv`), which is the convention
`work/rank_all_members_one_convention.py` uses for every other member we have.

Only after it sits on that table does it make sense to ask whether it should be wired in.
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
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

oof = pd.read_csv("work/models_axis/v52_oof_yunus.csv")
print("candidate OOF:", oof.shape, "| cols:", list(oof.columns)[:3], "...")
idc = "StudyInstanceUID"

gt = pd.read_csv("downloads/labels/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv")
print("v5 labels    :", gt.shape, "| cols:", list(gt.columns)[:4], "...")

if idc not in gt.columns:
    for c in gt.columns:
        if "uid" in c.lower() or "study" in c.lower():
            idc = c
            break
print("join key:", idc, "| present in both:", idc in oof.columns and idc in gt.columns)

m = oof.merge(gt, on=idc, how="inner", suffixes=("_p", "_y"))
print("joined studies:", len(m))
if len(m) == 0:
    raise SystemExit("no overlap -- check the key and the label columns")

ycols, pcols, missing = [], [], []
for L in LAB:
    yc = [c for c in m.columns if c == L + "_y"]
    pc = [c for c in m.columns if c == L + "_p"]
    if yc and pc:
        ycols.append(yc[0]); pcols.append(pc[0])
    else:
        missing.append(L)
print("labels matched: %d/12" % len(ycols), ("missing: %s" % missing) if missing else "")

aucs = {}
for L, yc, pc in zip([L for L in LAB], ycols, pcols):
    y = m[yc].to_numpy(dtype=float)
    p = m[pc].to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(set((y > 0.5).astype(int))) > 1:
        aucs[L] = float(roc_auc_score((y > 0.5).astype(int), p))

print()
print("=== per-label AUC (candidate vs the V5 labels we also train on) ===")
for L, v in sorted(aucs.items(), key=lambda kv: kv[1]):
    print("   %-20s %.4f" % (L, v))
print()
print("MACRO AUC over %d labels, n=%d studies: %.4f" % (len(aucs), len(m), float(np.mean(list(aucs.values())))))
print()
print("For comparison, the repo's one-convention table (work/rank_all_members_one_convention.py):")
for name, v in (("surrogate_radfree", 0.8960), ("msft2", 0.8847), ("d64cn0", 0.8679),
                ("cn0", 0.8631), ("d64nx3360", 0.8501), ("cx0", 0.8493)):
    print("   %-20s %.4f" % (name, v))
print()
print("A167 member gate is solo >= ~0.90. The card's own gold-58 claim was 0.8251.")
