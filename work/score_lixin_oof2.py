"""Finish the lixin checkpoint check with the RIGHT instrument.

The gold-58 route gave 0.8454 on only **8** studies -- n=8 is noise, not a result, and reporting it as one
would be the error this repository has already paid for twice (gold-58's CI is +/-0.021 on all 58; on 8 it
is useless).

The OOF covers **882** held-out studies, and we hold a label table covering all 4407. That is the
convention `work/rank_all_members_one_convention.py` uses for every other member, so it is what this scores
against.

⚠️ EXCLUDE THE 58 GOLD ROWS FROM THE LABEL TABLE. `FIND-2026-10-05` established that
`report_labels_v5.csv` equals the gold truth elementwise on exactly those rows, so including them would
both flatter the member and make the number incomparable to the rest of the table.
"""
import glob
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

oof = pd.read_csv("work/lixin_ckpt/dinov2_seed17_fold0_oof.csv")
oof["StudyInstanceUID"] = oof["StudyInstanceUID"].astype(str)
print("OOF: %d studies" % len(oof))

g = sorted(glob.glob("downloads/**/raptor_gold_*.npz", recursive=True))[0]
gold_ids = {str(x) for x in np.load(g, allow_pickle=True)["ids"]}
print("gold-58 ids loaded; overlap with the OOF: %d" % len(gold_ids & set(oof["StudyInstanceUID"])))

v5 = pd.read_csv("downloads/labels/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv")
v5["StudyInstanceUID"] = v5["StudyInstanceUID"].astype(str)

m = oof.merge(v5, on="StudyInstanceUID", how="inner", suffixes=("_p", "_y"))
print("joined with the v5 label table: %d studies" % len(m))
before = len(m)
m = m[~m["StudyInstanceUID"].isin(gold_ids)]
print("after removing the 58 gold rows: %d (dropped %d)" % (len(m), before - len(m)))

pcols = [c for c in m.columns if c.endswith("_p") and c[:-2] in LAB]
ycols = [c for c in m.columns if c.endswith("_y") and c[:-2] in LAB]
print("paired columns: %d" % len(pcols))
if len(pcols) < 12:
    raise SystemExit("column pairing failed: %s" % pcols)

aucs = {}
for L in LAB:
    y = m[L + "_y"].to_numpy(dtype=float)
    p = m[L + "_p"].to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    yy = (y[ok] > 0.5).astype(int)
    if len(set(yy)) > 1:
        aucs[L] = float(roc_auc_score(yy, p[ok]))

print()
print("=== per-label AUC vs the v5 labels, n=%d ===" % len(m))
for L, v in sorted(aucs.items(), key=lambda kv: kv[1]):
    print("   %-20s %.4f" % (L, v))
print()
print("MACRO over %d labels: %.4f" % (len(aucs), float(np.mean(list(aucs.values())))))
print()
print("for comparison, the same convention (work/rank_all_members_one_convention.py):")
for name, v in (("surrogate_radfree", 0.8960), ("msft2", 0.8847), ("d64cn0", 0.8679),
                ("cn0", 0.8631), ("cx0", 0.8493), ("rejected yunus head (gold-58 route)", 0.7992)):
    print("   %-38s %.4f" % (name, v))
