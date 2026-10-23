"""Lever 2's premise, tested on OUR data at zero GPU cost.

The claim from goodpjw2008's notebook (panel 3), measured against the 58 official labels:

    labels read from the reports      0.896
    their model's held-out predictions 0.912
    a 50/50 mix                        0.921

If that holds here too, then `report_labels_v5.csv` -- which EVERY member in this repository is trained
on -- is measurably worse than a trained model, and refining the labels is a real axis. If it does not
hold, it is a property of their pipeline and we drop it.

Method: treat a source's per-study label VALUES as scores and compute AUC against the gold BINARY truth,
per label, then macro. Same convention for the labels and for a model's predictions, so the comparison
is like-for-like.
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
import glob
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

gold_files = sorted(glob.glob("downloads/**/raptor_gold_*.npz", recursive=True)) + \
             sorted(glob.glob("work/**/raptor_gold_*.npz", recursive=True))
print("gold npz files found: %d" % len(gold_files))
for g in gold_files[:8]:
    print("   ", g)

if not gold_files:
    raise SystemExit("no raptor_gold_*.npz -- cannot get the 58 gold truth")

# any of them carries the same truth; take the first that loads
truth = ids = None
for g in gold_files:
    try:
        z = np.load(g, allow_pickle=True)
        if "truth" in z and "ids" in z:
            truth = z["truth"]
            ids = np.array([str(x) for x in z["ids"]])
            print("\ngold truth from %s: truth%s ids%s" % (g, truth.shape, ids.shape))
            break
    except Exception as e:  # noqa: BLE001
        print("   load failed %s: %s" % (g, str(e)[:80]))
if truth is None:
    raise SystemExit("could not load a gold truth matrix")

# gold may be soft; binarise for AUC
Ybin = (truth > 0.5).astype(int)
print("gold positives per label:", {LAB[i]: int(Ybin[:, i].sum()) for i in range(min(12, Ybin.shape[1]))})

def macro_auc(scores, name):
    out = {}
    for j, L in enumerate(LAB):
        if j >= scores.shape[1]:
            break
        y = Ybin[:, j]
        s = np.asarray(scores[:, j], dtype=float)
        ok = np.isfinite(s)
        if len(set(y[ok])) > 1:
            out[L] = float(roc_auc_score(y[ok], s[ok]))
    if out:
        print("\n%-42s macro over %2d labels = %.4f" % (name, len(out), float(np.mean(list(out.values())))))
        print("   " + "  ".join("%s %.3f" % (k, v) for k, v in sorted(out.items(), key=lambda kv: kv[1])))
    return out

# --- source 1: the report-derived label table every member is trained on -------------------------
v5path = "downloads/labels/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv"
v5 = pd.read_csv(v5path)
v5["StudyInstanceUID"] = v5["StudyInstanceUID"].astype(str)
g = pd.DataFrame({"StudyInstanceUID": ids})
# gold ids may not be in the v5 table's order; join on the uid
mg = g.merge(v5, on="StudyInstanceUID", how="left", suffixes=("", "_v5"))
print("\nv5 rows matched to gold: %d / %d" % (int(mg[LAB[0]].notna().sum()), len(g)))
macro_auc(mg[LAB].to_numpy(dtype=float), "report_labels_v5 (the labels we train on)")

# --- source 2: a trained member's own gold predictions -------------------------------------------
for g2 in gold_files:
    if "msft0" in g2 or "cn0" in g2 or "d64cn0" in g2:
        z = np.load(g2, allow_pickle=True)
        if "pred" in z and "truth" in z and np.array_equal(np.asarray(z["truth"]).shape, truth.shape):
            macro_auc(np.asarray(z["pred"], dtype=float), "member predictions: " + pathlib.Path(g2).stem)
            break
else:
    print("\n(no member gold-prediction npz matched for the model side)")
