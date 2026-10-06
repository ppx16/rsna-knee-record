"""Find a label table that is NOT contaminated by the 58 official gold labels, then run Lever 2's test.

`report_labels_v5.csv` equals the gold truth elementwise on all 58 studies (696/696 cells), so it can
never measure "how good are report-derived labels" -- it IS the gold there. That is why the first
attempt returned AUC = 1.0000: it compared gold with gold.

This scans every local label table for the same contamination, then scores the clean ones against the
gold exactly as goodpjw2008's panel 3 does. If a clean report-only table scores materially below a
trained member's predictions (~0.897), their finding reproduces here and the label-refinement axis is
real for us too.
"""
import glob
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

z = np.load(sorted(glob.glob("downloads/**/raptor_gold_*.npz", recursive=True))[0], allow_pickle=True)
truth = np.asarray(z["truth"], dtype=float)
ids = [str(x) for x in z["ids"]]

CANDIDATES = [
    "downloads/labels/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv",
    "downloads/labels/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv",
    "downloads/labels/rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv",
    "outputs/report_labels_consensus.csv",
    "outputs/report_labels_severity.csv",
    "outputs/report_pseudo_labels.csv",
    "downloads/research/oof/report_labels_v2.csv",
]
CANDIDATES += sorted(glob.glob("downloads/**/report_labels*.csv", recursive=True))[:12]

seen, rows = set(), []
for p in CANDIDATES:
    if p in seen or not pathlib.Path(p).is_file():
        continue
    seen.add(p)
    try:
        df = pd.read_csv(p)
    except Exception as e:  # noqa: BLE001
        print("  SKIP %s (%s)" % (p, str(e)[:60]))
        continue
    key = next((c for c in df.columns if "uid" in c.lower()), None)
    if key is None or not set(LAB).issubset(df.columns):
        print("  SKIP %s (cols: %s)" % (p, list(df.columns)[:6]))
        continue
    df[key] = df[key].astype(str)
    m = pd.DataFrame({"StudyInstanceUID": ids}).merge(df, on=key, how="left")
    V = m[LAB].to_numpy(dtype=float)
    if np.isnan(V).any():
        rows.append((p, "missing rows", np.nan, np.nan))
        continue
    contam = np.array_equal(truth, V)
    aucs = {}
    for j, L in enumerate(LAB):
        y = (truth[:, j] > 0.5).astype(int)
        s = V[:, j]
        if len(set(y)) > 1:
            aucs[L] = float(roc_auc_score(y, s))
    macro = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    rows.append((p, "YES" if contam else "no", macro, np.unique(np.round(V, 3)).size))

print("\n%-70s %-11s %-8s %s" % ("label table", "= gold(58)?", "macro", "distinct values"))
for p, c, macro, nd in rows:
    print("%-70s %-11s %-8s %s" % (p[-70:], c, ("%.4f" % macro) if macro == macro else "-", nd))

print("\nmember predictions for comparison (from raptor_gold_msft0.npz):")
g0 = [g for g in glob.glob("downloads/**/raptor_gold_msft0*.npz", recursive=True)]
if g0:
    zz = np.load(g0[0], allow_pickle=True)
    P = np.asarray(zz["pred"], dtype=float)
    a = {}
    for j, L in enumerate(LAB):
        y = (truth[:, j] > 0.5).astype(int)
        if len(set(y)) > 1:
            a[L] = float(roc_auc_score(y, P[:, j]))
    print("   %-68s %.4f" % (pathlib.Path(g0[0]).stem, float(np.mean(list(a.values())))))
else:
    print("   (no msft0 gold npz)")
