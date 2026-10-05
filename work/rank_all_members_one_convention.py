"""Put EVERY local member OOF on ONE convention, so members can be ranked against each other.

Why this matters before spending TPU: my A5-lite readings are 0.7053 (A5 head) and 0.6770
(our head), while A152 records `cn0@f0 = 0.8769` on ITS convention. If A-lite really sits ~0.17
below the members we already use, then the problem is not "6 slices vs 16" and building a 6x16
cache cannot close a 0.17 gap to the 0.90 member gate.

Truth is the report labels the arms were trained against; the convention is applied IDENTICALLY
to every arm, which is what makes the ranking meaningful even if the absolute level differs from
A152/A153's (they used a confidence mask, this does not).
"""
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
ROOT = pathlib.Path(r"D:\kaggle\rsna-knee")
TRUTH = ROOT / "downloads/labels/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv"

lab = pd.read_csv(TRUTH, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
have = [l for l in LABELS if l in lab.columns]
print(f"truth: {TRUTH.name}, {len(lab)} studies, {len(have)}/12 label columns\n")


def macro(pred, truth):
    per, used = {}, []
    for j, l in enumerate(LABELS):
        p, t = pred[:, j], truth[:, j]
        m = np.isfinite(p) & np.isfinite(t)
        if m.sum() < 10 or len(np.unique(t[m])) < 2:
            per[l] = None
            continue
        per[l] = float(roc_auc_score((t[m] > 0.5).astype(int), p[m]))
        used.append(l)
    return (sum(per[l] for l in used) / len(used) if used else float("nan")), len(used)


def score(path):
    z = np.load(path, allow_pickle=True)
    if "pred" not in z:
        return None
    pred = z["pred"].astype(np.float64)
    ids = np.asarray(z["ids"]).astype(str) if "ids" in z else None
    if ids is None or pred.ndim != 2 or pred.shape[0] != len(ids):
        return None
    T = np.full_like(pred, np.nan)
    for i, s in enumerate(ids):
        if s in lab.index:
            r = lab.loc[s]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            for j, l in enumerate(LABELS):
                if l in lab.columns:
                    try:
                        T[i, j] = float(r[l])
                    except (TypeError, ValueError):
                        pass
    m, ncol = macro(pred, T)
    return m, pred.shape[0], ncol


rows = []
for p in sorted(ROOT.rglob("*.npz")):
    if "gold" in p.name:
        continue
    try:
        r = score(p)
    except Exception as e:
        continue
    if r and np.isfinite(r[0]) and r[1] >= 100:
        rows.append((r[0], r[2], r[1], str(p.relative_to(ROOT))))

rows.sort(reverse=True)
print(f"{'macro':>7} {'cols':>4} {'n':>5}  file")
print("-" * 100)
for m, ncol, n, name in rows:
    print(f"{m:>7.4f} {ncol:>4} {n:>5}  {name}")
print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)
print("  A152 records cn0@f0 = 0.8769 and d3f0 = 0.8530 on the report-labels-with-MASK convention.")
print("  The ranking below uses NO mask, applied identically to every arm, so the ORDER is the")
print("  meaningful part: it shows whether the A-lite arms sit near the members we already use.")
