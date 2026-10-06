"""Finish the A153 experiment: A5 head vs our raptor head, BOTH with --aug 1.

A153 measured +0.0575 for the frontier's A5 head over ours at a DEGRADED operating point
(no augmentation, bs=2, n_slice=6) and explicitly required the transfer to be re-measured with
the real trainer's augmentation before any further investment:

    "give BOTH arms the SAME augmentation and fight again. If the advantage survives (even down
     to +0.02) the cache is worth building; if it collapses, A ends here."

Both runs are COMPLETE and their 881-study OOFs exist; only the comparison was never computed.
Truth = the report labels the arm itself was trained against (flight0234/rsna-knee-hybrid-report-labels).
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

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
D = pathlib.Path(r"D:\kaggle\rsna-knee\work\a5pair")
TRUTH = pathlib.Path(
    r"D:\kaggle\rsna-knee\downloads\labels\rsna-knee-hybrid-report-labels\report_labels_v4hybrid.csv")

lab = pd.read_csv(TRUTH, dtype={"StudyInstanceUID": str})
print("truth file:", TRUTH.name, lab.shape)
print("columns:", list(lab.columns)[:20])
uid_col = "StudyInstanceUID" if "StudyInstanceUID" in lab.columns else lab.columns[0]
print("uid column:", uid_col)


def value_for(row, label):
    """Labels may carry __conf / __verdict sidecars; use the value column and a mask."""
    for cand in (label, f"{label}__value", f"{label}__label"):
        if cand in lab.columns:
            try:
                return float(row[cand])
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def usable(label):
    for cand in (label, f"{label}__value", f"{label}__label"):
        if cand in lab.columns:
            return cand
    return None


have = {l: usable(l) for l in LABELS}
print("label columns found:", {k: v for k, v in have.items() if v})
missing = [k for k, v in have.items() if not v]
if missing:
    print("!! NO column for:", missing)

lab = lab.set_index(uid_col)


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
    return (sum(per[l] for l in used) / len(used) if used else float("nan")), per, len(used)


res = {}
for tag in ("a5fa", "rfa"):
    z = np.load(D / f"raptor_oof_{tag}_fold0.npz", allow_pickle=True)
    ids = np.asarray(z["ids"]).astype(str)
    pred = z["pred"].astype(np.float64)
    T = np.full_like(pred, np.nan)
    kept = 0
    for i, s in enumerate(ids):
        if s in lab.index:
            r = lab.loc[s]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            for j, l in enumerate(LABELS):
                if have[l]:
                    try:
                        T[i, j] = float(r[have[l]])
                    except (TypeError, ValueError):
                        pass
            kept += 1
    m, per, ncol = macro(pred, T)
    res[tag] = {"macro": m, "per": per, "ids": ids, "pred": pred, "truth": T}
    print(f"\n{tag}: {len(ids)} OOF studies, {kept} matched in the label table, "
          f"{ncol}/12 usable columns -> macro ROC-AUC = {m:.4f}")

if len(res) == 2:
    a, b = res["a5fa"], res["rfa"]
    print()
    print("=" * 76)
    print("A153's MISSING EXPERIMENT -- A5 head vs raptor head, BOTH with --aug 1")
    print("=" * 76)
    print(f"  A5 head     (a5fa) = {a['macro']:.4f}")
    print(f"  raptor head (rfa)  = {b['macro']:.4f}")
    print(f"  delta (A5 - raptor) = {a['macro'] - b['macro']:+.4f}")
    print(f"  A153 found +0.0575 WITHOUT augmentation; its decision rule:")
    print(f"     'advantage survives at >= +0.02  => the cache is worth building;")
    print(f"      it collapses                    => A ends here.'")

    same = set(a["ids"]) & set(b["ids"])
    ia = {s: i for i, s in enumerate(a["ids"])}
    ib = {s: i for i, s in enumerate(b["ids"])}
    ai = [ia[s] for s in a["ids"] if s in same]
    bi = [ib[s] for s in a["ids"] if s in same]
    pa, pt = a["pred"][ai], a["truth"][ai]
    qb = b["pred"][bi]
    rng = np.random.default_rng(20261005)
    d = []
    for _ in range(400):
        k = rng.integers(0, len(ai), len(ai))
        try:
            ma, _, _ = macro(pa[k], pt[k])
            mb, _, _ = macro(qb[k], pt[k])
            d.append(ma - mb)
        except Exception:
            pass
    d = np.asarray(d)
    if len(d):
        print(f"\n  paired bootstrap over studies ({len(d)} draws):")
        print(f"     mean {d.mean():+.4f}  95% CI [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]"
              f"  P(A5>raptor) = {np.mean(d > 0):.3f}")
    print("\n  per label:")
    for l in LABELS:
        x, y = a["per"].get(l), b["per"].get(l)
        if x is None or y is None:
            continue
        print(f"     {l:<18} a5={x:.4f}  raptor={y:.4f}  d={x - y:+.4f}")
