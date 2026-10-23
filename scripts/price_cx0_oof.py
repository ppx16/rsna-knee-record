"""Price the ConvNeXt fold-0 member on the instrument that can actually resolve it.

The pre-registered kill rule thresholds at SWA gold 0.88 / 0.90, but that number comes from
gold-58, whose bootstrap CI half-width this repo measured at about +-0.021 (A83) -- WIDER than
the whole 0.88..0.90 band. So "0.9003 clears the 0.90 bar" is not a resolvable statement.

The run also wrote an OOF over the 870 held-out studies of fold 0. That is ~15x the sample, so
its CI is roughly sqrt(15) = ~4x tighter. This computes the macro ROC-AUC on it and compares
the two instruments, so the continue/stop call rests on the tighter one.
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

import numpy as np

D = pathlib.Path(r"D:\kaggle\rsna-knee\work\cx0-oof")
z = np.load(D / "raptor_oof_cx0_fold0.npz", allow_pickle=True)
print("npz keys:", list(z.keys()))
for k in z.keys():
    a = z[k]
    print(f"  {k:<20} shape={getattr(a, 'shape', None)} dtype={getattr(a, 'dtype', None)}")
    if a.dtype == object or a.ndim == 0:
        print("      value:", str(a)[:200])

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def macro_auc(pred, truth):
    """Per-column ROC-AUC on finite, non-degenerate columns, then the macro mean."""
    from sklearn.metrics import roc_auc_score
    aucs, used = {}, []
    for j, lab in enumerate(LABELS):
        p, t = pred[:, j], truth[:, j]
        m = np.isfinite(p) & np.isfinite(t)
        if m.sum() < 10 or len(np.unique(t[m])) < 2:
            aucs[lab] = None
            continue
        aucs[lab] = float(roc_auc_score((t[m] > 0.5).astype(int), p[m]))
        used.append(lab)
    vals = [aucs[l] for l in used]
    return (sum(vals) / len(vals) if vals else float("nan")), aucs


pred = None
truth = None
for k in z.keys():
    lk = k.lower()
    if pred is None and ("pred" in lk or "prob" in lk or "oof" in lk) and z[k].ndim == 2:
        pred = z[k].astype(np.float64)
    if truth is None and ("truth" in lk or "label" in lk or "target" in lk or "y" == lk) and z[k].ndim == 2:
        truth = z[k].astype(np.float64)

if pred is None or truth is None:
    print("\n!! could not identify pred/truth arrays automatically; dumping shapes above")
    raise SystemExit(1)

print(f"\npred {pred.shape}  truth {truth.shape}")
macro, aucs = macro_auc(pred, truth)
print(f"\nOOF macro ROC-AUC on {pred.shape[0]} held-out studies = {macro:.4f}")
print("\nper label:")
for lab in LABELS:
    v = aucs.get(lab)
    print(f"   {lab:<18} {'n/a (degenerate or masked)' if v is None else f'{v:.4f}'}")

print()
print("=" * 74)
print("COMPARING THE TWO INSTRUMENTS")
print("=" * 74)
print(f"  gold-58 (n=58, CI half-width ~0.021 per A83) : SWA 0.9003")
print(f"  OOF    (n={pred.shape[0]}, ~4x tighter)            : {macro:.4f}")
print()
print("  Per A83 the project bands an effect of 0.004-0.010 as 'worth upgrading the instrument,")
print("  not building a new member', and the A167 member gate is solo >= ~0.90.")
