# -*- coding: utf-8 -*-
"""
The gold-58 local instrument, built from the raptor_gold_*.npz files already on disk.

Why this exists: the prereg's threshold is "local delta >= +0.0005", but nothing had
ever calibrated that threshold against a KNOWN board result. msft2 is the one arm with a
measured board gain (solo 0.9069, "+0.00025 by the 11x decay law"). If the instrument
cannot see msft2 move, then a null result on v9 means nothing and the threshold is
meaningless.

Every raptor_gold_*.npz holds pred/truth/ids over the same competition gold split
(train.csv rows with all 12 LAB columns non-null). truth is SOFT, so it is binarised at
>0.5 -- the convention already recorded, because roc_auc_score refuses soft labels.
"""
import pathlib
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def macro_auc(pred, truth_bin):
    """Macro ROC-AUC over the findings that actually have both classes present."""
    aucs, used = [], []
    for j, name in enumerate(LAB):
        y = truth_bin[:, j]
        if len(set(y.tolist())) > 1:
            aucs.append(roc_auc_score(y, pred[:, j]))
            used.append(name)
    return float(np.mean(aucs)), aucs, used


def load(path):
    z = np.load(path, allow_pickle=True)
    return (np.asarray(z["pred"], dtype=np.float64),
            np.asarray(z["truth"], dtype=np.float64),
            [str(u) for u in z["ids"]])


def main():
    files = sorted(ROOT.rglob("raptor_gold_*.npz"))
    if not files:
        print("no raptor_gold_*.npz found")
        return
    rows = []
    ref_ids = ref_truth = None
    for p in files:
        try:
            pred, truth, ids = load(p)
        except Exception as e:
            print(f"  SKIP {p.name}: {e}")
            continue
        if pred.ndim != 2 or pred.shape[1] != 12:
            print(f"  SKIP {p.name}: pred shape {pred.shape}")
            continue
        if ref_ids is None:
            ref_ids, ref_truth = ids, truth
        same = (ids == ref_ids)
        tb = (truth > 0.5).astype(int)
        m, aucs, used = macro_auc(pred, tb)
        rows.append((m, p.relative_to(ROOT).as_posix(), len(ids), same,
                     float(np.mean(np.abs(truth - ref_truth))) if same else float("nan")))

    rows.sort(reverse=True)
    print(f"gold-58 instrument: {len(rows)} prediction files, "
          f"reference split = {len(ref_ids)} studies, {len(set(tuple(i) for i in [ref_ids]))} ids")
    print(f"binarised truth>0.5 -> positives per finding: "
          f"{[int((ref_truth[:, j] > 0.5).sum()) for j in range(12)]}")
    print()
    print(f"{'macroAUC':>9}  {'n':>4}  {'same-ids':>8}  {'d-truth':>8}  file")
    print("-" * 110)
    for m, rel, n, same, dtruth in rows:
        print(f"{m:9.4f}  {n:4d}  {str(same):>8}  {dtruth:8.5f}  {rel}")

    # ---- calibrate: does the instrument see msft2 move? -------------------------------
    print()
    print("=" * 110)
    print("CALIBRATION -- the instrument must be able to see a KNOWN board move")
    print("  known: msft2 solo = 0.9069 board, and '+0.00025' as a member (11x decay law)")
    by = {rel: m for m, rel, n, same, d in rows}

    def get(sub):
        for rel, m in by.items():
            if sub in rel:
                return m, rel
        return None, None

    for tag in ("raptor_gold_msft0.npz", "raptor_gold_nx2240.npz", "raptor_gold_cn0.npz"):
        m, rel = get(tag)
        if m is not None:
            print(f"  {tag:<32} gold-58 macroAUC = {m:.4f}   ({rel})")
        else:
            print(f"  {tag:<32} not present locally")


if __name__ == "__main__":
    main()
