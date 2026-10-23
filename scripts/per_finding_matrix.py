# -*- coding: utf-8 -*-
"""
Per-finding AUC across every arm we have, on the 870-study fold-0 instrument.

WHY THIS, AND WHY NOW
    The RSNA metric is macro ROC-AUC over 12 findings -- a plain mean. So +0.01 on ANY one
    finding is +0.000833 on macro, and a +0.005 macro gain needs only +0.06 spread across the
    findings. Our own v5 eval just showed the per-finding profile is very uneven:

        Effusion 0.983  Medial OA 0.974  ACL 0.963   <- strong
        Lateral OA 0.807  PF OA 0.830  Contusion 0.833  Synovitis 0.835   <- weak

    The strategy question that decides everything: are the weak findings weak for EVERYONE
    (intrinsic metric difficulty -> no lever) or weak only for US (a real, addressable gap)?
    A per-finding matrix answers it directly, and costs nothing -- every number comes from
    OOF files already on disk.

    Two diagnostics come out of it:
      SPREAD   across arms per finding: where ensembling or specialisation can still move.
      HEADROOM best-arm minus our shipped arm per finding: where we are leaving points.
"""
import pathlib
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def per_finding(pred, truth):
    out = []
    for j in range(12):
        y = (truth[:, j] > 0.5).astype(int)
        out.append(roc_auc_score(y, pred[:, j]) if len(set(y.tolist())) > 1 else np.nan)
    return np.array(out)


def main():
    arms = {}
    for p in ROOT.rglob("raptor_oof_*_fold0.npz"):
        try:
            z = np.load(p, allow_pickle=True)
            pred = np.asarray(z["pred"], np.float64)
            truth = np.asarray(z["truth"], np.float64)
            ids = [str(u) for u in z["ids"]]
        except Exception:
            continue
        if pred.ndim != 2 or pred.shape[1] != 12 or pred.shape[0] < 800:
            continue
        tag = p.name.replace("raptor_oof_", "").replace("_fold0.npz", "")
        if tag in arms:
            continue
        arms[tag] = (per_finding(pred, truth), pred, truth, ids)

    if not arms:
        print("no large-n OOF arms found")
        return
    names = sorted(arms)
    M = np.vstack([arms[n][0] for n in names])
    macro = M.mean(axis=1)
    order = np.argsort(-macro)

    print(f"{len(names)} arms on the 870-study instrument\n")
    print(f"{'finding':<18} {'best':>6} {'arm':<18} {'worst':>6} {'spread':>7} {'mean':>6}")
    print("-" * 74)
    rows = []
    for j, lab in enumerate(LAB):
        col = M[:, j]
        b, w = int(np.nanargmax(col)), int(np.nanargmin(col))
        rows.append((lab, col[b], names[b], col[w], col[b] - col[w], np.nanmean(col)))
    for lab, bv, bn, wv, sp, mu in sorted(rows, key=lambda r: -r[4]):
        print(f"{lab:<18} {bv:6.4f} {bn:<18} {wv:6.4f} {sp:7.4f} {mu:6.4f}")

    print()
    print("=== macro-AUC: top arms ===")
    for i in order[:8]:
        print(f"  {macro[i]:.4f}  {names[i]}")

    # what a per-finding ORACLE would buy -- the upper bound on per-label selection
    oracle = np.nanmax(M, axis=0).mean()
    bestsingle = macro[order[0]]
    print()
    print(f"best single arm macro          : {bestsingle:.4f}  ({names[order[0]]})")
    print(f"per-finding ORACLE macro       : {oracle:.4f}   (+{oracle - bestsingle:.4f})")
    print("  ^ that is the CEILING for per-label arm selection, chosen with hindsight on this")
    print("    very split, so it is an upper bound, not an expectation. If it is small, per-label")
    print("    selection is not the lever; if it is large, it is worth a real selection scheme.")
    print()
    print("=== where we are furthest below the best arm (our shipped msft0 vs the best) ===")
    if "msft0" in arms:
        ours = arms["msft0"][0]
        gap = np.nanmax(M, axis=0) - ours
        for j in np.argsort(-gap)[:6]:
            print(f"  {LAB[j]:<18} ours {ours[j]:.4f}  best {np.nanmax(M[:, j]):.4f}  gap {gap[j]:+.4f}")
        print(f"  macro gap to the ORACLE: {oracle - ours.mean():+.4f}")


if __name__ == "__main__":
    main()
