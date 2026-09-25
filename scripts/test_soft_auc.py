"""Does the SOFT-truth AUC actually make the CV a sharper instrument?

WHY
    `cv_score.py` scores the CV against the report parser's SOFT label, binarised at 0.5.  That
    is the right analogue of the competition metric, but binarising injects label noise into the
    EVALUATION: a study at 0.49 and one at 0.51 are near-identical yet land in opposite classes.

    `soft_auc` is the natural generalisation (probability that a draw from the positive soft
    mass outranks a draw from the negative soft mass).  It should be less noisy.  "Should be" is
    not a measurement, so this measures it, on the one fold we already have:

      1. SANITY   : with binary truth, soft_auc must reproduce roc_auc_score exactly.
      2. SHARPNESS: build a knowingly-damaged copy of cvg0's predictions, then compare the
                    PAIRED bootstrap CI width under both metrics.  A sharper instrument gives a
                    TIGHTER CI around the same true difference.  If soft_auc is not tighter,
                    the claim in its docstring is wrong and this prints that.

Usage:
    python scripts/test_soft_auc.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cv_score import soft_auc, macro_auc, macro_soft_auc, load  # noqa: E402

SRC = ROOT / "downloads" / "g0" / "raptor_oof_cvg0_fold0.npz"


def paired_ci(Y, Pa, Pb, metric, n=400, seed=0):
    """95% CI half-width for metric(Pa) - metric(Pb), bootstrapping studies."""
    rng = np.random.default_rng(seed)
    N = Y.shape[0]
    d = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        a = metric(Y[idx], Pa[idx])[0]
        b = metric(Y[idx], Pb[idx])[0]
        if a == a and b == b:
            d.append(a - b)
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(hi - lo), float(d.mean())


def main() -> int:
    if not SRC.exists():
        print("SKIP: %s not found" % SRC)
        return 1
    P, Y, ids = load(str(SRC))
    print("loaded %s  pred %s truth %s" % (SRC.name, P.shape, Y.shape))
    fails = []

    # ---- 1. sanity: binary truth -> ordinary AUC ----
    B = (Y >= 0.5).astype(float)
    worst = 0.0
    for j in range(Y.shape[1]):
        if 0 < B[:, j].sum() < len(B):
            worst = max(worst, abs(soft_auc(B[:, j], P[:, j]) - roc_auc_score(B[:, j], P[:, j])))
    ok = worst < 1e-9
    print("\n1. soft_auc == roc_auc_score on BINARY truth : %s (max abs diff %.2e)"
          % ("OK" if ok else "FAIL", worst))
    if not ok:
        fails.append(1)

    # ---- 2. sharpness on a knowingly-damaged copy ----
    rng = np.random.default_rng(7)
    Pd = P.copy()
    n = P.shape[0]
    idx = rng.permutation(n)[: n // 4]                 # 25% of studies, per-label shuffle
    for j in range(P.shape[1]):
        Pd[idx, j] = P[rng.permutation(idx), j]

    w_bin, d_bin = paired_ci(Y, P, Pd, macro_auc)
    w_sft, d_sft = paired_ci(Y, P, Pd, macro_soft_auc)
    print("\n2. paired CI width around a 25%%-shuffled damage")
    print("   binarised (n=870)  delta %+.4f   CI width %.4f" % (d_bin, w_bin))
    print("   soft      (n=870)  delta %+.4f   CI width %.4f" % (d_sft, w_sft))
    verdict = "TIGHTER" if w_sft < w_bin * 0.98 else ("similar" if w_sft < w_bin * 1.02 else "WIDER")
    print("   => soft-truth metric is %s (%.1f%% of the binarised width)"
          % (verdict, 100.0 * w_sft / w_bin))
    if verdict == "WIDER":
        print("   !! the docstring claim is WRONG -- do not switch, and say so in FACTS")

    print("\n3. macro values on the same predictions")
    print("   binarised macro %.4f   soft macro %.4f   (different scales by construction)"
          % (macro_auc(Y, P)[0], macro_soft_auc(Y, P)[0]))

    print("\nVERDICT: %s" % ("sanity OK" if not fails else "FAILED %s" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
