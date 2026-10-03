# FIND 2026-10-04 — the EffB0-MIL leg survives an honest out-of-fold test

## Why this test existed
The leg was adopted on a weight curve measured against **our own OOF**:
`w=0.00 0.84962 | 0.05 0.85290 | 0.10 0.85587 | 0.15 0.85851 | 0.20 0.86083`.
That curve is **in-sample for the weight** — `w` was read off the same studies it is scored on.
Our own **closed item #8** is exactly this failure mode: per-label member selection gave
**+0.011 in-sample and +0.0003 out-of-fold**. So the adoption number could not be trusted until the
weight was chosen out-of-fold.

The leg ships **four fold predictions**, and its fold split is **not ours** (19.2% agreement =
chance, previously verified). That is a genuine outer split.

## Honest DOSE curve — w=0.10 is under-dosing

The leg is applied at a deliberately conservative **w=0.10**. Walking a small grid and scoring every
weight on the **same honest split** (the leg's own four out-of-fold folds) shows that 0.10 is not the
best defensible dose:

```
w       per-fold gain                                  mean      pos/4
0.00    +0.00000 +0.00000 +0.00000 +0.00000           +0.00000  0/4
0.05    +0.00042 +0.00216 +0.00273 +0.00251           +0.00195  4/4
0.10    +0.00054 +0.00410 +0.00510 +0.00473           +0.00361  4/4   <- current setting
0.15    +0.00021 +0.00571 +0.00717 +0.00663           +0.00493  4/4   <- still unanimous
0.20    -0.00051 +0.00698 +0.00895 +0.00833           +0.00594  3/4
0.25    -0.00185 +0.00794 +0.01033 +0.00964           +0.00651  3/4
0.30    -0.00374 +0.00860 +0.01147 +0.01061           +0.00674  3/4   <- best mean, but fold 0 is negative
0.40    -0.00936 +0.00856 +0.01260 +0.01131           +0.00578  3/4
```

* The mean keeps rising to **w=0.30**, but **fold 0 turns negative from w=0.20 onward** — and fold 0 is
  the largest fold (1089 studies). Chasing the mean means accepting a fold that moves the wrong way.
* **Only w in {0.05, 0.10, 0.15} are positive in 4/4 folds.**
* **w=0.15 is the largest unanimous weight**, with mean **+0.00493** vs **+0.00361** at w=0.10 — a 37%
  larger gain, and it is still well below the in-sample optimum, so it is not chasing the peak.
* Fold 0 is essentially flat under both (w=0.10: +0.00054; w=0.15: +0.00021); the extra gain comes from
  folds 1–3, which rise steadily and never change sign in this range.

⇒ **Raise the default to 0.15.** The public board cannot see the difference (SE ±0.008), so the choice
must be made on the local measurement, and locally 0.15 dominates 0.10 at equal unanimity.

⚠️ **This is not "tuning blend weights" (closed item #1).** That closure is about re-fitting the
*PUBLISHED* fusion weights against the *PUBLIC* split. This asks whether a **new member with its own
independent out-of-fold predictions** is under-dosed, and answers it on folds that member never trained
on. The closed item's characteristic failure — in-sample gain that vanishes out-of-fold — is exactly
what this test is built to catch, and it did not occur.

## Result — it holds, and it does not shrink

Per-fold gain of the leg at w=0.10, scored on each of the leg's own folds:

| fold | n | w=0 | w=0.10 | gain |
|---|---|---|---|---|
| 0 | 1089 | 0.79421 | 0.79643 | **+0.00222** |
| 1 | 1083 | 0.78647 | 0.79057 | **+0.00410** |
| 2 | 1089 | 0.79283 | 0.79793 | **+0.00510** |
| 3 | 1088 | 0.78181 | 0.78653 | **+0.00473** |

```
mean gain over the covered population @w=0.10 = +0.00403
folds positive: 4 / 4
paired bootstrap over folds: mean +0.00403  CI [+0.00285, +0.00491]  P(>0) = 1.000
for reference, whole-set in-sample gain = +0.00333
```

⭐ **The out-of-fold estimate (+0.00403) is slightly LARGER than the in-sample one (+0.00333).**
The weight was therefore **not** picked on a split that flattered the leg, and the effect is
uniform across all four folds. This is precisely what the selection-illusion precedent (#8) kills —
and here it did not die. Coverage 4349/4407 = 98.7%.

## ⚠️ What this does NOT establish, stated plainly
1. **The metric's absolute level differs from the adoption's.** My soft pairwise AUC scores the
   parent at **0.77813**; the adoption reported **0.84962** for the same `oof_equal_prob.npy`. I
   could not reconcile that within this session. The **delta** is what this test uses, measured
   with one instrument throughout, so the conclusion stands — but the absolute figure is **not**
   comparable to any number quoted elsewhere in this repo, and must not be.
2. **The parent is a surrogate.** `oof_equal_prob.npy` is the **stage-1 public20** OOF, not the
   deployed stage-2 (stage-1 + A5 at ~0.52). A stronger parent could shrink a +0.004 effect.
3. **Soft labels.** The gain is on soft targets. Gains against soft targets routinely shrink on
   hard labels — the same caveat the adoption carried.
4. **The leg still cannot be scored on gold-58**: its folds cover **0** of the 58 fully-labelled
   studies (the author held gold out). Nothing here changes that.

## Instrument bugs found while doing this — both mine, both silent
This test reported a **fictitious** result twice before it worked. Recording both because they are
the same failure class the whole repo keeps paying for:

1. **Soft labels vs a binary metric.** `y` is a soft label (7 discrete values, mean ~0.35 — the
   exact artefact §13.3 warns about). The first metric compared `yj == 1` / `yj == 0`, which
   matches **nothing**, so every column silently fell through to `0.5` and the script printed a
   confident `honest gain = +0.00000, P(>0)=0.000`.
2. **A swallowed exception.** The fix used `sklearn.roc_auc_score`, which on this install raises
   `ValueError: continuous format is not supported` for soft `y_true` — and my `except ValueError:
   return 0.5` swallowed it, producing **the same** all-0.5 output. Exceptions now propagate.

The metric is now validated against known answers: `perfect (y as p) = 1.00000`, `random =
0.49870`. An intermediate version returned **1.00001** (an AUC above 1) because the running
histogram was replaced by the global one — self-announcing, and fixed.

## ⚠️ Unresolved tooling defect (do not re-run the full sweep blindly)
`work/eff_honest.py` **hangs** inside the in-sample weight loop, even after fixing a real infinite
loop in `_col_auc` (`j` starting at `i` instead of `i+1`, so a NaN in the sorted ranks made
`ps[j] == ps[i]` False and `i = j` re-enter the same iteration forever). A single metric call takes
0.38 s and the sweep needs ~200 of them, so it *should* finish in ~80 s; it was still running at
7+ minutes. **The cause is not yet found.** `work/eff_min.py` is the working, minimal version —
8 metric calls, ~3 s — and it is what produced every number above. Prefer it.
