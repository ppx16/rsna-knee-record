# PLAN — feasibility of an image+metadata arm

**Status: pre-registered, not started.** Written 2026-09-26. Nothing here runs until the five pending
submissions are read back and the final two slots are chosen.

## 1. The claim being tested

> Acquisition metadata carries ordering information that the image arms do not already have, and a
> model that consumes **both** can capture more of it than a rank blend of the two.

**Why this is not idle speculation.** FACTS A109 measured the metadata-only contribution to two
genuine out-of-fold arms, choosing the blend weight on the training folds and never on the held-out
one:

| arm | alone | + metadata | gain | 95% CI |
|---|---|---|---|---|
| Rad | 0.8424 | 0.8507 | **+0.0082** | [+0.0063, +0.0100] |
| A5 | 0.9089 | 0.9108 | **+0.0019** | [+0.0008, +0.0029] |

The weights were stable across all five folds, and metadata alone predicts the labels at **0.678**
with no images at all. **Every arm in the four-stage pipeline is image-only** — DINOv2, A5,
RadImageNet, CoAtNet — so this information channel is unused except insofar as the 88-feature
calibrator consumes protocol counts.

**Why it may still fail, stated up front.** The gain shrinks as the arm strengthens
(0.842 → +0.008, 0.909 → +0.002). Extrapolated to the ensemble at ~0.94 the marginal gain is roughly
**0.000–0.001**, below our instrument's resolution. And most metadata is a **scanner/site proxy**,
while scanner was measured to be a covariate shared between training and the hidden set (A101), not a
removable group — so a gain could be a site shortcut rather than a finding.

## 2. The design: three legs, trained identically

The decisive control is **not** image-only versus image+metadata. It is whether the metadata's
*information* is doing the work, or merely its extra capacity and implicit regularisation. So:

| leg | inputs | purpose |
|---|---|---|
| **A** | image only | the baseline |
| **B** | image **+ metadata** | the hypothesis |
| **C** | image **+ metadata permuted across studies** | ⭐ **the control that makes B interpretable** |

Leg C has the same parameter count, the same optimiser, the same schedule and the same input shape as
B. The only difference is that the metadata no longer carries per-study information. **If B does not
beat C, the metadata contributes nothing and the route closes** — no matter how much B beats A.

Everything else is held fixed: same architecture, same seed, same epochs, **fixed epoch schedule**
(no selection on held-out scores — the community standard in A99), same folds.

### 2.1 Metadata features, split into two families

A109 used 55 study-level features built for free from `site_probe.csv`. For this plan they are
**deliberately split**, because they carry different risks:

* **Protocol family** — series counts by plane × fat suppression, number of series, distinct
  descriptions, TR/TE/slice-thickness aggregates. This is *acquisition protocol*: about how the study
  was taken, plausibly transferable to any scanner pool.
* **Site family** — manufacturer, model, software version, field strength, pixel spacing, rows/cols.
  This is *scanner identity*: a proxy for site, and the likely shortcut.

**A gain that comes only from the site family is rejected**, because it would be the model learning
which scanner a study came from rather than anything about the knee. Reported results always split
the two families.

### 2.2 Where metadata enters

Concatenate the standardised metadata vector to the pooled image embedding before the classification
head, so the head may learn interactions. This is the simplest variant that can express "if scanner
is X and this region responds, the finding is more likely", and it keeps the training script's
existing shape so the frontier's own trainer stays usable.

## 3. The falsification ladder

Each rung is cheap and each has a pre-registered kill condition. **A rung that fails stops the line**;
there is no "try a bit more" clause.

### Rung 0 — does metadata add to an arm at all? ✅ **PASSED** (A109)
Done, free, on CPU. +0.008 on a weak arm, +0.002 on a strong one, weights fold-stable.

### Rung 1 — does a trained model extract more than a rank blend? (**the decisive cheap test**)
Train legs A, B and C for **one fold**, ~12 epochs, on a single rented card. Costed below.

**Pass condition — both must hold:**
1. paired OOF macro AUC, **B − A**, 95% CI excludes 0;
2. paired OOF macro AUC, **B − C**, 95% CI excludes 0.

**Kill condition — any of:**
* either interval includes 0;
* the B − A gain comes only from the **site** family (§2.1);
* B − A is positive but below **+0.002**, i.e. no better than the rank blend already measured in A109
  — a trained arm has to earn its place over a free one.

**Second fold, only if rung 1 passes.** A single fold cannot distinguish a real effect from a fold
accident, and this plan does not pretend otherwise: rung 1 is a screen, not a result.

### Rung 2 — is it good enough to matter?
Measure the new arm **on the reference scale already built** (`scripts/arm_reference.py`), against A5
on a genuinely common held-out set, using the same report-label source.

**Pass condition:** solo macro **≥ 0.90**.

This is not an arbitrary bar. D14 measured that members at 0.85–0.87, entering at 15% weight, made the
ensemble **worse** (0.936 against 0.940). Our best existing arm measures 0.873 on this scale (A106).

**Kill condition:** below 0.88 — then it cannot enter at any weight, and no amount of tuning changes
that ordering.

### Rung 3 — is it decorrelated?
Ensemble value comes from **uncorrelated errors**, not from another good arm. Measure the rank
correlation between the new arm's errors and each reference arm's errors.

**The baseline is measured, not assumed** (2026-09-26, on 4406 studies, A5 against Rad — two arms that
demonstrably do coexist in the ensemble):

| | mean over the 12 labels |
|---|---|
| rank correlation of the two arms' OOF predictions | **0.6998** |
| **residual correlation** (`y − rank`, the ensemble-relevant one) | **0.8061** |

Per label the residual correlation runs from 0.647 (Fracture, the most independent column) to 0.885
(Effusion). So two arms good enough to sit in the same ensemble are still **81% correlated in their
errors**.

**Pass condition:** the new arm's mean residual correlation with A5 must be **materially below 0.81**
— the point of adding a fifth view is the part of the error the existing arms do not share.

**Kill condition:** at or above Rad's 0.81. Then the new arm, however accurate, is telling the
ensemble what Rad already tells it.

### Rung 4 — can it be injected?
Only reached if rungs 1–3 all pass. Adding an arm means editing the validated submission pipeline
(the 20/20 fingerprint gate, the fail-closed DINOv2 stage, the staged submission contract) and
re-running the full acceptance test in `RUNBOOK-0943.md`. **This is the highest-risk step in the
whole plan and the one most likely to cost more than it returns.** It is a separate decision, taken
with the arm's measured numbers in hand, not a continuation of this plan.

## 4. Cost

Measured anchors: our CoAtNet arm ran **554 s/epoch** at batch 8 with gradient checkpointing on one
24 GB card, so 12 epochs ≈ **1.85 h per leg**.

| item | quantity | cost |
|---|---|---|
| Rung 1, three legs, one fold | 3 × 1.85 h | **~5.6 GPU-hours** |
| Rung 1, second fold (only if it passes) | 3 × 1.85 h | ~5.6 GPU-hours |
| Reference-scale measurement, correlations | CPU | 0 |
| **Total to a decision** | | **~5.6 GPU-hours, ~¥12 on a rented 4090-class card** |

Against the AutoDL balance of **¥30.82** this is affordable, and it is why the plan is worth writing:
the decisive test is cheap. Note that AutoDL is the correct venue here — this needs a 24 GB card and
no competition DICOM beyond what the geometry code derives itself.

⚠️ **Quota note:** rung 1 does **not** touch Kaggle GPU. Training is exactly what the standing policy
routes to AutoDL (long and, once started, urgent), while the only Kaggle-GPU dependency in this plan
is the reference measurement, and that is already done.

## 5. Instrument rules this plan commits to

These are not optional refinements; each corresponds to a mistake already made in this project.

1. **No selection on the evaluation metric.** Fixed epoch schedule, no gold-selected checkpoints. The
   published lineage does the opposite (A98) and it inflates every absolute number it reports.
2. **Bootstrap by report group, not by study.** Studies sharing a report carry identical labels; the
   report-group axis is nearly clean in our folds (2 of 49 groups split, 4 studies — A100) but
   resampling studies still makes intervals too narrow (A99).
3. **Score every arm against the same label source**, and say which one. Report labels are model-read
   from free text; they support relative statements only.
4. **Leave-one-fold-out for anything fitted**, including blend weights, feature standardisation and
   imputation. A109's first draft failed exactly here in spirit: a construction that could not fall
   back on its strong input returned −0.10, which measured the construction and not the data.
5. **The permutation leg is mandatory.** Any image+metadata result without leg C is unreportable.

## 6. What would make me abandon this before rung 1

* The five pending submissions reveal that the current recipe does not reproduce, because then the
  priority is understanding that, not adding an arm.
* The error correlation between A5 and Rad turns out to be very low, which would mean the ensemble
  already has the diversity this arm would provide. *(Checked — it is not low: **0.8061 residual
  correlation**, so there is room for a genuinely different fifth view. This bullet is retired.)*
* The AutoDL balance cannot cover ~¥12 of GPU without more funds.

## 7. Assets already in place

| need | have |
|---|---|
| metadata features for all 4407 studies | `downloads/research/siteprobe/site_probe.csv` (built free by a CPU probe, 128 s, A101) |
| feature/signal analysis, tested both ways | `scripts/metadata_signal.py` |
| a fold split to train and evaluate on | `dataset-cv-fold/fold_file.json` |
| the reference scale and a genuine strong arm | `scripts/arm_reference.py`, `downloads/research/a5oof_full/speed_a5_raw.npz` (A104) |
| a trainer and the geometry code | published by the frontier: `train_coatnet_global96_exact.py`, `trainer_a6000_epochs01_06.py`, `cnx_dicom_geometry.py` |
| acceptance discipline for a new artifact | the `verify_*.py` pattern, each with a positive and a negative control (A88) |
