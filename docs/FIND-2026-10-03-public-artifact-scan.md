# FIND 2026-10-03 — public-artifact scan: the board is a plateau, and one real leg exists

## 1. The ">0.943 public notebook" question is CLOSED — all four are name-inflated

Enumerated **400 competition kernels** (score-descending) and ~120 datasets via the authenticated API,
and **read the bodies**. Every `>0.943` title is false:

| notebook | what its own body says |
|---|---|
| `ryokucha/rsna-knee-d4-blend-0946-ours10` | banner: **"Public LB: 0.943"**; the 0.946 comes from the author's **private** ConvNeXt ensemble at 10%, absent from the public metadata |
| `yamadan96/rsna-knee-d4-public0946` | verbatim: **"Correction: this scores 0.943, not 0.946 … I submitted exactly this notebook on 2026-09-28 and it scored 0.943 … The parent notebook reaches 0.946 by blending in its author's private ConvNeXt ensemble at 10%."** |
| `sushanthtiruvaipati/rsna-knee-d4-public0946-no-correction` | same correction text |
| `adonis122/rsna-knee-parent0957-routing` | body: **"Public score 0.941."** |

The `"Frontier SOTA Engine / Public LB: 0.943"` banner is copy-pasted across many kernels — **treat it
as a fork marker, not a measurement.** Also rejected after reading: `dinosaur-v5` (closed #1/#6),
`knee-s75-w50`/`w60` (fusion-weight forks, closed #1), `d4-cnx-r1-five-percent` (closed #1),
`rad-noexclude` (re-skin), `the-bee-s-knees-final` (self-declares "the 0.940 belongs to that recipe,
not this notebook").

## 2. ⭐ The full public LB, for the first time — and it settles the target

`kaggle kernels output ryanholbrook/rsna-knee-abnormalities-efficiency-lb` yields
**`full_leaderboard.csv` — all 4745 teams**, best public score + date, snapshot through 2026-10-01 21:45.

```
median 0.920 | p75 0.941 | max 0.960
0.943 bin: 558 teams        [0.941,0.943): 990 teams
>=0.944: 240 | >=0.945: 195 | >=0.950: 79 | >=0.955: 31
0.944:45  0.945:46  0.946:28  0.947:16  0.948:11  0.949:15  0.950:13 ... 0.958:4  0.960:1
>=0.944 submissions still arriving ~30–50/day (09-27:20 09-28:37 09-29:31 09-30:37 10-01:49)
```

**⇒ 558 teams sit at exactly 0.943, and not one of the 240 teams at ≥0.944 publishes code.** So every
public "0.945+" artifact is private-asset-inflated or a lucky public draw. **We are inside the dominant
plateau; nothing separates 0.943 from 0.944 on the public board.** This CSV is now the best available
instrument for filtering, and it should be pulled before any future "let's copy that notebook" move.

## 3. ⭐ THE FINDING — `tbongc/rsna-knee-effb0mil-sl16-tv7w` (EfficientNet-B0 2-level MIL)

`https://www.kaggle.com/datasets/tbongc/rsna-knee-effb0mil-sl16-tv7w` (61 MB, 4 folds, 09-28).

**Measured by us, not the author's number.** Its published OOF (`oof_effb0mil_sl16_tv7w_f0..f3.npy` +
`_ids.csv`, 4349/4407 train studies) was aligned against our own
`downloads/oof/oof_equal_prob.npy` / `train_y.npy` and rank-averaged:

```
w=0.00  meanOOF 0.84962 | w=0.05 0.85290 | w=0.10 0.85587 | w=0.15 0.85851 | w=0.20 0.86083 | w=0.30 0.86440
```

* **+0.0063 at w=0.10** (and +0.0112 at w=0.20) against the labels we already select on.
* **Positive on all 12 labels. Never negative at any weight.** Monotone in w.
* **Largest gains exactly where we are weakest** — which is the whole point:

| label | ours | effb0 | Δ @w=0.10 |
|---|---|---|---|
| **Lateral Meniscus** | 0.8014 | 0.8363 | **+0.0131** |
| Medial Meniscus | — | — | +0.0093 |
| PF OA | — | — | +0.0088 |
| ACL | — | — | +0.0072 |

* **Genuinely decorrelated**: per-label rank correlation Fracture 0.483, ACL 0.609, LatMen 0.643,
  MCL 0.649, Baker's 0.667. This is the **divergence** that A163 / HANDOFF §13.2 says is the only thing
  that pays — and it is a *different architecture family* (2-level slice→slot→study MIL on EffNet-B0),
  not another DINOv2/CoAtNet view.
* **Confounds checked**: (a) its fold split is **not** ours — agreement 834/4349 = **19.2 %, i.e.
  chance**, so no fold leakage; (b) our labels score it 0.8438 while its author reports 0.9357 on his
  own `soft_labels_v1`, so it was **not** trained on our label set — the gain is cross-label-source, not
  self-fitting; (c) `tbongc/rsna-knee-labels` is 403 so the label tables could not be diffed directly.
* **It is in NONE of the nine closed items.** Not a fusion weight, not a per-label route, not a
  DINOv2/CoAtNet view, not a bigger backbone, not TTA, not per-label member selection.

### ⭐ Why this also resolves the §97.3 "meniscus specialist" shelving
`HANDOFF.md` §97.3 shelved `renta0426`'s meniscus residual model, and left the fact that
**Lateral Meniscus is the largest single shortfall in the macro average and no tested knob moved it**
(pooling-max was only +0.0009). The rejection reasons were that it was a **residual fitted to a
different (0.933) baseline** and that its author never wired it in. **Neither applies here**: this is an
independent leg, its marginal gain is measured *on our own OOF*, and it is strongest on exactly that
label. **The gap §97.3 recorded is still open, and this is the first artifact that addresses it.**

### Cost, and the honest caveat
Mount `tbongc/rsna-knee-cache336-v1` (public, **21 GB** — holds `sl16_fix.csv`, `s_cache_e13_224.npy`,
the 4-slot sag/cor/ax-fluid + sag-t1 16-slice 224 px **train** cache) + 4 × 16 MB checkpoints +
`effb0mil_lib.py`. Inference itself is trivial (EffNet-B0 @224). **The real work — and the real risk —
is building the TEST-side 224 px 4-slot cache with the same flip/laterality convention**: the library
carries an explicit per-row `flip` remap and a laterality fix, and **getting that wrong degrades
silently** rather than crashing.

⚠️ The +0.0063 is measured against **soft labels** — the same currency we select on, but gains against
soft targets routinely shrink on hard labels, and our own **closed item #8 (selection illusion:
+0.011 in-sample → +0.0003 out-of-fold)** is the precedent for exactly this failure. Expected LB:
**0.943 flat, with a genuine chance of 0.944.** At w=0.05–0.10 the downside is small.

## 4. Rejected / watch-list (so nobody re-opens these)

* **Different architecture, but no head and no OOF, 0–10 downloads** — not usable now:
  `kirilllabzin/knee-medsiglip448-v19-source` (1.6 GB MedSigLIP-448, offline runtime),
  `prithvirajrp/orthofoundation-rsna-knee-raw-embeddings` (2.9 GB, hours old, no head),
  `eduardozw/rsna-knee-weights` "Approach B" (frozen R(2+1)D-18 video + multilingual-e5 text teacher —
  structurally the most interesting thing on the board, but a frozen 128³ video encoder with a
  from-scratch head is very unlikely to clear the 0.90 gate),
  `sdndfn/rsna-knee-trained-checkpoint` (text-gated fusion, no score),
  `dhyey166/rsna-knee-resunetpp-src` (segmentation source only),
  `yutoshimomura/...crop140-518-cache` (cache only).
* **Same-family, low marginal value** — closed #2/#3: `tbongc/...v8geo-v7w`, `v10geo`,
  `kawaharataishi/...coatnet-distilled-checkpoint`.
* **Label-side** (training-side, expensive; and closed #9 is the precedent): `ctogaurav/...llm-labels-v4-blend`,
  `tranphunghia233871/...labels-recalibrated`, `sebgarciavillalobos/...report-annotations`,
  `gperdrizet/...pseudolabeled`, `dhiyaneshbabu/...labelled-train`. Note **all public label sets
  measure ≈0.89 gold-58 AUC — label quality is the ceiling, not ensembling.**
* **⭐ One false alarm worth recording**: the count of teams at ≥0.944 is **240**, not "very few". An
  earlier reading of that same artifact almost produced the opposite conclusion about whether we are
  behind the frontier. Read the distribution before drawing a strategic conclusion from it.
* **Watch, don't act**: `quannguyen0501/130mm-320-stage0-5-preprocessing-part1..3` — a genuinely new
  geometry (130 mm @320 px, **six** slots sag/cor/ax × fluid/struct, using struct series we don't) but
  stage-0.5 preprocessing only: no model, no score, 0 votes.
  `yoshitown/rsna-knee-src-modules` — an honest exp001→exp030 ablation repo. Independent confirmation at
  the 0.91 level: 140 mm physical crop, medial→lateral absolute alignment, **horizontal flip completely
  banned (negative TTA — matches closed #7)**, 12-way separate attention pooling as the biggest
  architectural win, soft labels + confidence-weighted masked BCE (`w = 2|p−0.5|`), quantile rank
  averaging, target-plane routing. **Nothing there exceeds our recipe.**
