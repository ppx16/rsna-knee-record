# RSNA Knee Abnormality Detection — Competitive Intelligence Report
Research date 2026-09-11. Read-only; nothing pushed to Kaggle. All artifacts in this directory.

## SO WHAT — highest-leverage actions

**1. Your 0.941-class submission is PENDING, not failed — do not resubmit it.**
`kaggle competitions submissions rsna-knee-abnormality-detection` (with `$env:PYTHONUTF8=1`):
```
ref 56166132  2026-09-11 13:21:00  public 0.941-class ensemble (kernel_sources restored)  SubmissionStatus.PENDING
ref 56164308  2026-09-11 11:10:58  v4 224x12 dinov2-large seed2026, 5 members rank-mean   COMPLETE  0.854
ref 56053045  2026-09-06 10:35:17  224px x6 slices self-trained (5fold seed2026)           COMPLETE  0.874
ref 56045533  2026-09-06 03:08:12  self-trained 5-fold weights (r336/slices3)               COMPLETE  0.842
ref 56030709  2026-09-05 10:35:32  fork pilkwang baseline v1 (inference, rank-mean 20 members) COMPLETE 0.891
```
Evidence: the submission row exists with a `ref` and `PENDING` status and blank scores. Also note your
0.891 is *already* pilkwang's baseline reproduced — so your own best is not behind the public baseline.

**2. Stop training. The 0.941 recipe is 100% inference-only and needs zero training.**
All five top notebooks load every weight from other people's public datasets (subagent deep-read of the
5 highest-scoring digests; zero optimizer/backward/training loops). Your own trained models have already
been measured as worthless *as ensemble additions* — see action 4. Redirect all effort to reproducing the
4-arm rank-blend and its per-target routing.

**3. Attach three datasets you are missing; they are the single biggest gaps.**
- `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` (2,698 MB, upd 2026-08-17) — used by **37 of the
  top-60 kernels**. It is a one-mount bundle: `rsna-knee-weights/manifest.json`, `knee-mri-fold-weights/`,
  `resnet-50-radimagenet-marwan/ResNet50.pt`, `rsna-knee-e9-radimagenet-heads-v15/v52_radimagenet_heads.pt`,
  `kernel-sources/rsna-knee-e13-train/rsna_rad_e11/v52_e11_heads.pt`. Replaces ~6 separate attachments.
- `dreaddevelopment/raptor-knee-widedense` (558 MB, 70 votes) — used by **38 of top-60 kernels**.
  Subtitle: "Trained CoAtNet weights that score twelve findings on a knee MRI study".
- `diversen/timm-coatnet-rmlp-2-rw-384-pretrained` (262 MB) — the exact ImageNet backbone the Raptor arm
  is built on. Cheap insurance against a timm/docker-image change breaking your commit with internet off.
Also add kernel source `romantamrazov/rsna-knee-dinosaur-v4-train` (**16 of top-60**, and it is the only
extra kernel source beyond your two e11/e13 ones at meaningful frequency).

**4. Do not spend submissions re-testing these — they are already measured negative.**
From `yosukeinada/rsna-knee-0-937-22-submissions-6-lessons` (22-submission ledger, all LB-verified) and
`starkhushi/six-ensembling-ideas-that-failed-0-873-to-0-919`:
- Your own trained model added to a strong public base: a **0.910** component at weight 0.10 and 0.15 gave
  **displayed delta 0.000**; a **0.874** component gave **0.933 / 0.935 — worse than the 0.936 base**.
- Forecast-vs-actual: an expected 0.941 (from a Gold58 estimate) came in at **0.937 = −0.004 error**.
  "A Gold58 delta is not a calibrated public-LB prediction."
- **The 58 gold studies cannot referee +0.004 decisions.** Bootstrapped 95% CI on a *fixed* prediction set
  is wider than the whole 0.873→0.919 climb (0.046). Two of three "best ideas" reversed sign under nested
  validation; per-label member dropping was in-sample +0.011 and out-of-fold **+0.0003**.
- **Ensembles are fed by disagreement, not member quality.** The *weakest* member (RadImageNet ResNet-50,
  solo 0.843) added **+0.009** on joining; a 7th DINOv2 fold, the *strongest* single member, added **−0.0007**.
  → Do not add more DINO folds. Add a medically-pretrained / differently-failing architecture.
- TTA on ensemble members: helps each member +0.003…+0.005 but moves the blend **−0.001**.
- Borrowing Effusion signal into Synovitis: monotonically worse at every mixing weight.

**5. The one concrete weight change worth +0.002–0.003 public: per-target outer routing.**
Documented progression, independently corroborated by `evgendvorkin/rsna-baseline` markdown:
| step | notebook | change | LB |
|---|---|---|---|
| 1 | `prvsiyan/head-and-shoulders-knees-and-toes` | 4 CoAt arms; outer CoAtNet weight **global 0.60** | 0.937 |
| 2 | `renta0426/rsna-knee-hybrid-raptor-lateral-meniscus-r100` | **only** change: Lateral Meniscus routing Transformer 0.00 / Raptor 1.00 | 0.940 |
| 3 | `mattiaangeli/bend-the-knee-to-the-dinosaurs` | + residual-gated CoAt e4/e6/e8 at 0.40 inside Raptor; + per-target routing | 0.941 |
Final 0.941 routing dict (verbatim from the source):
```python
_coatnet_weight = {label: 0.60 for label in _blend_labels}
_coatnet_weight.update({'ACL': 0.75, 'Medial Meniscus': 0.80,
                        'Lateral Meniscus': 1.00, 'Lateral OA': 0.75, 'Fracture': 0.75})
```
`evgendvorkin`'s own words: "Key idea: isolate one problematic target and give it the full weight of the
strongest branch." `maverickss26` states LM at 1.00 "bought +0.002 on the public split."

**6. One cheap, low-risk probe with a real LB-verified precedent (audit the exclude list).**
Every top notebook inherits `_RAD_EXCLUDE = ("Baker's", "Fracture")`, i.e. two labels never receive the
RadImageNet blend. `dariushafshar/rsna-0-921-audit-before-you-exclude-a-target` audited it per-label:
Fracture ranks **4th of 12** in blend lift (+0.0077 full-population) yet was excluded; Baker's ranks 11th
and goes **negative** on gold-58. Changing only that line to `_RAD_EXCLUDE = ("Baker's",)` scored
**0.921 vs 0.920** on the real leaderboard (ref `55654006`, 2026-08-20), moving rank 251→181 of ~2,070
(outside bronze → inside bronze). Offline prediction was +0.000639; direction and rough size confirmed.

**7. Get a real 4,407-study OOF table before touching another weight.**
Second-pass dataset search found at least four usable OOF/prediction artifacts:
`denisrukavishnikov/rnsa-knee-pred-public20-oof-audit` (9.0 MB, OOF + audit for the public-20 model),
`nekkon/rsna-knee-image-oof-gold` (tiny — a predictions table, not weights),
`hardiksoni7614/rsna-knee-pretrained-output` (198 MB), `prvsiyan/rsna-knee-v52-oof-receipt-20260810` (0.3 MB).
Fitting the outer routing weights and the exclude-list decision on the full population instead of n=58 is
the highest-value *process* fix available, and it directly addresses the documented −0.004 forecast error
and the "CI wider than the effect" problem in action 4.

---

## 1. Public notebooks, ranked (top 25 by `sortBy=scoreDescending`)

Caveat restated: **the API exposes no public-LB score field**, and `totalVotes` co-varies with this
ordering, so "rank" below is an *approximate* score proxy, not a verified score. Author-declared scores
(column "declared") are the only hard numbers. `kernels/list?competition=...&sortBy=scoreDescending&pageSize=100&page=1..5`.

| # | ref | author | votes | last run | declared LB | kind |
|---|---|---|---|---|---|---|
| 1 | mattiaangeli/bend-the-knee-to-the-dinosaurs | Mattia Angeli | 186 | 2026-09-08 | 0.941 | INFER (shell) |
| 2 | maverickss26/rsna-knee-0941-restructured | maverick_ss_26 | 67 | 2026-09-08 | 0.941 | INFER (shell) |
| 3 | renta0426/rsna-knee-hybrid-raptor-meniscus-specialist | renta.k | 34 | 2026-09-08 | ~0.940 | INFER (shell) |
| 4 | vaihero/rsna-knee-rentafork | Vaibhav Gupta | 10 | 2026-09-08 | — | INFER (fork) |
| 5 | evgendvorkin/rsna-baseline | Дворкин Евгений | 100 | 2026-09-11 | — | INFER + write-up |
| 6 | renta0426/rsna-knee-hybrid-raptor-lateral-meniscus-r100 | renta.k | 36 | 2026-09-08 | 0.940 | INFER (shell) |
| 7 | manderson240/cohezion-rsna-knee-sota-ensemble | Mike Anderson | 9 | 2026-09-08 | — | INFER |
| 8 | ask812/bee-knees-fork-v1 | Arnav Kshirsagar | 3 | 2026-09-10 | — | INFER (fork) |
| 9 | prvsiyan/head-and-shoulders-knees-and-toes | prvsiyan | 189 | 2026-09-08 | 0.937 | INFER (shell) |
| 10 | romantamrazov/rsna-knee-dinosaur-v4 | Roman Tamrazov | 161 | 2026-09-07 | 0.936–0.937 | INFER (shell) |
| 11 | renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid | renta.k | 58 | 2026-09-04 | 0.937 | INFER |
| 12 | hyakumanben2025/rsna-knee-0937-meniscus-resid-repro | 百万遍2025 | 37 | 2026-09-04 | 0.937 | INFER |
| 13 | hyakumanben2025/rsna-knee-v6-outer-weight-overlay | 百万遍2025 | 2 | 2026-09-05 | — | INFER (probe) |
| 14 | kelvincheungkw/rsna-knee-0-937-weak-label-dinov2-meniscus-resid | Changhu1933 | 2 | 2026-09-05 | 0.937 | INFER (fork) |
| 15 | kaiwalyaatulraut/rsna-knee-competition-solution | KAIWALYA RAUT | 11 | 2026-09-06 | — | INFER |
| 16 | anhadmahajan06/rsna-knee-take-care-of-your-knee | Anhad Mahajan | 123 | 2026-09-05 | — | INFER (Keras) |
| 17 | anvithpothula/rsna-base | Anvith Pothula | 102 | 2026-08-25 | — | INFER |
| 18 | nishantkharga/rsna-knee-full-4-arm-ensemble-v55 | Nishant Kharga | 79 | 2026-09-09 | — | INFER |
| 19 | ayodejiibrahimlateef/rsna-knee-c01-dinosaur-v4-dual-fusion | Ayodeji | 8 | 2026-08-25 | — | INFER (fork) |
| 20 | lynnsakurai/rsna-knee-dinosaur-v4-5-validated-repro | Arlene | 8 | 2026-08-27 | — | INFER (fork) |
| 21 | tamerlanomralinov/rsna-knee-crazy-lb-tune | Tamerlan Omralinov | 50 | 2026-08-29 | — | INFER (tuning) |
| 22 | maverickss26/rsna-knee-restructured-version-2 | maverick_ss_26 | 11 | 2026-08-28 | — | INFER |
| 23 | junaid512/rsna-base-enc | Muhammad Junaid | 7 | 2026-09-08 | — | INFER (fork) |
| 24 | hyakumanben2025/rsna-knee-dinosaur-v5-repro | 百万遍2025 | 14 | 2026-08-30 | — | INFER (repro) |
| 25 | hyakumanben2025/rsna-knee-dinosaur-v6-repro | 百万遍2025 | 7 | 2026-08-31 | — | INFER (repro) |

Notable notebooks outside the top-25 window that carry explicit declared scores (title scan of 675 unique
refs across three sort orders; regex `0[._]?9\d\d` on titles):
`starkhushi/rsna-knee-0-940-source-checklist-4-diffs` (0.940), `yosukeinada/rsna-knee-0-937-22-submissions-6-lessons`
(0.937, **22-submission ledger with real LB scores — highest-value read in this report**),
`devzerone/vigil-rsna-knee-public-0-936-baseline` (0.936), `nathanjacob/4-arm-ensemble-explained-rsna-knee-0-937`
(0.937), `paiky1995/rsna-knee-0-926-lb-coatnet-fine-tune-blend` (0.926),
`dariushafshar/rsna-0-921-audit-before-you-exclude-a-target` (0.921),
`starkhushi/six-ensembling-ideas-that-failed-0-873-to-0-919` (0.919),
`sofiaanjenje/rsna-knee-frontier-0-916-reproduction` (0.916), `aadigupta7686/0-899-let-me-cook` (0.899).

**EDA / non-submission notebooks** (excluded from the ranking above): `romanrozen/rsna-knee-data-structure-eda-baseline`
(121 votes), `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` (359 votes — the official daily efficiency
leaderboard, ranked 312 in scoreDescending), `mpwolke/rsna-knee-abnormality-detection-2026-dcm`,
`hamdimohamed22/rsna-dataset-discovery-eda`, `karnakbaevarthur/rsna-knee-eda-to-2-5d`,
`barun2104/baseline-rsna-knee-mri-2-5d-resnet34`, `hansinigunasekara/eda-knee-abnormality-detection`.

---

## 2. What separates ≥0.945/0.941 from the rest

**Nothing in the public set is ≥0.945.** The highest public notebook is 0.941. The 0.9500 gold line is
**not** represented by any public notebook I could find — that gap is being held by private solutions.
This is important: there is no public 0.95 recipe to copy. Everything below is the 0.937→0.941 delta.

Deep-read of 5 highest-scoring notebooks (all are **inference-only shells** — the same
`mattiaangeli` pipeline family, differing only in constants). Full paths in `nb/*/DIGEST.py`.

### Shared 4-stage recipe (identical across all five)
1. **DINO branch** — 20 members (`4 seeds × 5 folds`) of DINOv2-small + `SlotHead`, from
   `pilkwang/rsna-knee-weights` (`manifest.json` + `.pt`). Per-member percentile rank × per-target
   `target_weight`. `CROP_MM=130`, `CACHE_IMG=336`, `GROUP=3`, `SLICE_BAND=(0.2,0.8)`, 6 slots,
   overlapping windows (`TTA_OVERLAP`). `prvsiyan`/`romantamrazov` **hard-assert `len(members)==20`**.
2. **"A5" branch** — 5 DINOv3 folds `m_f{0..4}.pt` from `mattiaangeli/knee-mri-fold-weights`
   (backbone `vit_small_patch16_dinov3.lvd1689m`), 6 slots × 16 slices @336px, `CROP_MM=130`,
   `SLICE_BAND=(0.12,0.88)`. Rank-mean over folds, then `A5_W = 0.45` against stage 1.
3. **RadImageNet branch** — torchvision ResNet-50 trunk (`ResNet50.pt`, 23,508,032 params) + `_RadHead`
   (LN 2048→512, plane/position embeddings, 8-head MHA over 12 label queries). Heads
   `v52_radimagenet_heads.pt` loaded **twice with two different SHAs** (twin E10 families) and
   `v52_e11_heads.pt` (which is really the **E13** bundle), each 5 fold-heads, `strict=True`.
   Encoder input `image.float().div_(127.5).sub_(1.0)`. Constants: `_RAD_E13_MEMBER_WEIGHT=0.50`,
   `_RAD_ALPHA=0.50`, `_RAD_TWIN_ALT_WEIGHT=0.500001`, `_RAD_V48_SECOND_ALPHA=0.15`, `_RAD_CAL_W=0.40`.
   **`_RAD_EXCLUDE=("Baker's","Fracture")`** — those two ship raw.
4. **CoAtNet "Raptor" branch** — 4 arms on `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` + `RaptorClassifier`
   (LayerNorm + tanh-attention MIL + per-label clsW/clsb). Checkpoints `raptor_ft_coatnet_v5_full_swa.pt`
   (**used twice** — straight + slice-reversed `xw.flip(1)`), `raptor_ft_coatnet_v10_full.pt`,
   `raptor_ft_coatnet_v8_full_swa.pt`. `CROP_MM=140`, `SPAN=(0.02,0.98)` (v8 `(0.06,0.94)`),
   `_SLOTS64`=64 slices / `K_EVAL=62`, `_SLOTS44`=44/42 for v8, 3-adjacent-slice windows, 2/98
   percentile per series, `cv2.INTER_AREA`, ImageNet norm. Arms mixed by **normalized weighted
   PROBABILITY mean** then one global percentile rank. Plus a **private residual-gated CoAt** (e4/e6/e8
   via `coat_resgated_ep10_top3_manifest.json`, run in a subprocess) at `private_alpha = 0.40`.
5. **Final fusion** = per-label weighted **rank** blend of transformer-stack vs Raptor arm, then
   `rank(method='average', pct=True)`.

### Per-notebook deltas (the actual discriminators)

| notebook | declared LB | Raptor arm weights (v5 / v10 / rev / v8) | inner 0.60/0.40 re-ranked? | outer per-target routing | extra |
|---|---|---|---|---|---|
| mattiaangeli/bend-the-knee-to-the-dinosaurs | 0.941 | **0.60 / 0.10 / 0.10 / 0.20** | **NO** | ACL .75, MM .80, **LM 1.00**, LOA .75, Fx .75 | 20-member fingerprint + SHA pinning; embedded 12×88 LR calibrator; time-budget member dropping |
| maverickss26/rsna-knee-0941-restructured | 0.941 | 0.55 / 0.10 / 0.15 / 0.20 | YES | same 5 | single `RUN` config dict; presets `probe22`/`parent`/`halfway`; arm-agreement + tie-block diagnostics |
| renta0426/…-hybrid-raptor-meniscus-specialist | ~0.940 | 0.55 / 0.10 / 0.15 / 0.20 | YES | same 5, re-asserted to 1e-12 | other 11 cols hard-gated bit-equal to 0.939 parent |
| prvsiyan/head-and-shoulders-knees-and-toes | 0.937 | 0.60 / 0.10 / 0.10 / 0.20 | NO | **LM 1.00 only** | full reproduction audit harness (`atol=1e-15` rebuild); only `len(members)==20` assert |
| romantamrazov/rsna-knee-dinosaur-v4 | 0.936–0.937 | 0.55 / 0.10 / 0.15 / 0.20 | YES | **flat 0.60 (no overrides)** | MM-only overlay `0.30·tr + 0.60·raptor + 0.10·bag`; Synovitis residual α=0.08; **no private CoAt bundle** |

**Key structural finding: the "meniscus specialist" in the 0.940 notebook's title is NOT a model.**
There is no meniscus-specific backbone, checkpoint, resolution, slot set or head anywhere in the source.
It is (i) the outer routing weight LM 1.00 (which sets transformer weight to 0 → that column becomes pure
Raptor before the final re-rank), (ii) `'max'` window pooling for both meniscus labels, (iii) exclusion of
both meniscus labels from the 0.40-weighted Rad calibrator gate. Do not go build a meniscus expert model.

**The 0.941-vs-0.937 list, ranked by evidence strength:**
1. **Per-target outer routing** — 5 overrides vs 1 (prvsiyan) or 0 (romantamrazov). This is the delta
   that the 0.937→0.940→0.941 ladder attributes most cleanly, and `maverickss26` prices LM alone at +0.002.
2. **The private residual-gated CoAt (e4/e6/e8) at 0.40 inside the Raptor arm** — present in all four
   ≥0.940 notebooks, deliberately absent in romantamrazov V4 (the only one at 0.936–0.937).
3. **Arm-weight vector — unresolved, A/B it.** mattiaangeli v30 + prvsiyan use `0.60/0.10/0.10/0.20`;
   maverickss26 + renta0426 + romantamrazov use `0.55/0.10/0.15/0.20`. `evgendvorkin`'s write-up
   attributes the 0.941 step to "maxspan-v5 = 0.60, reverse = 0.10" (the *first* vector) while
   `maverickss26` claims 0.941 with the *second*. The two sources contradict each other — treat as a
   genuine ±0.001 unknown, not a settled fact.
4. **Inner Raptor/CoAt re-rank** — applied by maverickss26 + renta0426, not by mattiaangeli v30 +
   prvsiyan. mattiaangeli's own metadata flags this as its only predictive change
   (`revision: "v29-remove-inner-rerank"`), so it is score-relevant, not cosmetic.
5. **Audit/verification scaffolding** — differs a lot but is score-neutral. Do not mistake it for recipe.

**Per-label preprocessing detail worth copying** (from mattiaangeli):
`PUBLIC_FRONTIER_TARGET_POOL = {'Fracture':'max','Contusion':'max','Medial Meniscus':'max',
'Lateral Meniscus':'max','ACL':'top2','MCL':'top2',"Baker's":'max'}`;
`TTA_TARGET_POOL` additionally `'Synovitis':'original_mean'`; `SLOT_PRIOR_STRENGTH=0.55`.
The Rad calibrator's gate is **7 labels**: ACL, Medial OA, Lateral OA, PF OA, Effusion, Baker's, Contusion
(cross-confirmed by an independently decoded base64 payload in prvsiyan — 12×88 `coef`, protocol columns
`n_series,n_Sag,n_Cor,n_Axi,Fat,Fat_Sag,Fat_Cor,Fat_Axi,Flu,Flu_Sag,Flu_Cor,Flu_Axi`, scale ≈0.2886751).

**Dead code present in all five — do not treat as recipe:** `LEGACY_BUNDLE_FILE='rsna_20260807_v1.pt'`,
`legacy_group_members()` returns `{}`, `combine_public_members_by_fold`,
`blend_legacy_frontier_and_soft`,
`LEGACY_MEMBER_WEIGHT_BY_TARGET = {'Lateral Meniscus': 15.0, 'Medial OA': 2.5, 'Lateral OA': 15.0, 'Contusion': 5.0}`,
`SLOT_PRIOR_*` (live only if a member cfg sets `prior=True`).

---

## 3. Discussions

**I could not read any discussion body.** `/api/v1/discussions/*`, `/forum/*`, `/topics/{id}`,
`/competitions/{slug}/topics/{id}` all return 404. The HTML page for
`https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304` returns HTTP 200
with a 5,780-byte shell containing **zero** occurrences of the post title, `__NEXT_DATA__`, `self.__next_f`,
`AUC`, `metric`, or `postBody` — it is fully client-rendered. `web_fetch` on the same URL returned only
the literal string "RSNA Knee Abnormality Detection | Kaggle". **Treat all discussion content as unverified.**

What the topic *list* does give: 20 of 108 topics
(`GET /api/v1/competitions/{slug}/topics`; pageSize, page, sortBy, offset, group params are all silently
ignored — every variant returned the identical 20 ids).

Sticky/organizer posts (all 4 are sticky):
| topic | votes | comments | date | url |
|---|---|---|---|---|
| Knee Abnormality Detection AI Challenge Overview | 69 | 11 | 2026-08-06 | /discussion/733343 |
| Use of Commercially Hosted LLMs | 18 | 12 | 2026-08-09 | /discussion/733965 |
| Welcome to Knee Abnormality Detection Challenge! | 16 | 1 | 2026-08-06 | /discussion/733375 |
| How to get started + Competition's Official Discord | 5 | 1 | 2026-07-30 | /discussion/730709 |

Non-sticky, by relevance to strategy:
| topic | votes | comments | date | url |
|---|---|---|---|---|
| **Best single-model score** | **44** | **68** | 2026-08-14 | /discussion/735304 |
| Two knees are better than one? | 8 | 5 | 2026-08-17 | /discussion/735639 |
| **Plateau at 0.903: sharing my pipeline and ablations** | 4 | 6 | 2026-09-10 | /discussion/740610 |
| Submission running for more than 2 hours | 3 | 5 | 2026-08-29 | /discussion/738129 |
| 9h runtime question | 0 | 3 | 2026-09-09 | /discussion/740474 |
| **One fold(No valid setup) model is better than Five f…** | 0 | 4 | 2026-09-09 | /discussion/740375 |
| **Things that helped (and didn't) when approaching RSNA** | 1 | 1 | 2026-09-08 | /discussion/740162 |
| From AlexNet to DINOv3: A Practical Backbone History | 1 | 0 | 2026-09-08 | /discussion/740164 |
| **Metric edge cases: single-class label columns, and c…** | 0 | 0 | 2026-09-08 | /discussion/740094 |
| Stuck around 0.74 OOF - what separates the 0.87 tier | 1 | 3 | 2026-08-31 | /discussion/738339 |
| Clarification on Manual Annotation of Unlabeled Train… | 0 | 1 | 2026-09-10 | /discussion/740718 |
| DOES Ensemble allowed in RSNA? | −1 | 0 | 2026-09-10 | /discussion/740624 |
| Scoring error - submission formatting | 1 | 4 | 2026-09-03 | /discussion/739200 |
| The leaderboard shows 3 decimals and I spent 20 subm… | 0 | 0 | 2026-09-09 | /discussion/740439 |
| Working with our own GPU | 1 | 11 | 2026-09-02 | /discussion/739000 |
| Question about Winner License and Apache 2.0 Pretrai… | 1 | 0 | 2026-09-07 | /discussion/739905 |

**Priority order to read in a browser (I could not):** 735304 (68 comments — the de-facto "what is the
best single model" thread), 740094 (metric edge cases / single-class columns — directly hits the macro-AUC
trap below), 740610 (ablations at 0.903), 740162 (what helped/didn't), 740375 (one-fold vs five-fold).

### Authoritative facts I *did* extract from the official pages
Retrieved via `GET /api/v1/competitions/{slug}/pages` (10 pages, HTML-stripped, saved as `comp_pages.txt`):

**Evaluation.** "Submissions are evaluated by the average area under the ROC curve … across the twelve
targets" — `Final Score = (1/12)·Σ AUC_i`, "the macro-averaged AUC ROC." Submission file: one confidence
score per study per label. Sample submission sets all labels to `0.5`.
→ **Macro-AUC trap:** a label that is constant in the 1,322-study hidden set contributes 0.5, dragging the
macro mean down by ~0.042 per dead label. The organizers mitigate but do not guarantee: "Although efforts
have been made to ensure each abnormality is represented in each dataset, **the prevalence of abnormalities
is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets**."
There is **no label that must be excluded** — all 12 are scored. Do not drop columns.

**Labels (12).** ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion,
Synovitis, Baker's, Contusion, Fracture.

**Test set.** "There are about 1300 studies in the test set. The `Report` field will not be provided at the
testing stage." (Your brief says 1,322 — consistent; the page says "about 1300".)

**Data.** "Only a small subset of training studies carry per-condition labels. We also provide the original
text of the radiology report from which you may wish to derive the labels for the remaining studies."
Reports are in **several languages**. Series metadata gives `Fluid_Sensitive`, `Fat_Suppression`,
`Anatomical_Plane`; series "typically contain 20–45 slices (median 30), with a long tail out to a few
hundred." DICOMs: mixed transfer syntaxes (Explicit VR LE, JPEG Lossless, JPEG 2000, Implicit VR LE),
stripped to an allowlist of 86 tags.

**Code-competition constraints.** CPU/GPU notebook **≤ 9 hours** (32400 s), **internet access disabled**,
"Freely & publicly available external data is allowed, including pre-trained models", submission file must
be named `submission.csv`.

**Timeline / prizes.** Start 2026-07-30; Entry + Team Merger Deadline **2026-10-15**; Final Submission
**2026-10-22**; Winners' Requirement 2026-11-05. Main LB $9,000/$7,000/$6,500 then $6,000, $5,500, and
five × $5,000 (top 10 paid). Separate **Efficiency Track** $7,000/$6,000/$5,000, scored by
`Efficiency = AUC/(Benchmark − maxAUC) + RuntimeSeconds/32400` (minimize), eligible only for submissions
you select for the Leaderboard Prize.

---

## 4. Public datasets carrying weights / OOF / pseudo-labels

**132 unique dataset refs retrieved.** First pass: 94 (`datasets/list?search=…&pageSize=100`, 10 search
terms + per-owner listing; `datasets_raw.json`). Second pass: **+38 more** from 11 additional terms
(`rsna-knee-2`, `knee checkpoints`, `knee oof`, `knee pseudo`, `coatnet knee`, `knee dinov3`, etc.;
`datasets_raw2.json`). **The first pass was a ceiling, not a floor** — the search endpoint's coverage is
term-dependent, so treat any list built from a fixed term set as incomplete.
Below: the weight/OOF/pseudo-label-bearing ones. Sizes from `totalBytes`; votes/`lastUpdated` from
`datasets/list` and `datasets/view` (endpoints occasionally disagree on size — noted where seen).

### ⚠ NOT IN OUR CURRENT LIST — highest priority
| ref | MB | updated | votes | appears to contain |
|---|---|---|---|---|
| **tonylica/rsna-knee-bend-dinov3-0917-repro-assets** | 2502 | 2026-08-17 | 1 | **One-mount bundle of all weights**: `rsna-knee-weights/manifest.json`, `knee-mri-fold-weights/`, `resnet-50-radimagenet-marwan/ResNet50.pt`, `rsna-knee-e9-radimagenet-heads-v15/v52_radimagenet_heads.pt`, `kernel-sources/rsna-knee-e13-train/rsna_rad_e11/v52_e11_heads.pt`. Confirmed by romantamrazov's source reading exactly these paths. **Used by 37/60 top kernels.** |
| **dreaddevelopment/raptor-knee-widedense** | 518 | 2026-08-23 | **70** | "Trained CoAtNet weights that score twelve findings on a knee MRI study" — Raptor arm. **Used by 38/60.** |
| **dreaddevelopment/raptor-knee-arms** | 991 | 2026-08-20 | 5 | "Three knee MRI backbones, kept together for rank-mean ensembling" — **used by 7/60**. |
| **dreaddevelopment/raptor-knee-arms-x** | 259 | 2026-08-21 | 4 | "Single CoAtNet 384 knee MRI arm, **0.917 on the public leaderboard**" — used by 7/60. |
| **stevenleehans/rsna-knee-llm-report-labels** | 0.4 | 2026-08-09 | **46** | Pseudo-labels. Subtitle: "LLM-read labels for all 4,407 studies: **0.893 vs gold, where regex scores 0.814**". Used by 19/60 — a materially better teacher than `pilkwang/rsna-knee-llm-labels`. |
| **tonylica/rsna2026-models** | 1366 | 2026-08-07 | 1 | Model bundle — used by 18/60. |
| **mattiaangeli/rsna-knee-radimagenet-foldsv1-heads** | 56 | 2026-08-12 | 1 | RadImageNet fold heads — used by 18/60. |
| **lixin73/rsna-knee-llm-report-labels-sol56** | 0.3 | 2026-08-08 | 10 | Alternative LLM labels (GPT-5.6-Sol) — used by 5/60. |
| **dreaddevelopment/raptor-knee-finespacing** | 259 | 2026-08-23 | 3 | "CoAtNet fine-tuned for the 12 knee findings on the Fine Spacing slice sampling" — a 5th Raptor arm candidate. Used by 5/60. |
| **renta0426/rsna-knee-public0033-meniscus-bag-v1** | 308 | 2026-09-02 | 0 | Meniscus weak-label bag; consumed via the `public0033_meniscus10_bundle_v1` runtime in romantamrazov (MM overlay weight 0.10). Used by 4/60. Private byte-identical mirror: `mattiaangeli/rsna-knee-public0033-meniscus-bag-mirror`. |
| **royceplayz/rsna-medicalnet-resnet50-weights** | 353 | 2025-10-09 | 0 | MedicalNet ResNet-50 pretrained — used by 5/60. Another *differently-failing* encoder. |
| **prvsiyan/orthodiffusion-official-knee-mri-foundation** | 1582 | 2026-08-12 | 0 | OrthoDiffusion knee-MRI foundation weights — used by 6/60. |
| **dreaddevelopment/raptor-knee-fullspan** | 259 | 2026-08-22 | 0 | CoAtNet, full span sampling — another arm candidate. |
| **dreaddevelopment/raptor-knee-widefov** | 259 | 2026-08-22 | 0 | CoAtNet, wide FOV — another arm candidate. |
| **damoonshahhosseini/ct02-teacher-blend-coatnet-checkpoint** | 518 | 2026-09-07 | 0 | "Private CoAtNet weights for RSNA Knee CT02 teacher-blend t…" — newest, unvetted. |
| **lixin73/rsna-knee-train-checkpoints** | 79 | 2026-08-15 | 2 | Train checkpoints — used by 3/60. |
| **zaidaliiq1000/rsna-knee-ens55-weights** | 155 | 2026-08-14 | 0 | 5-fold ensemble weights. |
| **shotafrancerat/rsna-knee-rankmil-v10-models** | 198 | 2026-09-06 | 1 | RankMIL v10 models. |
| **jayabaya01/rsna-knee-weights-v5-and-earlier** | 969 | 2026-09-02 | 1 | Superset of `pilkwang/rsna-knee-weights` + v5 and earlier. |
| **stevenleehans/rsna-knee-champ-members-only** | 395 | 2026-08-15 | 0 | "champ arm, 5 members (rules targets)". |
| **stevenleehans/rsna-knee-llm199-e30-members** | 395 | 2026-08-15 | 0 | "llm199 e30 arm, 5 members (LLM targets)". |
| **octaviograu/rsna-knee-wide336-frozen-heads-v1** | 135 | 2026-09-04 | 2 | Wide-336 frozen heads. |
| **nursrijan/rsna-knee-dinov2-base-llm-models** | 616 | 2026-08-18 | 0 | DINOv2-base + LLM models. |
| **mikebelliveau/rsna-knee-public-weights** | 134 | 2026-08-27 | 0 | Public weights bundle. |
| **matteogiannetto/rsna-knee-i15-final-inference-assets** | 159 | 2026-09-01 | 0 | Final inference assets. |
| **ericwang03/rsna-knee-dinov2-mil-bundle** | 83 | 2026-08-09 | 0 | "deep-MIL fold checkpoints plus DINOv2 weights". |
| **hafidabelayd/rsna-knee-v10-orthofoundation-heads** | 56 | 2026-08-10 | 2 | OrthoFoundation heads. |
| **prvsiyan/rsna-knee-pulse-v6-checkpoints-public-20260813** | 453 | 2026-08-13 | 0 | "Fifteen authenticated train-only ResNet-18 fold checkpoints". |
| **prvsiyan/rsna-knee-v52-oof-receipt-20260810** | 0.8 | 2026-08-12 | 0 | **OOF predictions receipt** — the only explicit OOF artifact found. Useful for weight-fitting audits. |
| **prvsiyan/rsna-knee-two-teacher-bundle-v1** | 8 | 2026-08-12 | 1 | Two-teacher distillation bundle. |
| **prvsiyan/rsna-knee-b3-v47-public-deployment** | 213 | 2026-08-13 | 0 | B3 V47 deployment weights. |
| **chowagent/rsna-m4-ckpt** | 393 | 2026-08-22 | 0 | "DINOv2 vits14 + vitb14 weights, trimmed repo, M4 KneeNet c…". |
| **zaidaliiq1000/rsna-knee-backbone** | 385 | 2026-08-10 | 0 | Offline DINOv2 small+base. |
| **zaidaliiq1000/rsna-knee-ft3-weights** | 243 | 2026-08-16 | 1 | FT3 weights. |
| **ishan1989/rsna-knee-5fold-weights** | 382 | 2026-08-19 | 0 | 5-fold EfficientNetV2-S. |
| **sahil00001/rsna-knee-phase14-weights** | 62 | 2026-08-19 | 0 | Phase14 weights. |
| **starkhushi/rsna-knee-v6f1-weights** / **-v6f2-weights** | 80 each | 2026-08-26/27 | 0 | Per-fold v6 weights. |
| **dreaddevelopment/rsna-knee-model** | 16 | 2026-08-27 | 0 | Small model artifact. |
| **dreaddevelopment/raptor-cnn336** | 176 | 2026-08-16 | 0 | "ConvNeXt-Small 336 knee MRI arm, an early baseline" — used by 3/60 as a fallback path. |

### Pseudo-label / soft-target datasets NOT in our list
| ref | MB | updated | votes | notes |
|---|---|---|---|---|
| stevenleehans/rsna-knee-llm-report-labels | 0.4 | 2026-08-09 | 46 | "0.893 vs gold, where regex scores 0.814" |
| lixin73/rsna-knee-llm-report-labels-sol56 | 0.3 | 2026-08-08 | 10 | GPT-5.6-Sol reader |
| laymond/rsna-knee-abnormality-qwen3-8b-weak-labels | 6 | 2026-09-08 | 3 | "Qwen2.5-32B v2 weak labels; macro ROC AUC **0.8686**; Qwen3-8B …" |
| yunusgmsoy/rsna-knee-llm-labels-4-source-merged | 0.3 | 2026-08-23 | 1 | 4-way consensus (Gemini + StevenLeeHans + Pilkwang + …) |
| yunusgmsoy/rsna-knee-abnormality-3-source-merged-labels | 0.2 | 2026-08-20 | 1 | triply-cross-validated |
| yunusgmsoy/rsna-knee-llm-labels-2-source-merged | 0.2 | 2026-08-20 | 1 | Pilkwang + StevenLeeHans merge |
| achelijndiamantidis/knee-phase1-artifacts | 2 | 2026-08-20 | 1 | weak labels for 4,407 studies, ten languages |
| dreaddevelopment/rsna-knee-labels | 0.3 | 2026-08-14 | 0 | per-study soft probabilities |
| rayanbabur/rsna-knee-2026-calibrated-soft-targets | 0.4 | 2026-08-21 | 1 | "**0.8991 Gold AUC** Soft Targets & Weights" |
| barun2104/rsna-knee-stratified-folds-and-llm-soft-labels | 3 | 2026-08-07 | 0 | 5-fold stratified CV split + LLM soft labels |
| malikahsan/knee-mri-reports-translated-transcribed | 2 | 2026-08-22 | 1 | language detection / translation |
| pilkwang/rsna-knee-report-labels | 0.2 | 2026-08-06 | 0 | regex-extracted report labels |

### Datasets already in our list (confirmed present, for completeness)
`pilkwang/rsna-knee-weights` (1580 MB, 4 votes), `mattiaangeli/knee-mri-fold-weights` (419 MB, 1 vote),
`dreaddevelopment/raptor-knee-maxspan` (259 MB, 26 votes), `dreaddevelopment/raptor-knee-native384`
(8 votes), `dreaddevelopment/raptor-knee-native384dense` (5 votes),
`marwanmath/resnet-50-radimagenet-marwan` (83 MB), `antoinegg1/rsna-knee-e11-diverse-heads-v20`,
`antoinegg1/rsna-knee-e9-radimagenet-heads-v15`, `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`,
`pilkwang/rsna-knee-llm-labels` (16 votes), `mattiaangeli/rsna-knee-coat-resgated-ep10-top3`,
`mattiaangeli/opencv-python-headless-4120088-x86`.
*(`mattiaangeli/rsna-knee-public0033` returns HTTP 403 — private or removed; the `-meniscus-bag-mirror`
is the accessible byte-identical copy.)*

### Training-corpus datasets (large, for retraining if ever needed)
`barun2104/rsna-knee-mri-processed-3d-volumes` (16.6 GB, 34 votes, "24,371 normalized uint8 3D MRI volume
arrays (.npz)"), `dreaddevelopment/knee-raptor-corpus` (11.4 GB, 27 votes) + `-ext` (4.3 GB, 28 votes),
`alenic/rsna-knee-abnormality-detection-jpeg-224x224` (10.7 GB, 22 votes, pre-resized JPEGs).
Given finding 2 (no training needed), these are lower priority — but the JPEG-224 dataset is a cheap
way to iterate.

### OOF / prediction artifacts (second pass) — use these to fit weights honestly
| ref | MB | votes | appears to contain |
|---|---|---|---|
| **denisrukavishnikov/rnsa-knee-pred-public20-oof-audit** | 9.0 | 0 | OOF predictions + audit for the public-20 model. **Best lead for a full-population weight fit.** |
| **nekkon/rsna-knee-image-oof-gold** | 0.0 | 1 | "image OOF gold" — tiny, so a CSV/JSON table of predictions or scores, not weights |
| hardiksoni7614/rsna-knee-pretrained-output | 198.3 | 1 | "pretrained output" — predictions/logits, unclear which |
| prvsiyan/rsna-knee-v52-oof-receipt-20260810 | 0.3 | 0 | V52 OOF receipt |

> My earlier claim that prvsiyan's receipt was "the only explicit OOF artifact" was **wrong**; there are
> at least four. Fitting the outer routing weights and the exclude-list decision on the full 4,407
> study population instead of the 58 gold studies is the highest-value process fix available — it
> directly addresses the documented −0.004 forecast error and the unrefereeable-CI problem.

### CoAtNet / alternative backbone weights (second pass)
| ref | MB | votes | appears to contain |
|---|---|---|---|
| **diversen/timm-coatnet-rmlp-2-rw-384-pretrained** | 261.6 | 0 | The exact `coatnet_rmlp_2_rw_384` ImageNet weights the Raptor arm is built on. **Cheap insurance** against a timm/docker change breaking a commit with internet disabled. |
| **pritom2357/orthofoundation-l-dinov3** | 1074.4 | 0 | OrthoFoundation-L DINOv3 — a knee-specific, genuinely differently-failing encoder |
| surasan092/rsna-knee-dinov3-s-v13-1-gpu-util-full-ft-model | 1065.7 | 0 | DINOv3-S full fine-tune |
| myominhtet/rsna-knee-coat5fold | 1296.3 | 0 | 5-fold CoAtNet |
| tannguyenngoc/rsna-knee-eva-02b-2-5d-mil | 305.3 | 0 | EVA-02b 2.5D MIL — non-DINO, non-CoAt architecture |
| stefanoblando/rsna-knee-stage1-checkpoints / -stage2- / -stage3- | 259 / 259 / 518 | 0/0/1 | Staged checkpoint releases |
| shingo257/rsna-knee-trained-checkpoints-v1 | 1278.5 | 0 | Trained checkpoints |
| arav57/knee-mil-checkpoints | 400.3 | 0 | MIL checkpoint pool |
| phoenixdroid/rsna-knee-models | 80.4 | 0 | Model bundle |
| tomdifiore/rsna-knee-credible-checkpoints | 119.6 | 0 | Checkpoint pool |
| prvsiyan/rsna-knee-resnet18-student-v1 | 66.5 | 0 | "CPU-ready knee MRI student weights" |
| mfarreladitya/rsna-knee-grandmaster-checkpoints | 0.1 | 0 | Tiny — likely a manifest/index, not real weights |
| lorenzo26/rsna-knee-src-v1 | 101.1 | 0 | "Codice (wheel) + pesi ImageNet + subset v1" |

> Given the evidence that **disagreement beats member quality**, the diversity candidates worth a cheap
> probe are EVA-02b MIL and DINOv3-S full-FT — **not** another DINOv2 fold.

### Additional label / pseudo-label resources (second pass)
| ref | MB | votes | notes |
|---|---|---|---|
| **tsuyu122/rsna-knee-mri-balanced-labels-public** | 1.3 | 0 | "Harmonized **8,631** Knee MRI Studies & 12 Labels across RSNA, M…" — ~2× the 4,407 competition training set, i.e. an external weak-label pool |
| **flight0234/rsna-knee-hybrid-report-labels** | 0.2 | 0 | "v4_blend labels with **v2 Fracture column**; gold macro 0.899" — a specifically corrected Fracture column, directly relevant to the `_RAD_EXCLUDE` finding |
| tasmeemreza/rsna-knee-refined-llm-labels | 0.1 | 0 | "Corrected-final labels: prompt fixes + two-model-agreement" |
| zaidaliiq1000/rsna-knee-llm-labels | 0.2 | 0 | "LLM multi-model fused labels (mean-sev) + gold labels" |
| uddipanb/ensemble-labels-for-rsna-knees | 0.2 | 0 | Ensemble labels |
| jezrealmomoh/rsna-knee-pseudo-labels | 1.6 | 1 | "HAIM_Matrix_Model_v1" |
| giovannicr/rodilla-pseudo-labels | 0.6 | 0 | Pseudo-labels |
| freshtime/rsna-knee-pseudo-labels-betta | 0.1 | 0 | Pseudo-labels |
| **dariushafshar/rsna-knee-2026-grouped-cv-folds** | 0.2 | 0 | "Seeded 5-fold study-grouped CV split + series metadata" |
| **flight0234/rsna-knee-dual-grouped-folds** | 0.1 | 0 | "5-fold split guarding **both the shared-report and shared-scan**…" — implies naive splits leak |

### Additional large preprocessed corpora (only if retraining; you should not)
`mansianil/rsna-knee-part-2` (42.5 GB) + `-mega-part-2-1786952420` (34.7 GB) + `-part-2-1787591966`
(8.5 GB); `bigsbds/rsna-knee-384-b2` (15.9 GB — "Standardized **384x384** volumes container batch 2
(5705 series)" — matches the Raptor arm's native resolution).

### Raptor arm-weight usage frequency across top-60 kernels (dataset attach counts)
```
56  dreaddevelopment/raptor-knee-maxspan
38  dreaddevelopment/raptor-knee-widedense        <-- NOT in our list
37  tonylica/rsna-knee-bend-dinov3-0917-repro-assets <-- NOT in our list
33  mattiaangeli/knee-mri-fold-weights
33  marwanmath/resnet-50-radimagenet-marwan
33  antoinegg1/rsna-knee-e9-radimagenet-heads-v15
33  prvsiyan/rsna-knee-v52-radimagenet-heads-20260812
33  pilkwang/rsna-knee-weights
28  pilkwang/rsna-knee-llm-labels
24  antoinegg1/rsna-knee-e11-diverse-heads-v20
19  stevenleehans/rsna-knee-llm-report-labels       <-- NOT in our list
18  mattiaangeli/rsna-knee-radimagenet-foldsv1-heads <-- NOT in our list
18  tonylica/rsna2026-models                        <-- NOT in our list
16  dreaddevelopment/raptor-knee-native384 / -native384dense
```

### Kernel sources used by top-60 kernels (beyond e11/e13)
```
23  sofiaanjenje/rsna-knee-e11-train      (already used)
23  sofiaanjenje/rsna-knee-e13-train      (already used)
16  romantamrazov/rsna-knee-dinosaur-v4-train   <-- NOT used by us
 3  nishantkharga/rsna-knee-v16-slothead-train-nk <-- NOT used by us
```

---

## 5. Submission limits

Two independent sources, both retrieved today:

**A. `GET /api/v1/competitions/list?search=rsna-knee-abnormality-detection`** returned, verbatim:
```
maxDailySubmissions = 5
isKernelsSubmissionsOnly = True
evaluationMetric = Roc Auc Score
maxTeamSize = 5
teamCount = 3519
mergerDeadline = 2026-10-15T23:59:00Z
newEntrantDeadline = 2026-10-15T23:59:00Z
deadline = 2026-10-22T23:59:00Z
enabledDate = 2026-08-05T15:51:31.833Z
awardsPoints = True
submissionsDisabled = False
userRank = 2096
```

**B. Competition rules page** (`GET /api/v1/competitions/{slug}/pages`, page 0, section "2. SUBMISSION LIMITS"):
> "a. You may submit a maximum of five (5) Submissions per day.
> b. You may select up to two (2) Final Submissions for judging."

**Answers: 5 submissions per day. Yes — you may select up to 2 final submissions** for judging from among
your submissions. Also from the rules: team mergers are allowed up to the Team Merger Deadline (2026-10-15)
provided the merged team's total submission count is within the limit; ties are broken in favour of the
earlier submission; and winners must deliver training code, weights (as a public Kaggle dataset), a video,
and a method description by 2026-11-05.

Practical consequence: at 5/day from today (2026-09-11) to 2026-10-22 you have ~41 days ≈ **~205 submissions**.
That is enough to A/B the four unresolved constants (arm-weight vector, inner re-rank, the two exclude-list
variants, routing values) empirically rather than by argument — provided each submission isolates one variable.
