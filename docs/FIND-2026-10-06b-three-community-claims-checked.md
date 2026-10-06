# FIND 2026-10-06 (b) — three community claims checked against artefacts; two were labels without checkpoints

Autonomous iteration while `56872230` (the reader leg) is pending. Three things were followed up from the
public sweep and all three were resolved by looking at **files**, not at notebook titles.

---

## 1. ⚠️ "4-Way Consensus" is a label nobody can run — the 4th CoAt member has no published checkpoint

Multiple public notebooks -- including the runtime benchmark table and every "0.943 SOTA" fork -- describe
the stack's CoAt branch as *"4-Way Consensus: Residual + D4 + Global96 + **Repair-v1** (25% each)"*, and
`chenchinglong/rsna-kad-bench-jiweiliu` lists it as a fused member. The code is real and detailed: a child
runner `_run_repairv1_reader`, a pinned manifest hash `RV1_MAN_SHA`, outputs `_coat_repairv1.csv` /
`coatnet_repairv1_top3_predictions.npz`, and graceful-failure events
(`coat_repairv1_child_failed`, `coat_repairv1_fallback_studies`).

**Our stack has three CoAt members and no Repair-v1 at all** — its own log prints
`[coat-arm] CoAt family members: resgated_top3, global96_top3, d4_swa3`. So this looked like a concrete
missing lever.

**It is not.** Checked against artefacts:

| check | result |
|---|---|
| Kaggle dataset search, 6 queries (`repair`, `coatnet repairv1`, …) | **no dataset anywhere has "repair" in its slug or title** |
| the three CoAt datasets our arm mounts, full file listings | **zero files matching `repair`** |

⇒ **The checkpoint does not exist publicly.** The notebooks configure the member, pin a hash for it, and
carry the fallback for when it cannot load — and then fall back. **The community's headline "4-Way
Consensus" describes a 3-member family in practice, which is exactly what we already run.**

⚠️ **This is the `F_dim=768` disease at scale**: a documented, hashed, event-instrumented capability with
no artefact behind it, repeated across a dozen public notebooks and a benchmark table. **Our stack is not
missing anything; it is at parity.** (And `qierroo/rsna-knee-public-blend-drop-repair-v1`'s extra token
`drop-repairv1` is an A/B **mode name**, not an action — the reader still runs; the mode is logged as
`coat_family_candidate_mode`.)

## 2. ✅ The runtime scare was wrong: the 0.943 stack is a ~30-minute run, not 7 hours

`goodpjw2008` states the stack needs **6.5–8 h of the 9 h cap**, which I had recorded as "the real
constraint" on every arm. Two independent sources contradict it, and they are specific:

```
(pjmathematician's d4-lite / the SOTA fork benchmark table)
  Raptor Baseline    0.924   1 CoAtNet single-pass          ~9 hours
  DINOsaur Blend     0.941   DINOv2 + Raptor MaxSpan        ~4.5 hours
  Speedy Raptors v7  0.942   Residual Top-3 + D4 (50/50)    ~1.5 hours
  v9 / v34           0.943   4-Way Consensus (25% each)     < 30 mins
```

and `qierroo`/`tensorhydra`: *"sub-30 minute inference on the entire hidden test set"*.

⇒ **The timeout risk that shaped several decisions this session is largely absent.** It also explains the
user's own "26m" observation. ⚠️ `goodpjw2008`'s Part A is a *reproduction* of the stack (skarin's), which
appears to be the slow implementation — a reminder that two public sources describing "the same stack" can
differ by an order of magnitude in runtime.

## 3. Three newly-surfaced CoAtNet checkpoints, catalogued and two rejected

A dataset search for CoAt checkpoints found four we had never catalogued. Inspected by state dict, not by
title (`work/probe_coat_ckpts.py`, `work/check_coat_geometry.py`):

| dataset | verdict |
|---|---|
| `kawaharataishi/rsna-knee-coatnet-distilled-checkpoint` | ✅ **same family** (`coatnet_rmlp_2_rw_384`, res 384, 486 tensors, head `clsW (12,1024)` + `backbone.norm`). Claims `auc 0.9231` — **higher than our msft2's gold-58 0.9069**. ⚠️ **But no geometry documentation, and its author has NO knee notebook** (their kernels are all other competitions), so the `auc` key's meaning and the training sampling are both unknown. **Wiring it means guessing the preprocessing — rejected on that basis.** It stays catalogued, not used. |
| `tannguyenngoc/rsna-knee-coatnet-a100-best` | ✅ same family (488 tensors) but **`gold_auc: None`, `val_auc: nan`, `n_val_auc_classes: 0`** — it never computed any validation AUC, and its `input_mean/std = [0.5, 0.5]` differs from our norm. **No evidence of quality — rejected.** |
| `myominhtet/rsna-knee-coat5fold` | ❌ different family (`enc.` prefixes, head `att.0.weight (128, 512)`) — needs its own loader and geometry. |
| `haideptry/rsna-knee-coatnet-25d-masked-checkpoint` | ❌ different (smaller) model, 768-dim tail, `cls_weight`/`cls_bias` naming. |

⚠️ The pattern in rows 1 and 2 is worth naming: **a checkpoint being loadable is not the same as being
usable.** Our loader can load both; neither has a documented geometry, and a wrong geometry produces
plausible wrong predictions that no gate in this repository can see. **The `hunter_pool` lesson holds.**

## 4. What was NOT pursued, and why

* **`kiritohayate95/rsna-knee-weak-label`** — an honest out-of-fold comparison of six weak-label sources
  and every subset. Attractive because it is exactly the measurement I wanted after finding
  `report_labels_v5` has the 58 gold baked in. **But the axis is closed by `FACTS` D4** (label-function
  changes: severity 0.6384 solo, stacking degraded 0.8991 → 0.8464, qwen3:30b 0.8611 ≈ rule 0.8617), and
  my own clean-table test did not reproduce the premise (0.8991 labels vs 0.8968 model). **Not reopened.**
* **`lnifold`'s loss weighting (W=3.0 on the 58 gold)** — genuinely not in D4, but it requires **training
  on the 58 gold**, which would destroy the only clean holdout this repository has, and it is adjacent to
  D19 (full-data retrain, unreadable at ±0.021). **Recorded, not pursued.**

## 5. The decision tree for `56872230`, pre-committed

Written now so the next step is mechanical rather than a fresh judgement:

| reader-leg result (in-block percentile vs **16.03 %**, margin vs **−118**) | action |
|---|---|
| **≥ 3 points better** | the composition works → the `0943cfmsft2-rdr` becomes the primary candidate; next iterate on the leg itself (the community's Stack++ variant upgrades the reader's aggregation to a fold × TTA **rank** ensemble and is unverified) |
| **within ±3 points** | no measurable change → **do not select it**; the pair stays `56814081` + `56599932` |
| **worse** | composition is harmful → drop the leg; record, and stop adding members |

⚠️ **The displayed score is not the criterion** — the author's own pipeline scored 0.944 / 0.944 / 0.943
across three identical runs, so landing on 0.943 refutes nothing.
