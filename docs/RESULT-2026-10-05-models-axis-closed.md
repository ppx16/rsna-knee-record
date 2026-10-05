# RESULT 2026-10-05 — the Kaggle Models axis: the shot was taken, and it is CLOSED

**Zero GPU spent. The candidate member was measured, not guessed.**

---

## 1. What the axis was

`FIND-2026-10-05-external-data-closed-and-the-models-axis.md` found that our kernels mount exactly one
Kaggle **Model** (`metaresearch/dinov2/PyTorch/small/1`, a *generic* DINOv2) and **zero
competition-specific models**, while a whole ecosystem of them exists. That was the one lever which is
simultaneously permitted (the rules explicitly allow pre-trained models), mountable with internet off,
and structurally different — the "somebody else's model" the strategy document named.

## 2. Three candidates, and two of them cannot be used responsibly

| Model | files | verdict |
|---|---|---|
| `ruicompany/rsna-knee-a5-rad` | `a5/m_f0..4.pt`, `rad/ResNet50.pt`, `rad/v52_radimagenet_heads.pt` | ⛔ **a repackage of three datasets we already mount** (`mattiaangeli/knee-mri-fold-weights`, `marwanmath/resnet-50-radimagenet-marwan`, `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`). **Zero new information** — the "same corpus, laundered" trap. |
| `yaolinghan/rsna-knee-pool-models` | `hunter_pool_base_best.pt` (351 MB), `hunter_pool_v1/v2_best.pt` | ⛔ ConvNeXtV2, genuinely a different family — **but its entire model card is one sentence**: *"ConvNeXtV2 tiny/base 2.5D study-level pooling models trained on RSNA Knee 2026 data (weak labels)."* **No slot layout, no span, no crop, no img, no k_eval, no arch string.** Wiring it means guessing the author's preprocessing — the silent-failure class the meniscus arm flagged. |
| `yunusgmsoy/rsna-knee-radhead-resnet50` | `v52_rad_heads_yunus.pt` + **`v52_oof_yunus.csv`** | ✅ fully documented **and it ships its own OOF** — see §3 |

⚠️ **The second row is the honest limit of this axis.** An undocumented foreign model is not
"cheap to try"; it is an integration project whose failure mode is *plausible-looking wrong
predictions*, with no gate in this repository able to see it. **A model card one sentence long is not
an invitation.**

## 3. The one that could be measured — and the measurement closes it

`yunusgmsoy/rsna-knee-radhead-resnet50` documents its geometry completely (RadImageNet ResNet-50
encoder, frozen; `RadHead` = LayerNorm + Linear(2048→512) + **3 plane embeddings** + **8 slice-position
embeddings** + MHCA + dual residual pooling — i.e. **our `_RAD_E11/E13` geometry**: 3 slots, 8 slices,
224 px, 130 mm), and it ships a **4407-row OOF matrix**.

Its card claims **0.8251 on the 58-gold set**. ⚠️ gold-58's bootstrap CI is **±0.021**, so that number
cannot gate anything. So the OOF was scored **on the full study set against the same V5 label source the
card says it trained on**, which is the convention the repo's one-convention table already uses
(`work/score_candidate_member.py`):

```
MCL                0.7417      Medial Meniscus    0.8348
Lateral Meniscus   0.7418      Medial OA          0.8427
ACL                0.7493      Fracture           0.8583
Lateral OA         0.7902
PF OA              0.7927      MACRO, 12 labels, n=4407 = 0.7992
Baker's            0.7956
Effusion           0.7984
Contusion          0.8163
Synovitis          0.8285
```

**0.7992**, against our existing members:

| member | macro (same convention) |
|---|---|
| `surrogate_radfree` | 0.8960 |
| `msft2` | 0.8847 |
| `d64cn0` | 0.8679 |
| `cn0` | 0.8631 |
| **candidate** | **0.7992** |

⇒ **It is 0.05 below the *weakest* member already in the stack and 0.10 below the A167 member gate
(≈ 0.90).** And note *where* it is worst: **MCL (0.7417) and Lateral Meniscus (0.7418)** — the labels
this repository has independently identified as its hard cases.

**⇒ The Models axis is closed: the only candidate that could be evaluated fails decisively, on 4407
studies rather than 58, before a single GPU-second was spent.**

## 4. What this does and does not establish

**Establishes:** the competition-specific Kaggle Models are either repackages of what we already mount,
undocumented to the point of being unwireable, or measurably far below our member gate.

**Does not establish:** that *all* foreign models are useless. It establishes that **the ones on the
Kaggle Models hub for this competition are** — which is the set we could actually mount with internet
disabled. The general lever ("somebody else's model") is not refuted; the available instances of it are.

⚠️ **And the method point is the reusable part:** this axis looked like a one-arm GPU experiment, and it
was settled instead by downloading one CSV and scoring it on our own convention. **Before spending an
accelerator on a member, check whether its author already published the OOF** — a shipped OOF is a free,
full-population measurement of exactly the thing the arm would have been run to find out.
