# RESULT 2026-10-01 — exp085 gate: the FORWARD IS RIGHT (order A), the slice policy is the remaining gap

The CPU gate ran to **COMPLETE** (exit 0, no failure message). It answers the three questions it was
built for, and it answers the most important one **yes**.

## 1. What ran, and what it proves

```
[env] torch 2.10.0+cpu | cuda False
[env] transformers 5.0.0
[data] ckpt /kaggle/input/datasets/t88take/rsna-knee-t88-weights/exp085/fold0.pt (361 MB)
[data] volumes (4407, 44, 336, 336) dtype=uint8 range=[0.000, 255.000] | ids 4407
[data] GOLD set = 58 studies
[model] MISSING=0 UNEXPECTED=0  ([], [])
[model] strict=True load PASS
[data] depth 44 slices, central 0.2-0.8 -> indices [9, 11, 14, 16, 18, 20, 23, 25, 27, 29, 32, 34]
[data] intensity scale = /255 (observed max 255.000)
```

* **the reconstruction loads strictly on Kaggle too** — 0 missing, 0 unexpected — so the module tree
  recovered locally from the checkpoint alone is correct on the real runtime as well;
* **the gold set is 58 studies**, i.e. the same `gold58` the author's manifest quotes, so the comparison
  is in the right units;
* the corpus is `(4407, 44, 336, 336)` **uint8 0–255**, which the kernel decided from the data rather
  than assuming (the `/255` is logged with the observed max).

## 2. ⭐ The answer: forward order **A** is correct

| crop | order A | order B | order C |
|---|---|---|---|
| 336 (full frame) | **0.7929** | 0.6827 | 0.6642 |
| 262 (centre crop) | **0.7997** | 0.6776 | 0.6642 |

**A wrong forward pass would read ≈0.5 — scrambled logits cannot rank labels.** A reads **0.80**, a
~0.13 margin over B and C, and the ordering is stable across both crop scales. ⇒ **the per-finding
classifier reading is right**: `logit_f = head.out.weight[f] · proj(pooled)[f] + head.out.bias[f]`,
i.e. each of the 12 findings is classified by its own row of the read-out, applied to the vector that
the depth-conditioned slice attention produced for *that finding*.

That was the one thing shapes could not determine, and it is now measured rather than guessed.

## 3. ⚠️ The remaining gap is the SLICE POLICY: 0.7997 against the author's 0.9062

The gap is **0.106** and it is *not* the head — the head works. It is the input construction, and the
checkpoint's own config says exactly which knobs are involved:

```
slice_select band_groups | n_group 4 | group_pick per_study
slice_sampling center    | slice_depth_range (0.2, 0.8)   | crop_mm 130.0
```

The kernel used the crudest reading of `band_groups`: **12 slices spread evenly across the central
60 % of depth** (`[9, 11, 14, 16, 18, 20, 23, 25, 27, 29, 32, 34]`). `n_group: 4` almost certainly
means the range is split into **4 bands with 3 slices chosen per band**, and `slice_sampling: center`
means those three sit around each band's centre rather than being spread globally — which gives a very
different index set. And `crop_mm: 130.0` cannot be converted to pixels without the corpus spacing,
which is why **two crop scales were run instead of one guessed value**; the tighter crop won slightly
(0.7997 > 0.7929), which is weak evidence the true crop is smaller than the full 336 frame.

⚠️ **Honest limits.** (a) The author's 0.9062 is *their* number on *their* preprocessing; nothing here
shows our slice policy is close, only that the head is right. (b) Two crop scales and one slice policy
is a 6-cell grid — enough to separate "wrong forward" from "wrong input", not enough to locate the
right input. (c) **0.7997 is below the A167 member gate (solo ≥ 0.90), so exp085 is NOT yet usable as
a member** — it must first be driven to ~0.90 by fixing the preprocessing.

## 4. The next kernel, concretely

One more CPU run, same harness, varying only the input construction:

1. **slice policies** -- (i) global `linspace` (the current control), (ii) 4 bands × 3 slices evenly
   inside each band, (iii) 4 bands × 3 slices centred in each band, (iv) 4 bands × 3 slices at the
   band centre ± a small offset;
2. **crop scales** -- 336, 288, 273, 262, 240 (273 ≈ 130 mm if the 336-px frame is a 160 mm FOV, which
   is a guess worth testing rather than believing);
3. **order A only** -- the other two are now settled and would only triple the cost.

Every cell is 58 studies × 12 slices = 696 images, and the whole 6-cell grid took ~1 h on CPU, so a
20-cell grid is a few hours and needs **no accelerator and no quota**.
