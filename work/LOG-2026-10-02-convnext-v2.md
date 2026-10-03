# RUN LOG — 2026-10-02 (round: ConvNeXt TPU arm, v1 ERROR → v2 relaunched)

Machine gets shut down without warning. This file is the resume point.

---

## 1. v1 of `ppxl16/rsna-knee-msft2-cx` ended in **ERROR** — and the guard that stopped it was mine

⚠️ It did **not** "finish". `kernels_status` returned `{"status": "ERROR", "failureMessage": null}`
and **no checkpoints were produced**. Log: `work/cx_out/rsna-knee-msft2-cx.log` (pulled).

The log, in order:

| line | what it says | meaning |
|---|---|---|
| 20 | `[raptor] loaded backbone ... \| loaded 342 tensors, missing 0, unexpected 0` | ✅ **the ConvNeXt backbone matches the checkpoint EXACTLY.** `convnext_small.in12k_ft_in1k` was the right arch string, and `--res 384` is fine. This was the main risk and it is now retired. |
| 21 | `[raptor] head tensors carried: 11 [...]` | 11 non-backbone tensors came with the checkpoint |
| 23 | `[msft2] head is INCOMPATIBLE with this architecture: absent=[all 11]` | **the exit.** `SystemExit`, raised by the msft2 head guard |
| 24–25 | `[NbConvertApp] Converting notebook ...` | the harness converting the notebook after the script died |

So the run died at about **1997 s ≈ 33 min**, *before* the zero-shot evaluation and *before any
gradient step*. Almost all of the 33 min was TPU startup + corpus loading (line 7 at t=979 s, line 9
at t=1962 s). Burned: ≈0.55 TPU-hours; nothing else. **No training happened, so no evidence about
ConvNeXt quality exists yet.**

### Why the guard fired, and why that was correct
`RaptorClassifier`'s head is `norm / att / clsW / clsb`. The v4cx checkpoint's head is a completely
different design. That guard exists because of the **2026-09-28 bug**: a run loaded an empty
`state_dict` under `strict=False`, trained a random head, and printed a success-shaped line; its
epoch-0 score of 0.8795 was then **misread as a damaged warm start**. Refusing loudly is the right
default. The guard simply had no override for the *legitimate* case of deliberately fine-tuning a
different architecture.

---

## 2. The foreign head, read out of the checkpoint (`work/inspect_v4cx.py`)

Downloaded only the one file (`dataset_download_file`, 190 MB, 12 s) to `work/v4cx/`.
Top level is an `OrderedDict` — **no `model`/`student` wrapper, no `epoch` key**.

```
backbone.*   342 tensors   (stem.0 96ch 4x4, stages.N.blocks.M.conv_dw/gamma/mlp, head.norm)
head         11 tensors:
   pos_proj.weight      (768, 16)      pos_proj.bias   (768,)
   freqs                (8,)           <- buffer, not trained
   pool.att.0.weight    (192, 768)     pool.att.0.bias (192,)
   pool.att.2.weight    (5, 192)       pool.att.2.bias (5,)
   head.0.weight        (5376, 5376)   head.0.bias     (5376,)
   head.2.weight        (12, 5376)     head.2.bias     (12,)
```

Reads as a **position-aware attention-MIL head**: a 16-dim positional descriptor (8 `freqs` → sin/cos)
projected to 768 and added to slice features; attention pooling 768→192→5; then a 5376→5376→12 head,
where 5376 = 7 × 768 (7 slices flattened).

### ⚠️ Decision: do NOT port this head
The checkpoint does not reveal **how `freqs`, `pos_proj`, `pool.att` and `head` are wired** — what
activation, where the positional term is added, what the `5` indexes, whether `head.0` sees a
flattened 7×768 or something else. Guessing the forward pass produces a model that **runs and emits
plausible numbers while being wrong**, which is the worst failure mode available and is the same class
as every other bug this project has hit (silent head swap, silently dropped dataset source, silently
empty `state_dict`). Porting needs the author's own model definition, which is not in the dataset.
**Recorded as not-done, not as done.**

---

## 3. v2 pushed: `ppxl16/rsna-knee-msft2-cx` — **QUEUED**

`work/push_cx2.log`, `EXIT=0`; `py_compile` on the generated script, `EXIT=0`.
Status at 2026-10-02: `{"status": "QUEUED"}`.

**The change is one flag plus a cleaner guard, applied to BOTH kernels** (the builder regenerates
`cx` from `msft2`, so an edit only in `cx` would be silently reverted by the next rebuild):

1. `kernels/kernel-msft2/knee-mri-training-the-twelve-finding-model.py` and
   `kernels/kernel-msft2-cx/knee-mri-convnext-finetune.py` — the head guard at ~line 639 now raises
   only when `--allow_bad_warmstart` is absent; when overridden it sets `_FOREIGN_HEAD = True` and
   prints `DISCARDING 11 foreign head tensor(s) ... training the head FROM SCRATCH`.
2. The zero-shot gate (~line 708) does **not** fire when `_FOREIGN_HEAD`, and prints
   `VERDICT: N/A -- head was FOREIGN ... judge by the per-epoch curve`. Rationale: with a
   from-scratch head the score is ~chance *by construction*, so failing on it would be reading
   noise as a verdict — and passing `--allow_bad_warmstart` to silence it would destroy that flag's
   meaning (it is for a warm start that genuinely scored badly).
3. `scripts/build_msft2_cx_kernel.py` argv now passes `--allow_bad_warmstart`, with the reasoning
   in a comment.

Default behaviour is unchanged for every other invocation, so `msft2` (COMPLETE) is unaffected.

### What v2 will and will not tell us
- ✅ It will produce a real **per-epoch gold58 macro-AUC curve** and an SWA number for a ConvNeXt
  member — the thing A163 wants (a genuinely *decorrelated* architecture).
- ❌ The zero-shot line will be ~0.5 and **means nothing**; do not quote it as backbone quality.
- ⚠️ Expect it to land **near or below the 0.90 A167 gate**: v4cx starts from a *random* head, whereas
  msft2 started at 0.8961 with a carried head. The gate is a heuristic, not a law — a sub-gate but
  decorrelated member can still be worth blending, but it must be *priced on the blend*, not assumed.
- ⚠️ Budget: ~16.5 TPU-hours remained after v1. A 4-epoch run took 2.6 h for msft2. GPU is still
  **0.00 h until 2026-10-03**, which is why TPU is the only place this can run.

---

## 4. Still the actual bottleneck, unchanged

**The msft2 member (gold58 0.9069, SWA, already published as `ppxl16/rsna-knee-msft2-ckpt`) has still
never been added to the ensemble nor submitted.** That remains the whole of the paid-for headroom.
The arm lives in `kernels/kernel-0943s-2coat` (`rsna-knee-restructured-version-3.ipynb`,
`enable_gpu: true`, `NvidiaTeslaT4`) and cannot even be *tested* until GPU refreshes on 10-03.

---

## 5. Traps hit this round

- `api.dataset_download_file(slug, filename, path=...)` works and pulls a single file out of a
  5-file, ~1 GB dataset — 190 MB in 12 s. Do not download the whole set.
- The checkpoint has **no wrapper key and no `epoch`** (unlike `raptor_ft_coatnet_v5_full_swa.pt`,
  which is wrapped under `model`). `load_raptor` handles both; the `ssl epoch ?` in the log is
  expected, not an error.
- `--allow_bad_warmstart` existed but **only** guarded the zero-shot score, not the head shape. Two
  guards, one flag — reading the code rather than the flag name is what caught this.
- A kwarg-less belief that "the builder will regenerate it" would have reverted the guard fix;
  hence both kernels were patched.
