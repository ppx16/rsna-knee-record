# MSFT fine-tune + 3D smoke: results (2026-09-28)

## 1. `kernel-tpu-msft` — the fine-tune FAILED its pre-registered bar, and degraded the parent

**Pre-registered bar:** its `gold_auc` must beat the parent `raptor_ft_coatnet_v5_full_swa.pt` at
**0.9214**.

**Result: `best_gold_auc` = 0.8968 — fails by 0.0246.**

| epoch | loss | GOLD macro-AUC | cumulative time |
|---|---|---|---|
| 0 | 0.789 | **0.8795** | 13,360 s |
| 1 | 0.776 | **0.8938** | 25,270 s |
| 2 | 0.730 | **0.8968** | 37,132 s |

⚠️ **The important reading is not "it stopped short" — it is that the warm start was damaged
immediately.** The parent scores **0.9214** and the very first epoch scored **0.8795**. So the
fine-tune was *below the parent from epoch 0* and never recovered. This is forgetting, not
under-training: at `bb_lr=3e-5 / head_lr=1e-3` over 3,479 training studies whose labels are mostly
**extracted from reports by a text model** (only 58 carry all twelve labels), the model is being
pulled off good weights toward noisy targets.

⇒ **The pre-registered bar correctly rejects this direction**, and it is rejected on mechanism, not on
a stopped run. Do not spend another GPU session on the same recipe.

⚠️ **The run never finished.** `"partial": true, "epochs_done": 3` of 4 — it was cancelled at the 12 h
session cap. Each epoch cost **13,360 s ≈ 3.7 h**, so 3 epochs + ~1,325 s of prelude = **10.7 h**.

## 2. ⚠️ `kernel-tpu-msft` never used a TPU — it burned GPU quota

The log line is unambiguous:

```
[tpu] no XLA (ModuleNotFoundError("No module named 'torch_xla'")) -> dev=cuda
device cuda | res 384 | k 12/24 | bs 2 | tag msft0 | roi=False(compartment)
```

and the metadata explains why:

```json
"enable_gpu": true,
"enable_tpu": false
```

So a kernel named **tpu**-msft was pushed as a **GPU** kernel. Consequences: no `torch_xla`, a T4 at
`bs=2 / res 384` doing **3.7 h per epoch**, and **~10.7 h of the 30 h weekly GPU quota spent** on a
run whose own premise (TPU) was never in force. ⚠️ This is the same family as the earlier
`machine_shape` trap (A-numbered: a GPU push failed because an inherited `machine_shape: Tpu1VmV38`
overrode the flags) — the flags and the name disagree, and nothing checks them against each other.

## 3. `kernel-tpu-3dsmoke` — a real TPU, killed by a one-character bug

This one **did** get a TPU and came up cleanly:

```
[3d] torch_xla 2.8.0 ; device xla:0 ; world_size 1
[3d] vols shape (3200, 44, 336, 336) dtype uint8
[3d] labels : 4407 rows parsed
```

and then died at ~75 s on:

```
TypeError: not all arguments converted during string formatting
  File "/kaggle/src/script.py", line 209, in main
    log('[3d] model: Conv3d stack width=%d, %.2fM params, input (1,%d,%d,%d)'
```

**Cause:** the format string has **5** placeholders, but the args tuple passed **6** values — the
literal `1` inside `input (1,%d,%d,%d)` was *also* supplied as an argument:

```python
% (a.width, nparam / 1e6, 1, a.depth, a.hw, a.hw)     # 6 args, 5 placeholders
```

**Fix:** `% (a.width, nparam / 1e6, a.depth, a.hw, a.hw)`. Verified: the file compiles and a
placeholder-vs-argument AST scan now reports no mismatch.

⚠️ **Why this is worth more than a one-line fix:** it sat on the *first* log line after the model was
built, so it killed the entire 3D TPU run **after** the expensive part (data load, XLA init) had
succeeded — the run looks like an infrastructure failure and is not one.

**Sweep:** all **78** kernel scripts were scanned with an AST checker for placeholder/argument
mismatches. Exactly **one** real instance existed (the above). The only other hit,
`kernel-diag-slots/diag_slots.py:161`, is a **false positive** — `"%s/%s" % key` where `key` is a
runtime 2-tuple, which the checker cannot see.

## What this changes

1. **The MSFT member is dead** on its own pre-registered terms; do not re-run that recipe.
2. **Anything named `tpu-*` must be checked for `enable_tpu`** before it runs — otherwise it silently
   consumes the scarce GPU quota at ~4× the cost.
3. **The 3D TPU line is unblocked** and is the cheapest untried direction, on a quota that is
   separate from the GPU's.
