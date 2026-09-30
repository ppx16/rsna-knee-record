# RESULT 2026-09-30 — the TPU fine-tune FAILED, and the cause was `--grad_ckpt` on XLA

The `ppxl16/rsna-knee-msft2` version-3 TPU run finished in **ERROR**. This is what it was, why it is a
one-line-class fix, and what was pushed in response (**version 4, QUEUED** at 2026-09-30 13:38Z).

## 1. How far it got — almost everything

Decoded from `work/msft2_fail/rsna-knee-msft2.log` (`scripts/klog.py`). **Every part of the run that
previous sessions fixed, worked:**

```
[tpu] deps ok (timm+sklearn importable, BLAS threads pinned to 1)
[prelude] labels report_labels_v4hybrid.csv -> labels.parquet  shape=(4407, 25)
[prelude] warm start -> raptor_ft_coatnet_v5_full_swa.pt
[tpu] xla_device=xla:0 (single process, no xmp.spawn)      <-- TPU came up
device xla | res 384 | k 12/24 | bs 2 | tag msft0 | roi=False(compartment)
[cv] fold 0/5: train 3479 | OOF held-out 870
[raptor] loaded backbone ... 478 tensors, missing 0, unexpected 0
[raptor] head tensors carried: 8 ['att.0.bias', 'att.0.weight', 'att.3.bias', 'att.3.weight',
                                 'clsW', 'clsb', 'norm.bias', 'norm.weight']
[raptor] gradient checkpointing ON
[msft2] HEAD loaded: 8/8 tensors, names and shapes verified   <-- the 2026-09-29 correction WORKS
```

⇒ **The A-correction is confirmed on TPU at runtime**: the 8 head tensors are carried *and* verified
8/8. The warm start is intact. The run then died on the **first forward pass**, before a single
training step.

## 2. The cause

```
File ".../torch/utils/checkpoint.py", line 93, in _get_device_module
    device_module = getattr(torch, device)
File ".../torch/__init__.py", line 2745, in __getattr__
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
AttributeError: module 'torch' has no attribute 'xla'
```

Reached from `torch/utils/checkpoint.py:494 _checkpoint_without_reentrant_generator`, called by
`timm/models/_manipulate.py:279 checkpoint_seq`, called by `timm/models/maxxvit.py:1262`.

**`--grad_ckpt` → `bb.set_grad_checkpointing(True)` → `torch.utils.checkpoint.checkpoint`, and
torch's NON-REENTRANT path asks torch for the device module of the tensor's device —
`getattr(torch, "xla")` — which `torch_xla` never registers.** Nothing is wrong with the model, the
checkpoint, the data or the recipe.

## 3. ⚠️ Why `--grad_ckpt` could not simply be removed

The script's own docstring measures it: *"The CoAtNet at 384 with `--bs 8` and `--grad_ckpt` sits at
about **23.7 GB**, so it needs a 24 GB card."* **Checkpointing is what makes the model fit at all**,
so turning it off trades a crash for an OOM. The fix has to make checkpointing *work* on XLA.

## 4. The fix, pushed as version 4

`scripts/build_msft2_kernel.py` now injects an XLA-safe checkpoint shim into the TPU prelude, before
any model is built. Preference order, with the reason for each step:

1. **`torch_xla.utils.checkpoint.checkpoint`** — torch_xla ships its own XLA-aware implementation.
   The wrapper **strips `use_reentrant` / `determinism_check` / `debug` / `context_fn`**, because
   those are `torch.utils.checkpoint` parameters and passing them to the XLA function raises
   `TypeError`.
2. **If that import fails, force `use_reentrant=True`.** The reentrant path is
   `CheckpointFunction.apply`, which **never calls `_get_device_module`** — so it cannot raise the
   error above.
3. `timm.models._manipulate.checkpoint` is patched too when the module bound the symbol directly,
   since `checkpoint_seq` may reference the bound name rather than the attribute.

It prints `[tpu] activation checkpointing -> ...` so the log says which branch was taken — the
absence of that line in the next log would itself be a finding.

## 5. ⚠️ Honest limits on this fix

* **It cannot be tested locally.** The failure needs an XLA device; there is no TPU on this machine,
  and a CPU run never executes this path. It is a reasoned fix, verified only by *compiling*.
* **A wrong guess costs a queue wait, not quota.** The account's single TPU slot serialises these
  runs; version 3 was QUEUED for hours before running. TPU quota remaining: **18.07 h**.
* ⚠️ **The `--device` default is `gpu`.** The first rebuild of this fix produced
  `enable_tpu=False enable_gpu=True`, which would have been **refused** by the exhausted weekly GPU
  quota (30.45 h of 30.00, refreshing 2026-10-03). It was rebuilt with `--device tpu`; the metadata
  now reads `enable_tpu: true, enable_gpu: false`. **Check that field before every push.**
* If branch 2 is taken and torch's reentrant checkpoint complains that no input requires grad, the
  next fix is a larger change (a custom XLA `torch.autograd.Function`), not another constant.

## 6. The reliable fallback, if the TPU path keeps fighting back

The **same recipe is proven on GPU**: the 2026-09-28 run trained (`--grad_ckpt` on CUDA is fine) and
its only faults were the head-loading bug — since fixed and now verified — and running on the wrong
accelerator. **The weekly GPU quota refreshes 2026-10-03**, which still leaves 19 days before the
2026-10-22 deadline. If version 4 fails again, the GPU is the route with no unknowns.
