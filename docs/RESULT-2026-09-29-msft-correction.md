# CORRECTION: the MSFT fine-tune was rejected on a wrong mechanism, and the fix is small

Written 2026-09-29. **This corrects §1 of [`RESULT-2026-09-28-msft.md`](RESULT-2026-09-28-msft.md)**, which
concluded *"the warm start was damaged immediately … this is forgetting, not under-training"* and
*"do not spend another GPU session on the same recipe"*.

## What was actually read from the checkpoint

`raptor_ft_coatnet_v5_full_swa.pt` (293 MB, local at `.testtmp/ms/`) contains **486 tensors** under a
`model` key: **478 `backbone.*` plus 8 head tensors**, whose names and nesting are:

```
clsW (12, 1024)          clsb (12,)
norm.weight (1024,)      norm.bias (1024,)
att.0.weight (256,1024)  att.0.bias (256,)
att.3.weight (12, 256)   att.3.bias (12,)
```

And our head in `kernels/kernel-tpu-msft/knee-mri-training-the-twelve-finding-model.py` is

```python
self.norm = nn.LayerNorm(F_dim)
self.att  = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
self.clsW = nn.Parameter(torch.zeros(n, F_dim))
self.clsb = nn.Parameter(torch.zeros(n))
```

— **the same names, the same nesting** (`att` is a Sequential with its Linears at indices 0 and 3), and
`F_dim = bb.num_features` (line 579) supplies the right **1024**. **The parent's head is shape- and
name-compatible with ours.**

## Why it never loaded

`load_raptor()` takes **only** the backbone:

```python
bbst = {k[len("backbone."):]: v for k, v in ck.items() if k.startswith("backbone.")}
if len(bbst) == 0: raise SystemExit(...)          # guards the EMPTY-dict trap
missing, unexpected = bb.load_state_dict(bbst, strict=False)
```

That filter is deliberate and it is a good guard — it was written to fix a real earlier bug where the
published wrappers hid the tensors under `model`/`student` and `strict=False` **silently accepted an
empty dict**. But the consequence is that **the 8 head tensors are discarded** and the head is
re-initialised (`clsW`/`clsb` start at zeros, `att` random).

## ⇒ The verdict is wrong, and so is the reason it gave

The fine-tune therefore ran as **a good backbone plus a brand-new head**. Under that reading:

* **epoch 0 = 0.8795 is not a damaged warm start — it is an untrained head**, and it *must* start below
  the parent's 0.9214 because the parent's 0.9214 was produced by the parent's **trained** head;
* the trajectory was **still rising** — 0.8795 → 0.8938 → 0.8968 — when the 12 h session cap cancelled it
  at **3 of 4 epochs**, i.e. `"partial": true`. It was **under-training, not forgetting** — the exact
  opposite of what the result doc concluded;
* and the pre-registered bar was therefore **mis-specified**: "beat 0.9214 within 4 epochs of scratch-head
  training" is not a fair test of the direction.

## The fix, and the gate that was missing

1. **Load the head too.** The 8 tensors are already correctly named and shaped; they only need to be kept
   rather than filtered out. The `len(bbst) == 0` guard should stay, extended to assert the head loaded as
   well.
2. **Verify the warm start at ZERO training, before any gradient step.** Evaluate the freshly-loaded model
   and require `gold_auc ≥ 0.9214` — the parent's own number. With the head loaded this must hold, and it
   is the check whose absence made the whole comparison ambiguous. It is cheap: one forward pass over the
   gold set.
3. Only then fine-tune, and compare against **the parent reloaded the same way** (an apples-to-apples
   baseline), not against a number taken with a different head.

## Why this matters beyond the bug

**A167 established the hard gate for any new member: solo ≥ 0.90** — *"Below ~0.90 solo, a member costs
more than its diversity buys"* — and noted that our own training ceiling is ~0.877, so **nothing we have
ever trained can pass it**. The one route with empirical precedent is fine-tuning a published strong
checkpoint, and this checkpoint is **0.9214 — already above the gate**. A correctly warm-started fine-tune
therefore does not have to *reach* the gate; it starts inside it. That is the only such starting point this
project has.

⚠️ **Honest limits.** (a) The 0.9214 is the checkpoint's own metadata, not a number this repo measured.
(b) Whether a fine-tuned variant then survives the **member gate** is a separate question and untested —
A163's lesson is that a better model in isolation need not add to the ensemble. (c) The earlier run also had
a **separate** hardware bug (`enable_gpu: true, enable_tpu: false`, so a kernel named `tpu-msft` ran on a T4
for 10.7 h); that must be fixed in the same pass, or the recipe is untestable at a sane cost.
