# RESULT 2026-10-02 — the msft2 TPU fine-tune COMPLETED, and it CLEARS the member gate

`ppxl16/rsna-knee-msft2` version 4 finished **COMPLETE, 4/4 epochs, `partial: false`**. The XLA
checkpoint shim (`docs/RESULT-2026-09-30-msft2-tpu-xla-checkpoint.md`) worked: no `torch.xla` error,
9308 s of training, and it wrote its checkpoints and per-epoch predictions.

## 1. ⭐ The warm start is intact, and then it improves

```
[zeroshot] zero training steps: GOLD macro-AUC 0.8961 over 58 studies
[zeroshot] VERDICT: PASS (parent bar 0.8963, got 0.8961)      <- the head-carry fix, verified at runtime
ep0  loss 0.767  GOLD macro-AUC 0.8903  (best 0.8903)
ep1  loss 0.776  GOLD macro-AUC 0.8946  (best 0.8946)
ep2  loss 0.728  GOLD macro-AUC 0.9007  (best 0.9007)         <- crosses 0.90
ep3  loss 0.688  GOLD macro-AUC 0.9048  (best 0.9048)
SWA over epochs [3, 2, 1] | gold 0.9069 (best-epoch 0.9048)   BETTER
epoch-ensemble 0.9068
```

| reference point | gold58 macro-AUC |
|---|---|
| the 2026-09-28 run (random head, wrong mechanism) | 0.8795 at epoch 0 |
| **the parent checkpoint reloaded through OUR pipeline** | **0.8963** — the bar |
| this run, zero-shot (warm start, no training) | **0.8961** — PASS |
| this run, best epoch (ep3) | **0.9048** |
| ⭐ **this run, SWA over epochs [3,2,1]** | ⭐ **0.9069** |

**⇒ +0.0106 over the pipeline's own warm-start number, and 0.9069 CLEARS the A167 member gate
(`solo ≥ 0.90`).** Per A167's own analysis, **nothing this project has ever trained could pass that
gate** — our from-scratch ceiling is ~0.877, and the only route was fine-tuning a published strong
checkpoint. **This is the first artifact of ours that passes it.**

For calibration: exp085's author reports `gold58 = 0.9062`, and this SWA reaches **0.9069** — the same
band, from a completely different route.

## 2. What was produced

```
raptor_ft_msft0.pt        292,824,723   best-epoch (ep3) fine-tuned checkpoint
raptor_ft_msft0_swa.pt    292,826,755   SWA over epochs [3,2,1]
raptor_ft_msft0.json                     the summary quoted above
raptor_gold_msft0_*.npz   5 files        gold predictions: best, ep1, ep2, epoch-ensemble, swa
raptor_oof_msft0_*_fold0.npz 5 files     OOF predictions for fold 0, same five variants
```

⚠️ **These are on Kaggle as kernel output, not in a dataset we own.** The exp085 lesson applies
directly: **copy them into a dataset under `ppxl16` before anything depends on them**, because a
kernel's output can be superseded and a public dataset can be withdrawn.

## 3. ⚠️ What this does NOT yet establish

1. **58 gold studies is a small set.** A +0.0106 move on 58 studies is inside the kind of spread a
   single study can produce; the number is real but its precision is not.
2. **Passing the solo gate is necessary, not sufficient.** A163's lesson stands: *a better model in
   isolation need not add to the ensemble*, because what a member must buy is **decorrelation**.
3. **The only number that matters is the leaderboard**, and the target is **+0.001** (0.943 → 0.944,
   which leaves the 599-team tie block). Nothing here is evidence of that yet.
4. The `UserWarning: None of the inputs have requires_grad=True` lines are torch's reentrant-checkpoint
   notice from the shim's fallback branch — and training demonstrably worked (loss fell 0.767 → 0.688,
   gold AUC rose every epoch), so it is noise on this path, not a silent failure.

## 4. Next, and the blocker

**The arm cannot be pushed today**: the weekly GPU quota is **0.00 h account-wide and refreshes
2026-10-03**, and an inference arm needs a GPU kernel version. So the sequence is:

1. now — **publish the two checkpoints as a `ppxl16` dataset** so they are ours to attach;
2. now — build the arm that adds this member to the `0943` stack;
3. **2026-10-03** — push it (GPU quota back) and submit, giving ~19 days against the 10-22 deadline.

## 5. The arc this closes

The 2026-09-28 result was rejected on a **wrong mechanism** (`load_raptor()` filtered to `backbone.`
keys, so the 8 head tensors were discarded and re-initialised). The correction said the right thing to
do was to carry the head, **verify the warm start at zero training, and only then fine-tune**. All
three were done, the gate passed at 0.8961 against a 0.8963 bar, and the fine-tune then moved the gold
number **up** by 0.0106 instead of "damaging a warm start". **The verdict now rests on a run rather
than on an argument.**
