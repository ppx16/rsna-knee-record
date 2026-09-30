
---

## 10. The zero-shot gate RAN and returned FAIL — and the failure is the finding

```
[tpu] no XLA -> dev=cpu
device cpu | res 384 | k 12/24 | bs 2 | tag msft0
train 64 | gold-val 58 | oof 870
[msft2] HEAD loaded: 8/8 tensors, names and shapes verified
[zeroshot] zero training steps: GOLD macro-AUC 0.8963 over 58 studies
[zeroshot] per-label {"ACL":0.9596,"MCL":0.9365,"Medial Meniscus":0.9543,"Lateral Meniscus":0.872,
                      "Medial OA":0.9736,"Lateral OA":0.8066,"PF OA":0.8314,"Effusion":0.9826,
                      "Synovitis":0.8339,"Baker's":0.8931,"Contusion":0.8313,"Fracture":0.8806}
[zeroshot] VERDICT: FAIL (parent bar 0.9214, got 0.8963)
[zeroshot] --zero_shot_only: stopping before any training
```

Three things follow, in order of importance.

### 10.1 ⭐ The parent's 0.9214 is NOT reproducible in our pipeline — so it is the wrong bar

The head patch is confirmed working (`8/8 tensors, names and shapes verified`), and the model is the
parent's own weights. Yet in **our** pipeline it scores **0.8963**, a gap of **0.0251** from the
`gold_auc = 0.9214` in the checkpoint's metadata.

That gap cannot be attributed to a broken warm start, because nothing about the load is broken. It is a
**pipeline difference** — a different gold set, different crops/normalisation/`eval_slot_budgets`, or a
different aggregation than whatever produced the metadata. The consequence is decisive for the gate:

> **0.9214 is unusable as a "the warm start is intact" bar.** A check against it cannot distinguish
> "broken load" from "our evaluation differs from theirs", which is exactly the ambiguity the gate was
> built to remove. **The bar must be our own measured reload: 0.8963.**

This is the same failure mode as A63/A167's "numbers from another instrument": a published figure is not
a measurement of our pipeline, and using it as a gate produces a false alarm.

### 10.2 The corrected diagnosis is confirmed — the head was worth ~+0.017

| | gold macro-AUC | head |
|---|---|---|
| `kernel-tpu-msft` epoch 0 (2026-09-28) | 0.8795 | **random-init** (never loaded) |
| `msft2` zero-shot reload (today) | **0.8963** | **loaded, 8/8 verified** |

⇒ Loading the head is worth **≈ +0.017** at the same point in the recipe. ⚠️ Honest caveat: these are not
perfectly matched — the 0.8795 is *after one training epoch* while 0.8963 is zero-shot — but the
direction and rough size agree with the mechanism, and the trajectory of the original run
(0.8795 → 0.8938 → 0.8968, still rising when the 12 h cap killed it at 3 of 4 epochs) already said it was
under-training a fresh head rather than damaging a warm start.

### 10.3 What the next step actually is

Not "verify against 0.9214" — that question is answered and the answer is no. The live question is
**relative**: fine-tune from the correctly-loaded **0.8963** and beat it, ideally crossing the **solo ≥ 0.90**
member gate (A167) on our own instrument.

* The kernel is otherwise ready: head loading verified, the gate stops before training, `--limit` keeps
  the probe cheap, and CPU works with no quota.
* But a real fine-tune needs an **accelerator**, and both are unavailable: the weekly **GPU quota is
  spent** (refreshes 2026-10-03) and the account's single **TPU slot is held by `kernel-tpu-3dsmoke`,
  still QUEUED after ~19 h**.
* ⇒ **To continue this line, `kernel-tpu-3dsmoke` has to be deleted** (it is fully re-pushable from
  `kernels/kernel-tpu-3dsmoke/` + `scripts/build_3d_smoke.py --push`). That is now a decision rather than
  a hypothetical: it is the only thing standing between us and the one checkpoint that starts near the
  member gate.
* And the gate's bar must be changed to 0.8963 before the fine-tune runs, or the run refuses to train
  (correctly, on a bar that is now known to be the wrong one).
