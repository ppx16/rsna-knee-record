# RSNA Knee — the per-study runtime slope, partially MEASURED (2026-09-29)

Companion to `docs/EFFICIENCY-RUNTIME.md`, and a **correction** to its §3: that section ranked the nine
arms by 3-study wall clock, and the ranking turns out to be noise. This doc has the measurement that
shows why, plus the first measured per-study slopes.

## 1. Method: the slope is recoverable from logs we already have

`EFFICIENCY-PLAN.md` Step 1 assumed we must run a synthetic test root at several N. **We did not need
to.** The notebook emits `[phase] {"stage": ..., "status": "START"|"COMPLETE", "elapsed_seconds": ...}`
records, and the OOF kernels (`a5oof`, `a5oof_full`, `e2oof`) run the **same named stages over many
training studies**. So each stage has usable (N, seconds) points already on disk.
`scripts/arm_timing.py --phase-table` / `--phases <arm>` now dumps them.

⚠️ Read the `[  123.4s] label` stamps instead and you will conclude nothing: on a 337 s run they **stop
at 111 s**, because the two stages that actually loop over studies (Raptor, and the three CoAt members)
print no timestamped progress at all. The `[phase]` records are the only per-stage boundaries.

## 2 Measured

| stage | N=3 | N=48 | N=4407 | slope | fixed | **@1322** |
|---|---|---|---|---|---|---|
| `a5` | 20.1 s | 61.2 s | 3758.6 s | **0.849 s/study** | 19.0 s | **19.0 min** |
| `dinov2` | 76.4 s | 140.6 s | — | 1.427 s/study | 72.1 s | **32.6 min** |

`a5` rests on **three** points from three runs; its two pairwise slopes (0.913 and 0.848) agree within
7 %, so it is solid. `dinov2` rests on **two** points and the N=48 point comes from a different entry
(`e2oof`) which may skip the 20 fingerprint re-verifications, so its fixed term is less trustworthy —
treat it as suggestive.

**⇒ Two of the eight stages alone are ~52 min of the 1322-study run.**

## 3 ⭐ Why the 3-study arm ranking was noise, now explained

The fits say the per-study share of each stage **at N = 3** is only:

```
a5      : 11.8 %      dinov2 : 5.6 %
```

So ~88–94 % of what these stages cost on a 3-study run is **one-time loading**, and the run is measuring
checkpoint loads and process spawns, not the pipeline. That is exactly why §3 found arms with *identical*
configs spanning 28.9–44.0 s on `public_raptor`, and why the `raptor_k_eval` 62↔94 knob (a **34 %**
change in per-study window count) moved that stage not at all (`0943s` k=62 → 31.8 s vs `0943` k=94 →
28.9 s).

⚠️ **This corrects §3's headline.** "`0943` is the fastest arm (258.2 s)" is **not established** — that
ordering is indistinguishable from machine variance, and I should not have implied otherwise. What §3
does establish is the *config* fact, which is unaffected: `0943` and `0943cf` are on the same side of the
routing hedge as each other, and `0943p` is the other side.

## 4 Bounding the total

The six unmeasured per-study stages (`radimagenet`, `public_raptor`, `residual_coat`, `global96_coat`,
`d4`, plus publish) total **193.6 s at N=3**. Scaling each by the two measured slope-per-N=3-second
ratios (0.0187 from `dinov2`, 0.0422 from `a5`) bounds them at 3.6–8.2 s/study, i.e. **80–180 min** at
N=1322. Adding the measured 52 min:

> **Estimated 1322-study runtime ≈ 2.2 – 3.9 hours** (the plan's prior range was 20 min – 18 h).

⚠️ This is an **extrapolation through a proxy ratio**, and the three CoAt members are separate child
processes whose scaling may differ. It narrows the range; it does not replace the measurement. The
decisive run is a single kernel at moderate N (~200) with `RSNA_COMP_ROOT` pointed at a synthetic root
built from training studies — which the OOF logs prove is possible, since `e2oof` already built
`48 studies, 227 slot-series` out of training studies.

## 5 What this changes

**Runtime is 2–4 h, so `RuntimeSeconds/32400` ≈ 0.22–0.44 against an AUC term of ≈ −2.06.** Trimming 30 %
of 3 h = 3240 s = **0.10 efficiency**, and among the **308 teams tied with us at 0.943** runtime is the
*only* differentiator. Meanwhile the ~100 s of fixed overhead (EDA, fingerprints, publish) is **0.003** —
worth doing because it is free, but it is not where the money is.

⭐ **And it produces a concrete selection recommendation.** `0943s` is `0943` with **one** difference:
`preset=sparse`, i.e. `raptor_k_eval` back at its released 62/42 instead of 94. Both scored **0.943**.
So the 94-window override — whose own config comment calls it *"the accuracy change that needed no new
model, and it is only affordable because of the speed work"* — **bought nothing on the public board while
adding ~34 % to the Raptor stage's per-study work.** `0943s` therefore shares `0943`'s routing-hedge side
and its public AUC, at strictly less per-study compute.

⇒ **The efficiency-minded pair is `0943s` + `0943p`** (published routing + flat routing, the same hedge,
with the cheaper primary). The pre-registered `0943cf` + `0943p` remains the accuracy-minded pair.

⚠️ Caveat, stated because it is genuinely unresolved: 0.943 vs 0.943 is inside the board's ±0.008 noise,
so "the 94-window override bought nothing" is a statement about the *public* board only. The private
effect of denser sampling is exactly the kind of thing this project has been burned by before (A122), and
it is why this is a recommendation about *which knob to spend runtime on*, not a claim that 94 is useless.
