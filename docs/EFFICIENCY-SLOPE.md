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

## 2. Measured

| stage | points | slope | fixed | **@1322** | confidence |
|---|---|---|---|---|---|
| `a5` | N=3 (×10 logs), N=48, N=4407 | **0.849 s/study** | ~17 s | **19.0 min** | **solid** |
| `dinov2` | N=3 (×10 logs), N=48 | 1.05–1.73 s/study | ~30–70 s | **24–38 min** | **weak — bound only** |

`a5` is solid: the N=4407 point dominates the fit, and its two pairwise slopes (0.913 and 0.848) agree
within 7 %.

⚠️ **`dinov2` is a BOUND, not a measurement, and my first pass over-claimed it.** `scripts/phase_census.py`
pooled all 16 `dinov2` points and returned a **slope of −0.016 s/study**, which is nonsense — because the
ten N=3 replicates span **50.6 s to 93.3 s**, pure machine variance that is larger than the entire N=48
signal. Fitting a slope across replicates of the same N treats noise as if it were a study-count effect.
A two-point fit is no better: taking the min, mean or median of the N=3 replicates moves the slope from
1.05 to 1.73 s/study. And the N=48 point comes from `e2oof`, a **different entry point** that may skip
the 20 fingerprint re-verifications. So `dinov2` is reported as a range.

**⇒ Two of the eight stages are ~43–57 min of the 1322-study run.** The other six have **no** large-N
point anywhere in the repo (§5), which is why the total is still an extrapolation.

## 3. ⭐ Why the 3-study arm ranking was noise, now explained

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

## 4. Bounding the total

The six unmeasured per-study stages (`radimagenet`, `public_raptor`, `residual_coat`, `global96_coat`,
`d4`, plus publish) total ~169 s at N=3 (median across the ten arms). Scaling each by the measured
slope-per-N=3-second ratio (0.0422, from `a5`) gives ~7.2 s/study, i.e. **~158 min** at N=1322. Using
`dinov2`'s ratio instead (0.0207–0.0254) gives ~95 min. Adding the measured **43–57 min**:

> **Estimated 1322-study runtime ≈ 2.3 – 3.6 hours** (the plan's prior range was 20 min – 18 h).

⚠️ This is an **extrapolation through a proxy ratio**, and the three CoAt members are separate child
processes whose scaling may differ from `a5`'s. It narrows the range — by roughly 4× — but it does not
replace the measurement. §5 is how that measurement gets made.

## 5. What the census found, and why the remaining slopes need a run

`scripts/phase_census.py` scans every log under `state/ downloads/ outputs/ work/ scratch/` for `[phase]`
records, infers each log's study count from its own prose, and fits a slope per stage. Result:

* **`a5`** — 14 points at three distinct N (3, 48, 4407) ⇒ fittable, and it is the one solid number here.
* **`dinov2`** — 16 points, **two** distinct N ⇒ only a bound (§2).
* **the other six stages** (`radimagenet`, `public_raptor`, `residual_coat`, `global96_coat`, `d4`,
  publish) — **only N=3**, ten replicates each. No second N exists anywhere on disk.

⚠️ **The census also caught a trap worth naming**, because it nearly produced a confident wrong answer:
pooling the ten N=3 `dinov2` replicates *with* the N=48 point yields a slope of **−0.016 s/study**, i.e.
"this stage gets faster per study" — pure artifact. Replicates of one N measure **variance**; they carry
no slope information, and here the variance (50.6–93.3 s) was larger than the signal a second N would
provide. `phase_census.py` therefore requires **at least two distinct N** before printing a fit at all,
and then compares the within-N spread against the fitted effect across the N range, printing
**UNRELIABLE** when the noise wins. On the current data it reports exactly that:

```
stage                pts     N#    s/study   fixed_s   @1322_s   noise_s  verdict
a5                    14      3     0.8492      16.4      1139      13.5  ok
dinov2                16      3    -0.0157      69.2        49     140.4  UNRELIABLE
```

**⇒ The six remaining slopes cannot be recovered from existing data; they need one new run.**

⭐ **And the mechanism already exists — it does not require building a synthetic root.**
`kernels/kernel-e2oof/` carries a patch (cell 32) that repoints the *cohort*, not the files:

```python
_fold_df = pd.read_csv(_fold_csv, dtype={'StudyInstanceUID': str})
_want = _fold_df[_fold_df['fold'] == _oof_fold].StudyInstanceUID.tolist()
if _oof_limit: _want = _want[:_oof_limit]
globals()['_RSNA_OOF_ONLY'] = set(_want)
# ⚠️ THE COHORT GLOBAL MUST BE REPOINTED, and this is not optional.
# `build_cache` takes its study list from `_RSNA_TEST_IDS` ...
```

It reads the fold assignment from the published `v52_oof.csv`, selects N training studies, and rewrites
`_RSNA_TEST_IDS` so the *whole* pipeline treats them as the test cohort. The **submission** notebook and
`kernel-e2oof` share the same filename (`rsna-knee-restructured-version-3.ipynb`), so this is a port of
one self-contained block, not a new harness.

⇒ **The decisive run: port that block into the submission notebook with `_oof_limit ≈ 200`, push it to a
T4, and read the `[phase]` records.** Estimated ~15–35 min of GPU (from §4). It yields the total slope
directly, plus the six missing per-stage slopes — which is what says *which* stage to attack.

### 5.1 STATUS: built and RUNNING

`scripts/build_slope_kernel.py` builds `kernels/kernel-slope/` from `kernel-0943cf`'s notebook and
**pushed it as `ppxl16/rsna-knee-slope` on 2026-09-29**; it went straight to `RUNNING` (no queue).
`scripts/watch_slope.py` waits for it, pulls the log to `state/logs_slope.json`, and re-runs the census.

The builder applies four patches, each by exact-anchor replacement that asserts it matched **exactly
once**, and `compile()`s every patched cell before writing:

1. **`walk()` study filter** — adds `_RSNA_OOF_ONLY` and skips studies outside it. Without it the header
   pass reads every series in the split (a measured **8.7 h** for a full `train_series` pass).
2. **The cohort repoint** — the OOF block is extracted **verbatim** from `kernel-e2oof`'s cell[32] rather
   than retyped, and wrapped in `if _oof_limit:` / `else:` so the parent's `test.csv` path is untouched.
3. ⚠️ **The image source** — **the patch neither upstream notebook has**, because neither needed it:
   `kernel-0943cf` cell[37] hard-codes `SERIES_ROOT = COMP / 'test_series'` and **forbids** the train
   fallback, while `kernel-e2oof` is truncated (23 code cells) and never reaches cell[37]. Pointing the
   cohort at training studies while the pixels still came from `test_series` would have measured 200
   training IDs against 3 test studies' images — the silent-degradation failure `kernel-e2oof`'s own
   comment warns about. Now keyed on the cohort set, never on a missing directory.
4. **The env** — `RSNA_OOF_FOLD`/`RSNA_OOF_LIMIT` set in the RUN CONFIG cell, since Kaggle gives no
   env at push time.

Verified before pushing: the only two `rsna_initialize()` call sites are cell[17] (test cohort, early)
and cell[32] (OOF cohort, later) — so the repoint wins and nothing clobbers it afterwards.

## 6. What this changes

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
