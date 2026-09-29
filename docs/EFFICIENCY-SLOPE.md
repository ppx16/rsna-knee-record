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

| stage | studies used | slope | **@1322** | confidence |
|---|---|---|---|---|
| `a5` | 3 (×10 logs) → 4407 | **0.8493 s/study** | **18.9 – 19.1 min** | **solid** |
| `dinov2` | 3 (×10 logs) → **200** (slope probe) | **1.7517 s/study** | **37.4 – 41.4 min** | **solid** |

**⇒ ~58 min of the 1322-study run is in just these two stages**, and both are now *measured*, not
extrapolated. The ranges span the low-N (N=3) replicate spread, which is machine variance of 50–93 s.

`a5` is solid: the N=4407 point dominates the fit, and its two pairwise slopes (0.913 and 0.848) agree
within 7 %.

⭐ **`dinov2` was promoted from a bound to a measurement by the slope probe's second run**, which
reached `dinov2 COMPLETE` at N=200 in **414.2 s** against 76.4 s at N=3. Two further corrections came
with it, and both were found by running the tool rather than reading it:

* The naive pooled fit reported **−0.016 s/study** — see §5.
* The first `--pairwise` pass reported dinov2 as **0.6–1.1 min**, because it paired N=3 against
  **N=4407**, where `a5oof_full` logs `dinov2 COMPLETE` in **0.2 s**. That is a **skipped** stage (the
  DINOv2 features were precomputed in that run), not a fast one, and because it is the largest study
  count on record it poisoned the estimator. `phase_census.py` now drops sub-second completions as
  skips — with an **absolute** threshold, because a relative one would also discard `a5`'s genuine
  20.1 s point.

⚠️ **`dinov2` is a BOUND, not a measurement, and my first pass over-claimed it.** `scripts/phase_census.py`
pooled all 16 `dinov2` points and returned a **slope of −0.016 s/study**, which is nonsense — because the
ten N=3 replicates span **50.6 s to 93.3 s**, pure machine variance that is larger than the entire N=48
signal. Fitting a slope across replicates of the same N treats noise as if it were a study-count effect.
A two-point fit is no better: taking the min, mean or median of the N=3 replicates moves the slope from
1.05 to 1.73 s/study. And the N=48 point comes from `e2oof`, a **different entry point** that may skip
the 20 fingerprint re-verifications. So `dinov2` is reported as a range.

**⇒ Two of the eight stages are ~43–57 min of the 1322-study run.** The other six have **no** large-N
point anywhere in the repo (§5), which is why the total is still an extrapolation.

### 2.1 A new hard point from the slope probe's first (failed) run

The v1 run died after the dinov2 stage, but it did log its data-staging sub-stage at both sizes, and
that is a clean two-point measurement:

| data staging (`test g1: COMPLETE …`, **inside** the dinov2 phase) | studies | seconds |
|---|---|---|
| `0943cf` | 3 | 8.1 s |
| `rsna-knee-slope` v1 | **200** | **132.1 s** |

⇒ **0.629 s/study + 6.2 s fixed = 14.0 min at 1322 studies.** ⚠️ This sits *inside* the `dinov2` phase
(which spans 2.8–79.6 s at N=3), so it **refines** dinov2's slope rather than adding a stage — and it
independently confirms dinov2's slope is at least 0.63 s/study, which is the opposite of the −0.016 the
pooled fit produced. It also means dinov2's `24–38 min` bound is real and mostly *staging*, not
fingerprinting: staging alone is 14 of those minutes.

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

### 5.2 ⚠️ WHERE IT STOPPED, AND WHY THE COHORT REPOINT IS STRUCTURALLY BLOCKED

Run v1 failed in `main()`'s frontier validator; that was patchable (patch 5) and v2 **verified three of
the five patches working in the live log**:

```
[  415.7s] slope probe: frontier validated against the OOF cohort (200 studies)      <- patch 5
[  419.9s] slope probe: OOF active -> reading pixels from .../train_series ...        <- patch 3
dinov2 COMPLETE at 416.0s for N=200 (vs 76.4s at N=3)                                 <- the measurement
```

and then v2 died one stage later:

```
RuntimeError: A5 test/sample UID mismatch
```

found in cell[37]:

```python
sub_df = pd.read_csv(COMP / 'sample_submission.csv', dtype={'StudyInstanceUID': str})
ser_csv = pd.read_csv(COMP / 'test_series.csv', dtype={...})
if set(sub_df.StudyInstanceUID) != set(_RSNA_TEST_IDS):
    raise RuntimeError('A5 test/sample UID mismatch')
```

⚠️ **This is not a truncation artifact — the A5 stage requires the cohort to be the competition's own
`sample_submission.csv`.** The two obvious hypotheses are both dead: `--limit 0` (use the whole fold)
would not help, and the A5 pixels would still be grouped from `test_series.csv`, which has no rows for
training studies. Patching past it means repointing *three* further things inside that one stage
(`sample_submission.csv`, `test_series.csv`, and the `studies` list), and the remaining stages validate
against the test set the same way — so the six missing slopes would require rebuilding the pipeline's
whole cohort handling, which is what the purpose-built `kernel-a5oof` did for **one** stage.

**Decision: stop here.** Two of eight stages are measured, the estimate is bounded, and the remaining
six have a clearly identified target (below). Spending more GPU would refine a total that does not
change the action. GPU cost so far: **two runs, ~20 min**.

⭐ **The target is already identified without those slopes.** The six unmeasured stages total ~169 s at
N=3, of which the **three CoAt members are 145.5 s — 75 %**. Whatever their exact slope, the CoAt family
is where the per-study runtime lives, and that is what any runtime work should attack.

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

## 7. ⭐ The CoAt family dominates the rerun — and the receipt says why

`downloads/radcal00/_coat_arm_receipt.json` is the child process's own receipt for **one** CoAt member
(the resgated arm), and it decomposes that member's runtime properly instead of leaving it as a single
`[phase]` number:

| field | value | reading |
|---|---|---|
| `studies` / `series` | 3 / 15 | the visible cohort |
| `elapsed_seconds` | **37.33** | the whole child, for 3 studies |
| `shards` | 2, on GPU 0 and GPU 1 | the member splits the cohort across both T4s |
| `model_load_seconds` | **7.93** and **7.90** | 3 checkpoints each, so **loaded twice per member** |
| `dicom_preparations_per_study` | **1** | each member prepares its own DICOMs |
| shard 1 | 2 studies, 16.96 s | ⇒ **(16.96 − 7.90)/2 = 4.53 s/study** |
| shard 0 | 1 study, 18.28 s | ⇒ 10.35 s/study — inflated by first-shard warmup, so **4.5 is the better estimate** |

The parent's own phase for the same member is **53.6 s** (`0943cf`), so roughly **16 s of every member is
parent overhead in the critical path** — a `pip install --no-deps --target` of its own `timm-1.0.22`
wheel, plus process spawn.

### Revised estimate

Per member at N=1322 ≈ 16 (parent, fixed) + 15.8 (both shards' loads, fixed) + (1322/2) × 4.53 ≈ **50 min**.
Three members ⇒ **~2.1–2.5 h for the CoAt family alone**, which is the dominant term:

| | @1322 |
|---|---|
| `a5` | 19.0 min *(measured)* |
| `dinov2` | 37–41 min *(measured)* |
| **CoAt family ×3** | **~2.1–2.5 h** *(estimated from the receipt)* |
| `radimagenet` + `public_raptor` | ~20–40 min *(unmeasured)* |
| **total** | **~3.5–4 h** |

⚠️ So the earlier **2.3–3.6 h** is revised **upward to ~3.5–4 h**, and the reason is that the CoAt family
is bigger than the proxy-ratio extrapolation assumed. At 4 h, `RuntimeSeconds/32400` ≈ **0.44** against an
AUC term of ≈ −2.06, and among the **308 teams tied with us at 0.943** runtime is the only differentiator.

### The waste, precisely

1. **Each member is passed only `competition_root`** — no cache path — so every member re-prepares the
   DICOMs. The receipt's own `dicom_preparations_per_study: 1` is a within-member claim ("we prepare each
   study once"), not a cross-member one. **Three members ⇒ the image preparation is done three times**,
   and it is per-study, so it scales with the 1322-study rerun.
2. **Each member loads its 3 checkpoints twice**, once per shard. That is 47 s across the family on the
   visible run — a fixed cost, so it matters much less at N=1322 than item 1.

### Why merging is not the fix, and what is

The three members run as **separate `python -c` children with privately pip-installed dependency trees**
(`pip install --no-deps --target <envd> <timm wheel>`, and each asserts its own `cv2.__version__`). That
dependency isolation is presumably *why* they are separate processes, so folding them into one process
risks exactly the kind of version-conflict failure the isolation prevents.

⇒ The natural fix is **not** one process but **one preparation**: render the pixels once and have all
three children read the result. That is the lever worth investigating next, and it is a code question,
not a GPU one — the thing to establish first is whether the children can consume a preparation cache at
all, since `run_submission` lives in the artifact (`input_resgated_runtime.py` is generated from the
Kaggle dataset, not from this repo) and its signature as called here passes no cache.
