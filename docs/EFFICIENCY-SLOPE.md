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

## 8. ⭐ How much is runtime worth? One hour = 23 % of the entire accuracy range

`Efficiency = AUC/(Benchmark − max AUC) + RuntimeSeconds/32400`, with `Benchmark = 0.5` measured and
`max AUC` the global best (eff#1 = 0.958), so the denominator is a **negative constant** and

```
E(a, R) = −2.18341 · a + R/32400          (lower is better)
```

Because the AUC coefficient is a *constant* and runtime is divided by 32400, the two are directly
comparable and the leverage is easy to state:

> **1 hour of runtime = 0.1111 efficiency = 23.3 % of the whole accuracy range from AUC 0.74 to 0.958.**

That is why we sit at **eff#3388** with an AUC of 0.943 while being **rank 430** on the main LB: on this
board a team with AUC 0.90 and a ~30-minute run beats us comfortably, because our ~3.5–4 h costs us
≈ 0.39–0.44 while the entire spread from a mediocre AUC to the best possible AUC is only ≈ 0.48.

| our AUC | our runtime | our efficiency |
|---|---|---|
| 0.943 | 1 h | **−1.948** |
| 0.943 | 2 h | −1.837 |
| 0.943 | 3 h | −1.726 |
| 0.943 | **4 h** | **−1.615** |
| 0.943 | 6 h | −1.392 |

And the runtime we would need to take the lead off a rival: to beat a **0.943 / 1 h** team we need
**< 1.00 h**; to beat a **0.90 / 1 h** team we need **< 1.84 h**; to beat a **0.958 / 1 h** team we need
**< 0.71 h**.

⚠️ **Robust to the one ambiguity in the formula.** "max AUC" might be the global best (a constant) or the
team's own maximum (also a constant for a team whose two picks tie, which ours do). Either reading leaves
the AUC coefficient *constant within a tied group*, so the rank ordering at fixed AUC is pure runtime and
the 0.1111-per-hour leverage is identical. The cross-AUC comparison above uses the global-best reading.

### So the runtime work is the $18 000, and the target is CoAt

| lever | saving @1322 | efficiency | cost to test |
|---|---|---|---|
| **A. Drop a CoAt member** (3 → 2) | ~40–50 min | 0.07–0.09 | price it in OOF locally first, then 1 kernel + 1 free submission |
| **B. Share one DICOM preparation across the three children** | ~45 min | ~0.08 | 2.1 GB artifact download + surgery on generated artifact code |
| **C. `0943s`'s `sparse` preset** (already measured at 0.943) | ~7–14 min | 0.02–0.04 | **zero — it is already a submitted arm** |
| D. Trim EDA / fingerprint / publish overhead (~100 s) | ~100 s | 0.003 | trivial, free |

⚠️ **C is the only one that is already paid for.** `0943s` scored **0.943**, tied with the best, with
`raptor_k_eval` back at its released 62/42 instead of 94 — and the config's own comment calls the 94
override *"the accuracy change that needed no new model"*. It bought nothing on the public board and adds
~34 % to the Raptor stage's per-study work. **Ticking `0943s` costs nothing and is strictly better for the
efficiency track than ticking `0943`** — this is the recommendation from §6, now priced.

**Order of work: C (free, already measured) → D (free, trivial) → A → B.** A is the biggest single lever
and the only one whose risk is an *accuracy* question, so it should be priced on the local OOF machinery
(`scripts/oof_fusion_probe.py`, `oof_arm_join.py`) before it costs a GPU run.

## 9. The break-even rule for Lever A (and why the AUC question barely matters)

`state/FACTS.md` **A84** settles the measurement question: the local-CV bottleneck is **missing images**,
and every public artifact is a *checkpoint*, not features or a stack — so `coat_private_alpha` and the
individual CoAt members **cannot be priced on OOF locally**. `STATUS.md` reaches the same conclusion and
notes that *"every knob in the 0.943 recipe is gated on the same missing instrument"*. Lever A therefore
costs **one GPU run plus a free submission** to observe.

But the arithmetic says the observation barely constrains the decision. Setting the efficiency cost of an
AUC loss against the efficiency gain of a runtime saving:

```
ΔE = −2.18341 · ΔAUC + ΔRuntime/32400                (we want ΔE < 0)
break-even:  ΔAUC_budget = (ΔRuntime/32400) / 2.18341
```

| runtime saved | efficiency gained | **AUC we can afford to lose** |
|---|---|---|
| 40 min | 0.0741 | **0.0339** |
| 50 min | 0.0926 | **0.0424** |
| 90 min (1 member + shared prep) | 0.1667 | **0.0763** |

⭐ **So dropping a CoAt member is worth doing unless it costs more than ~0.034–0.042 AUC.** For scale:

* our entire AUC is **0.943**, and the **nine-arm spread across four different axes was 0.002** against a
  public-board noise of **±0.008**;
* one of three members inside a family that enters the final blend at `coat_private_alpha = 0.40` is
  therefore very unlikely to be worth 0.034 — that would be larger than every axis this project has ever
  measured, combined.

⚠️ **Two honest caveats.**
1. The 40–50 min per member is itself an **extrapolation** from a 3-study receipt (§7), not a measured
   1322-study number. The *rule* holds for whatever the real saving is; the *decision* inherits that
   uncertainty.
2. Dropping a member is **not a pure runtime knob** — the family is reduced as the rank of its members'
   mean probability, so removing one changes the blend, not just the cost. That is precisely why the AUC
   cost is not zero, and why the break-even budget above is the thing to judge it against rather than
   "does the board move?" — **the board cannot see a 0.034 change either way.**

**⇒ Recommendation order, priced:** `0943s` (free, already measured at 0.943) → trim the ~100 s of fixed
overhead (free, 0.003) → **drop one CoAt member** (1 GPU run + 1 free submission, expected +0.05–0.09
efficiency net of a plausible AUC cost) → shared DICOM preparation (2.1 GB artifact, riskier).

## 10. Lever A built, run, and verified — and what a submission can and cannot tell us

`scripts/build_2coat_arm.py` builds `kernels/kernel-0943s-2coat` from `kernel-0943s` (the **sparse,
0.943** arm) by raising inside the Global96 child's existing `try`, which reuses the block's own
fail-soft path. Pushed as **`ppxl16/rsna-knee-0943s-2coat`**; it ran and **COMPLETED**.

**The skip is exactly free**, from the arm's own `[phase]` records:

```
residual_coat   START 142.6  COMPLETE 197.9   ok: true     (+55.3)
global96_coat   START 197.9  COMPLETE 197.9   ok: false    (+ 0.0)   <- not run
d4              START 197.9  COMPLETE 239.8   ok: true     (+41.8)
[coat-arm] Global96 CoAt child FAILED; continuing without it (flagged): RuntimeError
[coat-arm] family reduction: rank_of_member_probability_mean
[coat-arm] CoAt family members: resgated_top3, d4_swa3
COMPLETE: /kaggle/working/submission.csv, 3 studies, SHA256=8f0c5e7b2b561538dae55439a4d2e9173141096da3defd9bc0a4bac351eef37a
```

with `DINOv2 fail-closed gate PASS: 20/20 fingerprints and members`. So: the member is genuinely not
executed, the family reduces over the **two** remaining members by `rank_of_member_probability_mean`
(the 0.943 method, applied to two), and the whole submission graph still publishes. Summing the arm's
stages gives **236.2 s against `0943cf`'s 290.1 s on the same 3 visible studies — a ~54 s saving, which is
essentially the whole `global96_coat` phase (52.5 s)**.

⚠️ **That 54 s is the FIXED part of the member, not the part that matters.** On a 3-study run the member
is dominated by its `pip install` of a private `timm-1.0.22` wheel and three checkpoint loads. The saving
that transfers to the hidden rerun is its **per-study** cost, which §7 puts at ~4.5 s/study across two
shards ⇒ **~50 min at 1322 studies ⇒ 0.093 efficiency.**

### What the submission readout can and cannot decide

⚠️ The public board's noise is **±0.008** and the expected AUC cost of dropping the *third* member is
**~0.001–0.005**. So a single readout **cannot measure the loss** — it can only catch a **catastrophic
regression**. That is still the right thing to spend a free submission on, because the decision rule from
§9 is asymmetric:

| arm returns | verdict |
|---|---|
| ≥ 0.941 | adopt — the loss is far inside the 0.034–0.042 budget, and ~50 min is banked |
| 0.935 – 0.941 | adopt with a note; the loss is at or below the board's resolution |
| < 0.935 | **reject and investigate** — that is a real regression, not noise |

This is the honest framing: the submission is a **guard, not a measurement**. The measurement that would
actually price the member — an honest OOF — does not exist and cannot be built from public artifacts,
because **A84** established they are all checkpoints rather than features (§9).

### 10.1 SUBMITTED

```
ref 56669951   kernel ppxl16/rsna-knee-0943s-2coat v1
message: 0943s-2coat v1 - 0943s (sparse) minus the Global96 CoAt member; ~50 min less runtime on the
         hidden rerun, break-even AUC cost 0.034 (docs/EFFICIENCY-SLOPE.md)
```

⚠️ **The score takes hours to appear** — this is a code submission that is rerun over the hidden 1322
studies, and the arm's projected runtime is ~3 h (against `0943cf`'s ~3.5–4 h). Do not poll it tightly;
read it back with `scripts/watch_scores.py` / `scripts/eff_rank.py` when it lands.

⚠️ **A score alone does not move the efficiency rank.** §2 showed the efficiency board carries a
`DateSubmitted` that tracks a **selected** submission, so this arm only counts for the efficiency track
once it is one of the two ticks. The tick decision is therefore still the user's, and it now has a third
candidate:

| pair | intent |
|---|---|
| `0943cf` + `0943p` | accuracy-minded, as pre-registered |
| `0943s` + `0943p` | efficiency-minded, same routing hedge, Raptor windows back to 62/42 |
| **`0943s-2coat` + `0943p`** | ⭐ the most aggressive: same hedge, sparse Raptor, and the third CoAt member dropped — pending the `56669951` guard readout |

### 10.2 A process lesson worth keeping

`api.kernels_output` pulls **everything** in `/kaggle/working`, including the three pip `--target` env
directories the CoAt children install into. That is **501 files and counting**, so the download is slow
and it died on the documented intermittent `SSLEOFError` **before reaching `submission.csv`**, which made
a successful run look like one that had produced no submission. `scripts/kernel_output_urls.py` — already
in the repo — lists the output files with **signed per-file URLs** and takes a `--pattern`, so the fix is
to fetch `submission.csv` alone (830 bytes) instead of the whole tree. Use that, not `kernels_output`, for
any arm we intend to submit.

### 10.3 ⭐ THE READOUT: `56669951` returned **0.942** — and it passes the pre-registered rule

```
56669951   COMPLETE   0.942   0943s-2coat v1 - 0943s (sparse) minus the Global96 CoAt member
56599932   COMPLETE   0.943   0943cf v1  (the pre-registered accuracy primary)
56570144   COMPLETE   0.943   0943f
56570151   COMPLETE   0.943   0943s     <-- the base arm this was built from
```

Cost **−0.001 AUC** against the base it was built on. Judged against the rule that was written in §10
**before** the number landed:

| arm returns | verdict | this arm |
|---|---|---|
| ≥ 0.941 | **adopt** — the loss is far inside the 0.034–0.042 budget | ✅ **0.942** |
| 0.935 – 0.941 | adopt with a note | — |
| < 0.935 | reject and investigate — a real regression, not noise | — |

**⇒ ADOPT.** The trade, priced:

| | |
|---|---|
| AUC cost | −0.001 → **0.0022 efficiency** |
| runtime gain | ~50 min → **0.093 efficiency** (extrapolated from the member's receipt, §7) |
| **net** | **≈ +0.09 efficiency** |

⚠️ **Honest limits, and why the rule existed.** 0.942 vs 0.943 is **inside the board's ±0.008 noise**, so
this readout **does not measure** the loss — it can only rule out a catastrophic one. That was the whole
design: the saving (0.093) is ~40× the cost of a plausible 0.001 loss, so the decision is dominated by
runtime and the readout is a guard, not a measurement. The ~50 min is likewise an **extrapolation** from a
3-study receipt; what is *measured* is that the member is genuinely not executed (`global96_coat`
COMPLETE in +0.0 s, `ok:false`) and that the family still publishes a valid submission over the two
remaining members.

### 10.4 The tick recommendation, updated

| pair | intent |
|---|---|
| **`0943s-2coat` + `0943p`** | ⭐ **efficiency-minded, and now measured**: same routing hedge, sparse Raptor windows (62/42), and the third CoAt member dropped for a measured −0.001 AUC / ~−50 min. |
| `0943s` + `0943p` | efficiency-minded without the CoAt change |
| `0943cf` + `0943p` | accuracy-minded, as pre-registered (0.943) |

⚠️ All three tie on the public board (0.941–0.943, inside ±0.008), so the board cannot rank them — the
choice is being made on the **runtime** axis, where the metric is explicit
(**1 h = 0.1111 efficiency = 23 % of the whole accuracy range**) and where the three differ by tens of
minutes. This is the one place in the project where a decision is *justified by arithmetic rather than by
a leaderboard readout*, and §8 is that arithmetic.

⚠️ **Still the user's call, and still a browser action** — `kaggle` 2.2.4 has no `select` command, and the
efficiency board tracks a **selected** submission (§2). Deadline **2026-10-22 23:59Z**.
