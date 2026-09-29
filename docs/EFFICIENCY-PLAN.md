# RSNA Knee — the Efficiency Track: reconnaissance and plan (2026-09-28)

`$18,000`, 3 places. **The same submission counts for both tracks**, so this is free upside that the
project spent two weeks not looking at. We sit at **eff#3343 / 4363** while our main LB is **rank 430
/ 4479**.

## The metric, and why minutes matter

```
Efficiency = AUC / (Benchmark − max AUC) + RuntimeSeconds / 32400        (lower is better)
```

`Benchmark` is `sample_submission.csv`, which **we measured**: every value is 0.5, so `Benchmark = 0.5`
and the denominator is **−0.459** (negative, so more AUC is better and less runtime is better).

| runtime | worth in AUC |
|---|---|
| 1 minute | 0.00085 |
| **1 hour** | **0.0510** |
| 3 hours | 0.153 |

⇒ **One hour of runtime costs more than the entire 0.016 gap between us and the leaderboard top.**
AUC and runtime are commensurable and both matter; neither can be waved away.

## What we CAN and CANNOT see

* ⚠️ **Our scored runtime is not observable.** The metric is computed on Kaggle's hidden **1322-study**
  rerun; the log we can download is from the **3 visible studies**.
* The efficiency CSV (`ryanholbrook/rsna-knee-abnormalities-efficiency-lb` → `full_leaderboard.csv`)
  carries only `EfficiencyRank, TeamName, PublicScore, DateSubmitted` — **no efficiency value**, so the
  formula cannot be inverted for an absolute runtime.
* ⭐ But one structural fact is derivable and useful: **within a fixed AUC, the rank ordering is pure
  runtime.** 266 teams share our 0.943; the best of them is **eff#14** and we are **eff#3343**. So the
  gap between us and eff#14 is *entirely* runtime, at our own accuracy.

## Where the 326 seconds actually go (3 visible studies, `0943cf`)

Parsed from `state/logs_0943cf.json` (`[phase]` records + stage prints):

| # | stage | ends at | block | scaling |
|---|---|---|---|---|
| 1 | imports, EDA tables, correlations | 29 s | **29 s** | **FIXED** |
| 2 | input/competition preflight, cache sizing | 32 s | 3 s | fixed |
| 3 | DINOv2: 20 **fingerprint verifications** | 64→104 s | **~40 s** | **FIXED** |
| 3b | DINOv2: decode + shared first-6 blocks + 20 members | 108 s | ~7 s | per-series |
| 4 | A5: strict-load 5 folds + inference | 100→129 s | ~20 s | mostly fixed load |
| 5 | RadImageNet (e10 / e13 / legacy_second) | 129→141 s | ~12 s | mostly fixed load |
| 6 | **raptor-fast** (loops over studies) | 141→177 s | **37 s** | **PER-STUDY** |
| 7 | **CoAt family** (resgated/global96/d4_swa3) | 177→297 s | **~148 s** | **PER-STUDY?** |
| 8 | assemble + publish submission.csv | 297→326 s | ~30 s | fixed |

**Read:** roughly **100 s of the 326 s is pure fixed overhead** — EDA (29 s), 20 weight-fingerprint
re-verifications (~40 s), publish (~30 s). The two stages that loop over studies are **raptor (37 s)**
and **CoAt (~148 s)** for 3 studies.

⚠️ **The one thing we still do NOT know is the per-study slope** — i.e. how the 1322-study rerun
composes. The CoAt block is either ~150 s of fixed checkpoint loading or ~50 s *per study*; those two
readings imply a full-run runtime anywhere from **~20 minutes to ~18 hours**, and every decision
depends on which it is. No existing log answers it: the 870-study run in `scratch/.testtmp/d64tpu.log`
is a **fine-tune**, not inference.

## The plan

**Step 1 — measure the slope (one GPU kernel, the decisive experiment).**
The notebook already honours `os.environ['RSNA_COMP_ROOT']` and reads `COMP/'test_series'`. So we can
point it at a **synthetic test root built from TRAINING studies** and run the identical pipeline at
N = 3, 10, 30, giving seconds-per-study per stage and a projection to 1322. That converts every
subsequent decision from guesswork into arithmetic, and it also tells us which single stage to attack.

**Step 2 — cut the fixed overhead (no risk to AUC).**
Skip the EDA block for the submission run (it is analysis, not inference); stop re-verifying 20
fingerprints that cannot change between runs; trim the publish step. ~100 s of 326 s on the visible
run, and pure free time on the rerun.

**Step 3 — attack whatever Step 1 shows is dominant.** Candidate levers, all of which must leave the
predictions bit-identical: raise batch/chunk sizes, overlap study I/O with GPU compute, avoid
reloading shared backbones, and use both T4s where the current stages are single-device.

**Step 4 — make sure the RIGHT submission is the tracked one.**
The efficiency board currently shows `DateSubmitted = Sat Sep 26 05:24:23 2026` = submission
**`56570144` (0943f)** — *not* our latest. So we must establish which rule picks the tracked
submission (best score? selected? last?) before we assume a faster arm will move our efficiency rank.
`scripts/eff_rank.py` prints this date prominently; re-running it after each submission resolves the
rule empirically.

## Blockers

* ✅ **RESOLVED:** `kernel-tpu-msft` finished (cancelled at the 12 h cap, `"partial": true`, 3 of 4
  epochs) and **failed its pre-registered bar**: `best_gold_auc` **0.8968** vs the required > **0.9214**.
  It was below the parent from epoch 0 (0.8795), i.e. it **damaged** the warm start rather than
  under-training — rejected on mechanism. ⚠️ It also **never used a TPU**: its metadata said
  `enable_gpu: true, enable_tpu: false`, so it ran on a T4 for **10.7 h** and spent ~⅓ of the weekly
  GPU quota. See `RESULT-2026-09-28-msft.md`.
* ⚠️ **The Kaggle auth is a landmine, now diagnosed.** Three facts combine badly:
  1. `~/.kaggle/kaggle.json` is **STALE** — authenticating with it *succeeds*, but the first real API
     call returns **401 Unauthorized** (`kaggle.json.bak-20260914` sits beside it).
  2. The credential that actually works is **`~/.kaggle/access_token`**, and that path makes a
     **network call** (`_introspect_token`) on every authentication.
  3. ⚠️ **`kaggle/__init__.py` runs `api.authenticate()` at IMPORT time**, so
     `from kaggle.api.kaggle_api_extended import KaggleApi` can raise `SSLError`/`SSLEOFError` *from
     the import itself* — outside any `try` that only wraps the API object.
  ⇒ **(a)** Do **not** "fix" this by deleting `access_token`: that was tried and it drops us to the
  stale `kaggle.json`, i.e. 401s. **(b)** Always put the **import inside the retry loop**, not just the
  call. **(c)** The failures are intermittent (one attempt failed, the next succeeded), so retrying is
  the correct response and there is no config change worth making. `scripts/watch_kernel.py` does (b).
