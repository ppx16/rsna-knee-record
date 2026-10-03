# RESULT 2026-10-04 — EffB0 v3: the pin fix WORKED, the budget prediction was RIGHT in kind and WRONG in location, and the gate found nothing to compare

Kernel `ppxl16/rsna-knee-0943cfm2eff` **v3**, COMPLETE after 1.65 h. Log decoded from UTF-16
(`work/eff_log_v3.json`).

## What worked

```
[effb0] leg weight 0.150
[effb0]   pin PASS | RULES (key/value pairs, sorted)         | expected [...] | got [...]
[effb0]   pin PASS | CACHE_SLICES (int)                      | expected 16 | got 16
[effb0]   pin PASS | IMG (int)                               | expected 224 | got 224
[effb0]   pin PASS | CROP_MM (float, 1e-9)                   | expected 130.0 | got 130.0
[effb0]   pin PASS | SLICE_BAND (tuple)                      | expected (0.12, 0.88) | got (0.12, 0.88)
[effb0]   pin PASS | slot geometry (plane, fat_suppressed)   | expected [(0,1),(1,1),(2,1),(0,0)] | got same
[effb0]   pin PASS | resolved slot indices are unique        | expected 4 | got 4
```

* **The per-term pin works and names every term** — the v2 defect is fixed, and the log is now
  diagnosable in one read instead of printing one side of a mystery.
* **Dose 0.150 confirmed** — the `kaggle kernels pull` verification was correct.
* Slot vocabulary, geometry cross-check, `slot_mask (4407,4)` fill `[0.9349,0.956,0.9857,0.9653]` all
  correct.

## The safety net fired exactly as designed

```
[effb0] LEG FAILED - continuing with kernel-0943cfm behaviour EXACTLY:
        _EffOutOfTime: convention gate [SAG_NOFS]: exceeded the 540s leg budget
[effb0] leg unavailable; _pipeline_stage.csv is EXACTLY kernel-0943cfm's output
```

⇒ Kernel COMPLETE, submission valid, **no half-built blend, no wasted submission.** Third run in a row
where the fail-safe turned a failure into a no-op.

## ⚠️ The prediction: RIGHT in kind, WRONG in location

`docs/PREDICT-2026-10-04-effb0-budget.md` predicted the leg would abandon on its budget, and it did —
but I said the overrun would be in the **test-cache build**, and it was in the **gate**:

```
[ 4212.4s] effb0-gate laterality: 2203 from the tag, 2116 from geometry, 88 unresolved;
           tag and geometry disagree on 176 (8.0% of the tagged)
[ 4246.6s] effb0-gate: 3 studies, 0 slot-series, 0.01 GiB pixels
```

**The laterality pass alone ran ~4212 s (70 minutes).** My 216–756 s figure came from C82's *corpus
build* rate; **the expensive step is scanning DICOM headers across the cohort to decide laterality**,
which I did not account for. The estimate was wrong by ~6–20×, in the right direction for the wrong
reason.

⚠️ Also note: the **gate check does cover the gate** (`since=_gate_t0`), so the overrun was caught
there. The unguarded test-cache build that v4 hardens was never reached — **v4's guard is still worth
having, but it is not what failed here.**

## ⛔ THE REAL BLOCKER — deeper than the budget, and v4 does NOT fix it

```
[effb0]   1.2.826...77125477  not built    R
[effb0]   1.2.826...49966712  not built    R
[effb0]   1.2.826...942485728 not built    L
[effb0]   1.2.826...568944496 not built    L
[effb0]   assignment slot3=SAG_NOFS: 0 studies, 0 slot-axis comparisons, 0 identical,
          0 mirror-only, 0 UNEXPLAINED, study patterns [], evaluable=False, consistent=True -> REJECTED
```

**The rebuild produced `0 slot-series` for every one of the four TRAIN studies**, so there was nothing
to compare against the frozen cache, `evaluable=False`, and the gate correctly refused.

⇒ **Even with an unlimited budget the gate would REJECT.** Raising `effb0mil_budget_s` cannot land this
leg. The rebuild is not matching any series to the four-slot layout
`['SAG_FLUID_FS','COR_FLUID_FS','AX_FLUID_FS','SAG_FLUID_NOFS']` — and the pin proves the *mapping* is
right, so the failure is in **series selection / slot population**, not in slot identity.

⚠️ Not determined: whether the rebuild found no series because (a) those particular studies genuinely
lack the layout, (b) the builder's series-matching is using the wrong corpus rules after the re-pin, or
(c) the 70-minute laterality abort left the build incomplete and "0 slot-series" is the abort summary
rather than a real count. The log order — `COMPLETE 0/0; reused ordering for 0 series` printed *before*
the exception — favours (a) or (b), but the timing (`0.01 GiB` of pixels) makes (c) hard to rule out.

**That distinction is the next thing to establish, and it is a question about the builder, not the
budget.**

## Status of the arm

`_pipeline_stage.csv` is **exactly `kernel-0943cfm`'s output** ⇒ this arm is a **duplicate** and must
**not** be submitted. (Confirms the acceptance rule: the log is the arbiter, the receipts/stage hashes
corroborate.)
