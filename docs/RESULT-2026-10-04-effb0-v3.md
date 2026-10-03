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

---

# ⛔ CLOSED 2026-10-04 — the EffB0-MIL line is abandoned, on the user's decision

**Do not re-open this without reading all four reasons.** Written in the style of
`state/EXPERIMENTS.md` §D so a future reader can dismiss it knowingly rather than rediscover it.

| # | reason |
|---|---|
| 1 | **Its headline rationale is falsified.** The case for this leg was that it is a meniscus / weak-label specialist — the one thing `HANDOFF.md` §97.3 recorded as an open gap ("Lateral Meniscus is the largest shortfall in the macro average and no tested knob moves it"). Re-measured per label, that is **not supported**: weakest-3 parent labels gain **+0.00414** vs strongest-3 **+0.00544**; our weakest label Fracture (0.6946) gains **+0.0013**; Lateral OA **+0.00013**; **Baker's is negative**. The effect is real in aggregate but **broad and unspecific**. (`docs/FIND-2026-10-04-effb0-honest-test.md`) |
| 2 | **The effect is below the instrument.** Best honest estimate **+0.0049** at w=0.15 (soft), +0.0087 (binary), 4/4 folds. The public board's SE is **±0.008**, and nine 0943-family arms spanned only 0.941→0.943. **We could never confirm it landed**, so it could never be safely preferred over the arm it duplicates. |
| 3 | **Cost so far is three 4-hour runs plus a full local analysis, and the remaining question is not about the budget.** v3 proved the rebuild produces **`0 slot-series` for every one of the four TRAIN studies** ⇒ `evaluable=False` ⇒ the gate **would reject even with unlimited budget**. So the v4 budget hardening (`scripts/harden_effb0_budget.py`, built and compile-verified, never pushed) **cannot land this leg**. The open question is in the builder's series selection, and answering it needs the 14 GB cache or the raw DICOMs — i.e. another GPU cycle per hypothesis. |
| 4 | **What was learned is already banked.** The pin/vocabulary defects, the receipt-vs-stage-hash acceptance rule, the UTF-16 log trap, and the honest-dose methodology are all recorded and reusable. Continuing buys no new information of that kind. |

## What this closure does NOT throw away
* **`scripts/verify_effb0_arm.py`** — the acceptance test, battle-tested (three silent-wrong-answer bugs
  found and fixed by the self-vs-self case). Reusable for ANY future arm.
* **`scripts/harden_effb0_budget.py`** — built, verified, never pushed. If the EffB0 line is ever
  revived, push it with v3's pin.
* **The honest-dose method** (`work/eff_min.py`, `work/eff_dose.py`, `work/eff_dose_parents.py`,
  `work/eff_convention.py`) — how to price a member on an outer split instead of in-sample. This is the
  transferable result, not the leg.
* **The EffB0 checkpoints and OOF remain public** (`tbongc/rsna-knee-effb0mil-sl16-tv7w`), so a future
  session can re-price it in minutes without any of the build work.

## ⚠️ The honest lesson
Three runs, three different failures, each caught by the leg's own defensive code — and the last one
showed the blocker was never the budget. The through-line is that **each round I found the failure
plausible enough to justify one more try**, and the case for the leg rested on a rationale I had not
re-tested. **The rationale should have been checked before v1, not after v3.** That check cost minutes;
the three runs cost ~5 GPU-hours.

