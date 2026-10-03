# PREDICTION 2026-10-04 — the EffB0 leg will most likely abandon on its own TIME BUDGET, not the gate

Written **before** `kernel-0943cfm2eff` v3 finished, so that the readout is read against a rule rather
than fitted after the fact (the discipline this repo keeps paying for).

## The code path (cell 50, verified by reading it, not inferred)

```
500  _gate_t0 = time.time()                              <- gate timing starts
521  _eff_check_budget("convention gate", since=_gate_t0)  <- covers ONLY the gate
555  _got = _eff_rebuild(_EFF_TEST_IDS, ...)               <- the TEST-CACHE build, all test studies
581  _EFF_TS, _EFF_TM = _effb0mil_build_test_cache()       <- called here
588  for _f in range(4):
589      _eff_check_budget(f"fold {_f}")                    <- the ONLY check of the cumulative leg budget
```

**The test-cache build sits outside the budget guard.** The only check before it covers the gate
alone; the cumulative leg budget is first examined at fold 0 — i.e. **after** the most expensive step
has already run.

## The numbers

* `_EFF_BUDGET_S = 540`, `_EFF_GATE_BUDGET_S = 0.6 × 540 = 324`, `_EFF_RESERVE_S = 240`.
* C82 records the corpus build at **0.2–0.7 h for 4407 studies** (CPU decode).
* The test set is ~1322 studies ⇒ **≈ 216–756 s** by proportion — *before* the 4-study gate and
  before four folds of EffNet-B0 inference over 16 slices each.

⇒ **540 s is at or below the estimate for the cache build alone.**

## Prediction

**v3 abandons with `_EffOutOfTime` at the fold-0 budget check, after the test cache has been built
and before any prediction is produced.** (Secondary possibility, much less likely: it completes.)

### Why that is SAFE, and what the real hazard is
* Leg budget exceeded ⇒ caught at fold 0 ⇒ leg abandoned ⇒ `_pipeline_stage.csv` keeps
  `kernel-0943cfm`'s exact bytes ⇒ **a valid submission, just the same 0943cfm one.** No submission
  is wasted; the arm is simply not a candidate.
* ⚠️ **The genuinely dangerous case is different**: if the cache build runs long enough to hit the
  kernel's fail-CLOSED `rsna_deadline`, the **whole kernel aborts and there is NO submission at all.**
  The leg's self-bounding exists precisely to prevent that, and the cache build is outside it. That is
  a real defect even if it does not fire this time — the pipeline runs ~3.5–4 h against
  `TIME_BUDGET = 8 h` and was pushed with `-t 34200`, so there is roughly 4 h of margin and it should
  not fire. **It should not be relied on.**

## The fix for v4 (do NOT apply to v3 — it is running)

Raise the leg budget, which is a RUN config value and needs no code change:

```python
"effb0mil_budget_s": ...   # 540 -> sized to the measured build, e.g. 2400
```

and, separately, **move a `_eff_check_budget` call to just before the test-cache build** so the leg
can abandon *before* spending the expensive step, and so an overrun can never reach `rsna_deadline`.
The second change is the one that actually matters: the first only widens the window.

## How the readout will be judged

| what the log shows | reading |
|---|---|
| `[effb0] fold 0 done (...)` then leg completes | prediction WRONG — the budget was sufficient; proceed to the acceptance test |
| `_EffOutOfTime` naming the leg budget, after `[effb0] test cache ...` | prediction RIGHT — raise `effb0mil_budget_s` and re-run; the leg itself is fine |
| `_EffOutOfTime` naming the **gate** budget (324 s) | the 4-study gate is slower than assumed — different problem, same fix shape |
| gate `REJECTED` / a slot-geometry failure | the pin fix did not hold — read the new per-term PASS/FAIL lines, which now name the failing term |
| `rsna_deadline` / kernel ERROR with no submission | the dangerous case fired — the cache build must be brought inside the guard before anything else |
