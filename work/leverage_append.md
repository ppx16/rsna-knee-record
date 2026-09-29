
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
