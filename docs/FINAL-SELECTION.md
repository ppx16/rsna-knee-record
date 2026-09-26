# Final submission selection — the one manual step left

**Deadline: 2026-10-22 23:59 UTC.** Everything else on this project is closed by measurement.
This is the only action that remains, and it is a **two-click job in the browser** — the Kaggle CLI
cannot do it (verified on 2.2.4: `kaggle competitions` has `list/files/download/submit/submissions/
leaderboard/team-submissions/submission-limits`, and no `select`).

---

## Current state — Kaggle picked two on its own, and it is not the pair we want

As of 2026-09-26 the Submissions page has **`RSNA knee 0943 - Version 2`** (`56568520`, 0.943) and
**`RSNA knee 0943f - Version 1`** (`56570144`, 0.943) ticked. It did **not** tick `0943s` (also
0.943), so the tie-break among three equal scores is not something we can predict — which is exactly
why this cannot be left to the default.

### Make these two clicks

1. **Untick** `RSNA knee 0943 - Version 2` (0.943).
2. **Tick** `RSNA knee 0943h - Version 2` (0.942).

Leave the `0943f` row ticked. Then confirm the page shows **`56570144` + `56568524`** as selected.

| # | ref | arm | public |
|---|---|---|---|
| 1 | **`56570144`** | `0943f` — fracture fork | 0.943 |
| 2 | **`56568524`** | `0943h` — halfway routing | 0.942 |

**If you would rather not touch it**, leaving Kaggle's pair is defensible — see the last section for
exactly what that gives up. Nothing else in this repository depends on the choice.

---

## Why this pair — when three arms scored 0.943 and two scored 0.942

**The board cannot rank these arms, so it was not used to.** Public noise is ±0.008 and the effects
in play are 0.001–0.003 (A96). All five arms sit inside the noise band. The choice rests on evidence
that is not the displayed score:

1. **Kaggle's default pair spends its second slot on a near-duplicate.** `0943f` is `0943` with a
   single token changed (`_RAD_EXCLUDE` no longer excludes `Fracture`), so the two differ in **one of
   twelve columns** — and that token's measured effect is +0.00099 macro in `0943f`'s favour
   (n=4185, positive in 4 of 5 folds). Under "best of the two counts", `0943` is close to a dominated
   version of `0943f`: the second slot buys almost the same bet twice. **Starting from the default
   pair, the fix is to make the second slot a different axis, not a nested variant.**

   ⚠️ This reason does **not** depend on the routing argument below. Even if `0943h` had exactly the
   same expected score as `0943`, replacing a near-duplicate with an equal-mean, different-axis arm
   still raises E[max of the two].

2. **`0943f` is the only arm whose change carries large-sample, non-public evidence.** Dropping
   `'Fracture'` from `_RAD_EXCLUDE` was re-measured in the published analysis at **n=4185, 156
   positives**: Fracture AUC 0.9187 → 0.9305, **macro +0.00099, positive in 4 of 5 folds**, reproduced
   by five independent label sources, with a single-peaked alpha curve (no new constant). By the rule
   this repository measured (A111: a tuning set needs ≈870 studies before per-label choices are
   safe), n=4185 is far past the bar.

3. **`0943h` hedges the one knob family that is documented as public-fitted — by its own author.**
   This no longer rests on our note about the frontier, or on an analogy. The frontier notebook says
   it in its own source comments, above the weights themselves:

   > `# stages, on the strength of a public split that is a fraction of the test`
   > `# set. It may be real. It is also the likeliest place to give back points`
   > `# privately, which is what the "parent" preset is for.`

   …and, at the diagnostic that prints per-finding arm agreement:

   > `actually decides the ordering — and therefore where a weight fitted to the`
   > `public split can do real damage privately:`

   So: the per-label outer routing (`coatnet_w`) was fitted to a public split that is **a fraction of
   the test set**, the author flags it as **the likeliest place to lose private points**, and the
   author's own remedy is a flattened preset. `0943h` softens that map halfway toward its default —
   the same direction, the milder dose (C57 measured softening as the cheaper of the two).

   ⚠️ **Honest limits.** This establishes the *direction*, not the *dose*: nothing we own can rank
   "soften halfway" against "flatten completely", because the only two numbers we have for that
   (C57's 0.001 and 0.002) were both measured on the public board — the very set the weights were
   fitted to — so they are biased the same way. And A111's −0.0013 for per-label versus one shared
   weight is an **isomorphism** (measured on the A5↔Rad blend weight, not on `coatnet_w`): the
   reasoning transfers, the number does not. That is why `0943h` is the second slot and not the
   first — it needs the direction to be right, not the dose.

4. **Under a "best of the two counts" rule, dispersion is worth more than a second copy of the same
   bet.** Neither arm covers both axes, but **the union does**: if either change is real, one of the
   two picks collects it, and if the routing hedge is wrong, `0943f` still holds the measured gain.

⚠️ Reason 4 depends on the widely-reported "better of the two selected" rule. **It could not be
verified from this machine** (web fetching is unavailable here). The pair is defensible on reasons
1–3 alone, so nothing hinges on it — but if you know the rule to be different, reason 1 is the one to
keep.

**If the interface allows only one selection, or only one is scored: pick `0943f` (`56570144`)** —
reason 2 is the hardest evidence in this table.

---

## What leaving the default costs, stated plainly

Keeping Kaggle's pair gives you `0943f` (the arm with the real evidence) **plus a second slot spent on
an arm that differs from it in one column**. You keep the exact published recipe and take no bet
against it. You give up the second slot's option value — which is small, and unmeasurable, and on the
private board sits inside ±0.005 of noise either way.

**Not building a "flatten completely" arm either.** The frontier's `parent` preset would be a few
lines to add, one submission slot, and ~0.1 GPU-hours — the cost is not the obstacle. The obstacle is
that **its readout cannot inform anything**: `coatnet_w`'s effect is only observable on the public
board, and the public board is the set those weights were fitted to. Submitting it would be
"submit and see" (A96), not confirmation of an evidenced improvement.

---

## What is already banked (no action needed)

- **All five arms ran the hidden rerun end to end** on 1322 studies and came back `COMPLETE` with a
  score — not just the baseline, but both non-default presets and the reduced-window arm. The
  pipeline had only ever been exercised on the 3 visible studies. De-risking that is independent of
  the score.
- **Window count is not a score lever**: `0943s` (62/42 windows) tied the 94-window config at 0.943.
  It is a *quota* lever (~5× cheaper) if this pipeline is ever run again.
- **Ledger:** `state/lb_ground_truth.csv` — 32 rows, best 0.943. Regenerate with
  `python scripts/lb_ledger.py`; it is generated from the competition's submission list, not by hand.
- **Quota:** GPU 2.24 h / 30 h, TPU 0 h / 20 h, refresh 2026-10-03. Selecting a final submission
  spends no quota — it is a flag on a submission that has already been scored.
