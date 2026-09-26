# Final submission selection — the one manual step left

**Deadline: 2026-10-22 23:59 UTC.** Everything else on this project is closed by measurement.
This is the only action that remains, and it is a **two-click job in the browser** — the Kaggle CLI
cannot do it (verified on 2.2.4: `kaggle competitions` has `list/files/download/submit/submissions/
leaderboard/team-submissions/submission-limits`, and no `select`).

---

## Tick these two rows, on the competition's **Submissions** page

| # | ref | arm | public |
|---|---|---|---|
| 1 | **`56570144`** | `0943f` — fracture fork | 0.943 |
| 2 | **`56568524`** | `0943h` — halfway routing | 0.942 |

Up to two submissions may be selected; each row carries its own select/tick control. After ticking,
**check that the page shows these two as the selected ones** — that display, not this file, is the
authoritative state.

⚠️ **Do not skip this.** If nothing is selected, Kaggle auto-selects **by public score**, and three
arms are tied at 0.943 (`56568520`, `56570144`, `56570151`), so which two it would take is not
something we can predict — and it would not be this pair.

---

## Why this pair, when the board says three other arms scored the same

**The board cannot rank these arms, so it was not used to.** Public noise is ±0.008; the effects in
play are 0.001–0.003 (A96). All five arms sit inside the noise band. The choice rests on evidence
that is not the displayed score:

1. **`0943f` is the only arm whose change carries large-sample, non-public evidence.** Dropping
   `'Fracture'` from `_RAD_EXCLUDE` was re-measured in the published analysis at **n=4185, 156
   positives**: Fracture AUC 0.9187 → 0.9305, **macro +0.00099, positive in 4 of 5 folds**, reproduced
   by five independent label sources, with a single-peaked alpha curve (no new constant). By the
   rule this repository measured (A111: a tuning set needs ≈870 studies before per-label choices are
   safe), n=4185 is far past the bar.
2. **`0943h` was pre-registered as a hedge, before any readout existed** (the arm's docstring is dated
   2026-09-25). It softens the one knob family that is *known* to have been tuned on the public split
   — the per-label outer CoAtNet routing, at ≈400 studies, exactly the regime where A111 measured
   same-sample reporting to be optimistic by +0.0028 and per-label tuning to transfer **worse** than a
   single shared weight by −0.0013. `outer060` hedges the same knob family more weakly; `halfway` is
   the better-supported direction (C57: softening cost 0.001 where flattening cost 0.002).
3. **Under a "best of the two selected counts" rule, dispersion is worth more than a second copy of
   the same bet.** `0943f` + `0943h` spread across two independent uncertainty axes (which labels the
   Rad arm is blended into × how sharp the outer routing is); `0943f` + `0943` would spread across one.

⚠️ Two honest caveats:

- Reason 3 depends on the widely-reported "better of the two selected" rule. **It could not be
  verified from this machine** (kaggle.com is not fetchable here). It is safe, because the pair is
  defensible on reasons 1 and 2 alone.
- **If the interface allows only one selection, or only one is scored: pick `0943f` (`56570144`)** —
  reason 1 is the hardest evidence in this table.

---

## What is already banked (no action needed)

- **All five arms ran the hidden rerun end to end** on 1322 studies and came back `COMPLETE` with a
  score — not just the baseline, but both non-default presets and the reduced-window arm. The
  pipeline had only ever been exercised on the 3 visible studies. That de-risking is independent of
  the score.
- **Ledger:** `state/lb_ground_truth.csv` — 32 rows, best 0.943. Regenerate any time with
  `python scripts/lb_ledger.py`; it is generated from the competition's submission list, not by hand.
- **Quota:** GPU 1.58 h / 30 h, TPU 0 h / 20 h, refresh 2026-10-03. Selecting a final submission
  spends no quota — it is a flag on a submission that has already been scored.
