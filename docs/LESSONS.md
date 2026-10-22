# LESSONS — the reasoning failures, and the rules they produced

This file is not about the score. It is about the thirteen times this project's *thinking* failed in a way that
a record could have caught, and the specific rule each failure produced. It is written after the models were
converged, because that is the only position from which the pattern is visible.

The score moved 0.943 → 0.958 in twelve days. Almost nothing in that climb came from an idea of ours: it came
from finding published work one step ahead and reproducing it exactly. What this project actually produced
is a set of falsifications and a set of instruments.

---

## I. Five predictions, five falsifications

Each of these was written down before the measurement. That is the only reason they count as falsifications
rather than as hindsight.

| # | Prediction | What happened | The rule it produced |
|---|---|---|---|
| 1 | Adding nartaa's S6 all-data arm would cost a thousandth (C195, on an independent team's report) | 0.958 at shares 0.40 and 0.60 — no visible cost. It took a 0.80 share to reproduce their −0.001 | **An external result may be true at a different point on the same axis.** "They measured X" is not "X holds at my setting" |
| 2 | Splitting the 384 arm's rank into plain + TTA would cost a thousandth, because the identical change cost one on the 224 arm (C204) | 0.958 — no cost at all | **Bind a prediction to the arm tested.** Same mechanism, different arm, different answer |
| 3 | heliosli's 0.9586 recipe, which uses flat weights + a reader, would display 0.957 (C203, from another team's version history) | 0.958 | **Reading one implementation's number to predict another implementation's number is not inference, it is analogy** |
| 4 | The 224 plain/TTA split would help slightly: the mechanism was clean, the source gave the exact weights (C197) | 0.957 — it cost a thousandth | **"The mechanism is sound" is not "the direction is right."** The source itself called it an unvalidated hypothesis; I graded it "weakly positive" |
| 5 | The public bronze margin would reach about −38 by the deadline (C208) | Refined to ≈ +5 once measured on a recent window: the −2.9/h rate had been estimated over a stretch that included the fastest inflation of the campaign (C210) | **Estimate rates on a window from the same regime.** A trend that averages an acceleration is not a trend |

**The common shape:** every one of these was a plausible, mechanism-supported, literature-anchored prediction
— and the answer was decided by a measurement of one thousandth, which this board cannot resolve anyway.
The predictions were not wrong because the reasoning was lazy. They were wrong because **the object they
were about is not measurable here**.

## II. Eight instruments that reported success without seeing anything

This is the more expensive failure class, because it corrupts the record rather than the plan.

1. **A "faithful" equivalence test that was scale-invariant** — it could not fail, so its pass meant nothing.
2. **A log decoder that turned two successful runs into FAILs** — it read a field that is absent on success.
3. **A submission pipeline that passed every gate while producing three all-zero rows** — the gates checked
   shape and identity, never that the numbers were predictions.
4. **A swap script whose comparison ran against a made-up 0.0** — the outgoing score was read from a field
   that does not exist, so "0.950 > 0.000" passed for the wrong reason. The decision was right; the log was empty.
5. **A fold log that hard-coded its own weight** — `folded ... at 0.4` was a string, so changing the
   arithmetic left a log claiming the old number. A log maintained separately from the computation becomes
   false evidence.
6. **A deadline check that went red on every run** — it counted append-only logs as uncommitted work, so it
   failed even when everything was fine. Three separate files had to be whitelisted.
7. **An audit that examined zero claims and reported success** — the first version of the ARC number audit
   matched nothing, printed `claims examined: 0`, and concluded "every claim agrees". Fixed by giving it a
   self-test that must match known bad probes, and an exit code of 2 when it examined nothing. **A check that
   cannot fail is not a check** — this one was written *while documenting that rule*.
8. **An audit pointed at the wrong repository** — the same script resolved its root from `__file__` and lives
   in the working tree, not the ledger checkout, so it scanned five unrelated files. Fixed by naming the path
   explicitly and refusing to run if `findings/` is not there.

**The rule:** an instrument must be shown to **pass on a known-good input and fail on a known-bad one**. If it
has never failed, it has never been tested. And a check that always fails will train its reader to ignore it —
which is worse than no check.

## III. What actually worked

1. **Reading other people's prose, not their code.** The one component our base was missing — the anatomical
   mirror on the 224 arm — was found by reading a notebook's markdown that said *"the 224-pixel checkpoint was
   published with one view only"*. Nobody's code comments say that.
2. **Version history as a controlled experiment.** One author's own v21/22/24 → v25/26 changed exactly one
   thing (adding a set of per-finding weights) and moved 0.957 → 0.958. That is the closest thing to an
   ablation the public surface offers, and it is worth more than any single notebook's claimed score.
3. **Exhaustive negative accounting.** Nine submitted-and-scored variants all landing at 0.957 or 0.958 is
   what turned "we cannot find an improvement" into "the board cannot resolve one" — a much stronger and more
   useful statement.
4. **The single source of truth, enforced by a script.** Once the tick lived in one JSON file, the failure
   mode became *documents drifting away from it*. `audit_docs_tick.py` found a stale instruction to retreat to
   a 0.954 artifact, a line asserting a policy we had deliberately reversed, and two runbook copies 49 lines
   apart — none of which any human read would have caught at 07:59 on a deadline morning.

## IV. The four rules worth carrying forward

1. **Record the prediction before the measurement, and bind it to the thing measured.** A prediction about
   "the same mechanism elsewhere" is not a prediction.
2. **Grade the evidence, not just the claim.** Measured / mechanism-only / unvalidated / asserted-by-a-title
   are four different things. The one time I read "an unvalidated hypothesis" as "weakly positive", it cost a
   submission.
3. **Make the source of truth a machine-readable file, then check every other document against it.** Prose
   drifts silently; JSON does not.
4. **When the instrument's noise exceeds the effect, stop proposing effects.** The public board here has a
   noise floor of roughly 0.003 across a band 1,500 teams wide; the effects we were chasing were 0.001. Every
   hour spent designing the next such experiment was an hour spent learning that the previous ones could not
   have worked.
