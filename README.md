# RSNA Knee Abnormality Detection — working record

A public record of one Kaggle **code competition**: 12-label macro ROC-AUC, 1,322 hidden studies,
deadline **2026-10-22**. It is written as a lab notebook, not a tutorial: the measurements, the
falsifications, and the places where I was wrong are all kept.

**Final standing: displayed score 0.958, our best artifact, held from 2026-10-16.**
At the last reading the 0.958 band held **~1,475 teams** and we sat at about **323/1,475** inside it —
inside the bronze cut, outside silver. No public notebook ever displayed higher than 0.958.

---

## The one-paragraph version

We climbed 0.943 → 0.950 → 0.954 → 0.956 → **0.958** in twelve days, mostly by finding public artifacts
one step ahead of us and reproducing them exactly rather than by training anything. When the public
surface ran out, we spent the remaining days measuring what could *not* be improved, which turned out to
be almost everything: at 0.958 the public leaderboard cannot resolve a thousandth, and every knob we
turned either did nothing or cost exactly that thousandth. The durable output of the last week is
therefore a set of **falsifications** rather than a higher score, and a deadline protocol that does not
depend on anyone remembering anything at 07:59 on a Friday.

---

## The rule this repository runs on

> **A conclusion is only as good as the instrument that produced it, and an instrument is only trusted
> after it has been shown to pass on a known-good input and fail on a known-bad one.**

Three times a check here reported a confident verdict it could not actually see: a scale-invariant
"faithful" test, a log decoder that turned two successful runs into FAILs, and a submission pipeline that
passed every gate while producing three all-zero rows. All three are in the record with the controls that
now catch them.

A fourth failure mode took longer to learn: **reading a claim as a measurement**. The leaderboard shows
`bestPublicScore`; the notebook title shows what its author hopes it is. When a notebook called
`…=0.9586` turns out to display 0.958, the honest reading is the board, not the title.

---

## What actually moved the number

| Date | Displayed | What changed |
|---|---|---|
| 10-09 → 10-10 | 0.943 → 0.950 | public Raptor CoAtNet arms |
| 10-11 | 0.954 | heliosli's restored "blend gold" |
| 10-13 | 0.956 | the R7 compact bundle, then the R6 bundle pinned to version 1 |
| **10-16** | **0.958** | **the base recipe was replaced, not tuned**: R6 frozen bundle + flip TTA + per-finding Raptor weights (`Medial OA 0.50, Lateral OA 0.20, PF OA 0.60, Synovitis 0.80, Baker's 0.35`) + `A224 = 0.35` |

The last step is the instructive one: it was not any single arm getting better, it was the **blend being
republished** by its author with a different weighting. Two days later, an independent author's version
history confirmed the same thing from the other side — their versions 21/22/24 (same models, flat weights)
scored 0.957 and their 25/26 (same models, the weights above) scored 0.958.

---

## What did not work — the measured negatives

Every one of these was submitted and scored, not argued about:

| Change, on the 0.958 base | Displayed |
|---|---|
| `A224 = 0.50` / `0.65` | 0.957 / 0.957 |
| `r9all` leg at 0.10 | 0.957 |
| `band32` leg at 0.08 / 0.15 | 0.957 / 0.956 |
| per-finding weights flattened to 0.5 (the un-fitted twin) | 0.957 |
| nartaa's S6 all-data arm at **0.80** of the compact half | 0.957 |
| plain/TTA rank split on the **224** arm | 0.957 |
| S6 arm at **0.40 / 0.60** | 0.958 (i.e. no effect) |
| anatomical mirror on the **224** arm | 0.958 |
| plain/TTA rank split on the **384** arm | 0.958 |

Seven artifacts display 0.958 and nine display 0.957. **The spread between "the best public recipe" and
"the best public recipe plus any change we can think of" is one band, in both directions.**

---

## Why the last thousandth is unreachable, not merely hard

- The visible test set is **3 studies**; the hidden test is 1,322. Scoring reruns the kernel, so the
  public score is one sample of an unrounded value, and **the unrounded value moves between reruns**.
- The published analysis of this competition puts the public half's standard error near **0.003** —
  larger than the entire spread of the top 400 teams.
- The band we are in holds **~1,475 teams**. Inside it, rank is not a measurement: our own rank moved
  **231 → 419 → 264** across days on which we submitted nothing, because reruns reshuffle ties.
- Therefore any claim of the form "component X is worth +0.001" is unfalsifiable on this board — and
  after three such claims of mine were falsified by the next submission, that is the standing position.

An independent author reached the same conclusion in their own write-up, almost word for word:
*"Without controlled ablation experiments, the contribution of each optional branch cannot be established
independently … Public leaderboard progression does not establish private leaderboard robustness."*

---

## The final artifacts

Only two submissions are selected at the deadline, and the platform keeps the better of them.

| Slot | ref | Score | Lineage |
|---|---|---|---|
| 1 | `57250245` | **0.958** | OAI-supervised Raptor arms (acknowledgement included as the rules require) |
| 2 | `56878092` | 0.944 | **non-OAI**, our own stack — insurance if the OAI-lineage models are ruled unusable |

Slot 1 is deliberately the artifact whose recipe is **entirely someone else's published work**, with no
addition of ours. When five artifacts tie at 0.958 and the tiebreak is unmeasurable, the defensible
choice is the least self-fitted one, not the most elaborate.

The protocol for the deadline is one command:

```powershell
python work\deadline_report.py     # exit 0 = safe to tick; prints the two refs and their descriptions
```

It re-verifies the pair against the live server, checks both ledgers are committed and pushed, reads the
board, and prints the opening and closing words of each description — because twenty-odd of our
submissions display 0.942/0.943 and the platform's default selection is not ours.

---

## Layout

```
README.md      this file
docs/          the written record: handovers, pre-registrations, findings, the deadline runbooks
kernels/       every Kaggle kernel we own, one directory each
scripts/       reusable tools
state/         FACTS.md (the ledger), final_pair.json (the single source of truth for the tick), logs
downloads/     pulled artifacts (evidence only)
archive/       one-off scrap kept for provenance
```

## Where to read

| Path | What it is |
|---|---|
| `state/FACTS.md` | **The ledger.** Every conclusion, numbered — `C1..` for measured findings on the models and the board, `A1..` for facts about the tooling. Append-only: a wrong entry is superseded, never edited. |
| `docs/HANDOVER-2026-10-2*.md` | The last three days, written for a successor with no memory of this session. |
| `docs/DEADLINE-CHECKLIST.md` | The five-minute deadline procedure, and the list of things **not** to do, each with the measurement behind it. |
| `docs/FINAL-TICK.md` | The one manual step: which two submissions to tick, and how to recognise them in the UI. |
| `docs/` (older) | The earlier phase: pre-registrations, falsified plans, and the pre-0.950 work. |

## Related

[`ppx16/arc-agi-3-ledger`](https://github.com/ppx16/arc-agi-3-ledger) — the simultaneous ARC Prize 2026
Milestone 2 campaign, where the objective is different in a way worth stating: the score is the **max**
over all of a team's submissions, so what matters is the **right tail**, not the mean. One draw per UTC
day, and the only rational policy is to never miss a day.

---

## What this project actually produced

Not the score — the score came almost entirely from finding published work one step ahead and reproducing it
exactly. What we produced is **[`docs/LESSONS.md`](docs/LESSONS.md)**: five predictions that were written
down before measurement and then falsified, six instruments that reported success without seeing anything,
and the four rules those two lists generated.

The short version: *when the instrument's noise exceeds the effect, stop proposing effects.* It took eleven
failures to believe that.

---

## Standing policies

**Submissions.** A submission is for **confirming** an improvement that local evidence already supports —
never for discovering whether one exists. When the board cannot resolve the effect being chased, "submit
and see" is multiple comparisons on noise.

**Recording.** Write facts at the moment they are measured. An unrecorded measurement is one we pay for
twice. A new fact appends a row; a correction appends a row that says so and cross-references the old one.

**Diagnosis.** Diagnose one cause, fix that cause, and be able to say which cause an edit addressed. A
diagnosis that lists three causes is not resolved by fixing one of them.
