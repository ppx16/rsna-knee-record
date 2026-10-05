# FIND 2026-10-05 — the final-selection assumption was never verified; now it is. And the rules gave up four other facts.

**Trigger**: auditing `docs/FINAL-SELECTION-*.md` turned up an **operational assumption nobody had checked** —
the sheets say *"two submissions may be selected"* and describe a tick UI, but **never establish which
submissions are eligible to be ticked.** `56613303` (`0943cflat`), the recommended second slot, was
submitted **2026-09-27**. If the pool were "the most recent N submissions", the entire recommendation
would be worthless and we would discover it at the deadline.

**Answer: the pool is unrestricted. The recommendation stands.** But it was luck, not diligence, that
it was correct — so the check is now on the record.

Source: `kaggle competitions pages` via `KaggleApi.competition_list_pages`, saved verbatim under
`work/comp_pages/` (the CLI's own `competitions pages` verb has an argparse ordering that contradicts
its usage string — `pages list <comp>` reads the comp as the subcommand, `pages <comp> list` answers
"No competition specified" — so the Python API is the route).

---

## 1. The rule, verbatim

> **2. SUBMISSION LIMITS**
> a. You may submit a maximum of five (5) Submissions per day.
> b. **You may select up to two (2) Final Submissions for judging.**
> c. For Hackathons, each Team may submit one (1) Submission only.

and the definition:

> A **"Final Submission"** is the Submission **selected by the user**, or automatically selected by
> Kaggle in the event not selected by the user, that is/are used for final placement on the
> competition leaderboard.

**Searches for a pool restriction returned ZERO hits** across the whole 37 kB rules page:
`most recent` 0 · `last two` 0 · `last 2` 0 · `previous submission` 0 · `only the` 0 · `within the last` 0.

⇒ **Selection is from the full submission history.** An 8-day-old arm is as selectable as yesterday's.
⇒ The "up to two" is a cap on how many **may be selected**, not on how far back one may reach.

## 2. Four other facts the rules settled (all previously assumed or mis-stated)

| fact | value | why it matters |
|---|---|---|
| **Metric** | `Final Score = (1/12) Σ AUCᵢ` — macro-averaged ROC AUC over the twelve targets | confirms the instrument the whole repo is built on |
| **Run-time cap** | **CPU Notebook ≤ 9 h; GPU Notebook ≤ 9 h**; internet **disabled**; file must be **`submission.csv`** | ⚠️ the earlier note "21600 s = 6 h" was not even the right number for this competition — the rules say **9 h** |
| **Entry Deadline** | **2026-10-15** — "You must accept the competition rules before this date in order to compete" | we are already on the board, so satisfied; but it is a *hard* gate, not a soft one |
| **Winners' Requirement Deadline** | **2026-11-05** — "the deadline for winners to submit to the host/Kaggle their **training code, video and method description**" | a real post-deadline obligation **if** we place in a prize position |

⚠️ **The 9 h figure is worth keeping next to the quota correction.** That correction
(`FIND-2026-10-05-quota-denominator-error.md`) established that the *weekly* GPU allowance is **30 h**
and that I had been quoting a **6 h** number as if it were that total. The rules now show the
per-session cap for this competition is **9 h**, so the 6 h figure was wrong for *both* readings. The
numerator `19664 s = 5.46 h used` is what was ever correct.

## 3. What this changes

**Nothing in the recommendation — it retires a risk instead of moving a decision.**

`docs/FINAL-SELECTION-2026-10-05.md` recommends **`56814081` (`0943cfmsft2`) primary +
`56613303` (`0943cflat`) second**, and both remain selectable. The sheet previously rested on an
unstated assumption; that assumption is now a **verified premise** with the source quoted.

**The generalisable lesson, which is the third time this session it has applied:** a plan that depends
on an external system's behaviour must quote the system's own text for that behaviour. "Two submissions
may be selected" was read from these same rules months ago and the *pool* half of the sentence was
never read at all — the same shape as reading `CROP_MM`'s declaration instead of its winning assignment,
and `F_dim=768`'s default instead of its call site.

Tooling: `work/fetch_rules.py` (saves all 10 pages, extracts selection sentences).
Evidence: `work/comp_pages/*.txt`.
