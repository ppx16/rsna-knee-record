# FIND 2026-10-05 — the rank measures your BEST submission, not your newest one

**This changes how every rank reading in this repo must be interpreted, and it corrects a hypothesis I
formed an hour ago and had to drop.** The data was already on disk: `state/rank_track.jsonl`.

---

## 1. The clarification

Kaggle's public leaderboard shows a team at the rank of its **best** public submission. So:

> **Submitting an arm that does not become your new best moves your rank by exactly nothing —
> no matter how good that arm is.**

Consequences, all of which follow directly:

* The rank instrument answers **"did this arm become our new best, and by how much?"** It does *not*
  answer "how good is this arm?".
* An arm that beats our second-best but not our best is **invisible** to it. Absence of a rank move is
  therefore **not** evidence that an arm is worthless — only that it did not take the top spot.
* Conversely a rank move **is** strong evidence: to move, the arm's *true* score (finer than the 3
  displayed decimals) had to exceed every previous submission's.

## 2. This independently corroborates the corrected final-selection sheet

Our best is therefore readable off the board, without trusting any local instrument:

| ref | arm | measured in-block move | ⇒ is it our best? |
|---|---|---|---|
| `56814081` | `0943cfmsft2` | 38.89 % → **13.66 %** (better) | ✅ **yes** |
| `56826372` | `0943cfmsft2m` | 13.67 % → 14.31 % (slightly worse) | ❌ no |

⇒ **Our displayed best is `0943cfmsft2` (`56814081`)** — which is exactly the primary that
`docs/FINAL-SELECTION-2026-10-05.md` recommends, arrived at there from the evidence each arm carries.
Two independent routes, same answer.

## 3. ⚠️ A hypothesis I formed and had to withdraw

Having noticed that FACTS **D14** closed the "add a 5th view" axis on the grounds that *"在公榜分辨率下这条
路读不出东西"* — the very premise the 2026-10-04 rank discovery falsified — I proposed that D14 had closed
a live axis on a dead instrument, and that its two arms (`add-widedense-arm`, `add-finespacing-arm`) might
have moved the rank unmeasured.

**The rank log does cover that period, and it does not support the hypothesis.** For
`add-finespacing-arm` (submitted 2026-09-20 09:46Z, scored 09-21 01:40Z):

```
2026-09-20T09:37:28Z   rank 391     <- before
2026-09-20T10:01:06Z   rank 395
2026-09-20T10:45:17Z   rank 395
2026-09-21T01:39:54Z   rank 393     <- after it scored (and the INBOX independently records 393)
2026-09-21T11:22:51Z   rank 432     <- +39 in 10 h, with no submission of ours
```

**The signal is 2 ranks; the noise is 39.** The field jumped 39 places in ten hours on its own, so a
±2 movement cannot be attributed to the arm. And §1 explains why: the arm scored 0.941 while our best was
already 0.941 — **so it was structurally incapable of moving the rank unless its true score exceeded our
best's true score**, which it did not.

⚠️ **What is missing is the in-block percentile, not the rank.** The block bounds for 2026-09-20 were
never recorded, and without them the absolute rank is unusable across a field that re-ranks in jumps.
That is precisely the instrument the 2026-10-04 discovery introduced — and it did not exist yet.

⇒ **D14's stated reason is superseded; D14's conclusion is not overturned.** Both of its arms remain
un-re-adjudicable, because you cannot retroactively recover a snapshot that was never taken. The axis
stays closed, and the specialist prereg's entry gate is unaffected.

## 4. The rule that follows

**A rank reading is only interpretable when it is paired with the block bounds at the same instant.**
`work/lb_snapshot.py` now persists `n_teams / rank / block lo..hi / in-block percentile` together
(`state/lb_snapshot_<UTC>.json`). Any arm submitted from now on gets a before and an after snapshot for
free — the discipline the 2026-10-04 finding demanded and that the September arms could not have followed.

⚠️ **And a second rule, specific to this instrument:** before reading a null result as "the arm did
nothing", confirm the arm was *capable* of moving the rank — i.e. that our previous best was not already
at or above it. Otherwise the measurement was never sensitive to the thing being tested.
