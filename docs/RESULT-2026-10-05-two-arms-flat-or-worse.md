# RESULT 2026-10-05 — both new arms landed at or below the baseline, and the preregistered gate fired

**Two arms were submitted and both completed. Neither beat `0.943`.**

| ref | arm | what it changes | public | vs baseline |
|---|---|---|---|---|
| `56814081` | `0943cfmsft2` | `C` + `S` | **0.943** | baseline — **still our best** |
| `56826372` | `0943cfmsft2m` | `C` + `S` + `M` | 0.943 | flat |
| `56599932` | `0943cf` | `C` | 0.943 | flat |
| **`56848247`** | **`0943cfmsft2flat`** | `C` + `S` + `R` | **0.942** | ⚠️ **below** |
| **`56845973`** | **`0943cfmsft2m-men`** | `C` + `S` + `M` + meniscus specialist | **0.940** | ⚠️ **below, and the worst of the round** |

`C` = cn0 member + C81 fracture fix · `S` = msft2 6th Raptor view · `M` = MCL out of `_RAD_EXCLUDE` ·
`R` = outer per-label routing flattened.

---

## 1. The meniscus specialist is a REGRESSION, and the prereg said in advance what that means

`0.940` against a `0.943` baseline. `docs/PREREG-2026-10-05-meniscus-arm.md` fixed the reading rule
before the run:

> 公榜 **≤ 0.940** ⇒ **真回归，丢掉**
> 格内百分位 **持平或更差** ⇒ 这一条也无效 —— 那么"专科"和"堆成员"一样到头，记录为负结果

**⇒ The label-specialist axis is falsified.** And the prereg chained it explicitly: with the
public-artifact, new-member and stacking axes already closed, a flat-or-worse meniscus means **this
round has no axis left.**

**⇒ `msft3-spec` (the self-trained hard-label specialist) is NOT FIRED.** Its own prereg §0 made the
meniscus result a hard entry gate, and the gate did not open. The builder stays on disk as a
documented, un-run artifact. ⚠️ **This is the prereg working, not a wasted build** — the alternative
was spending 2.6 GPU-hours on a hypothesis already measured dead.

⚠️ **Honest limit on this reading**, since it is what kills the axis: `0.940` vs `0.943` is `0.003`
against a public-LB SE of ≈ `0.008`. On the *score* alone this is not decisive. What makes it
actionable is (a) this competition's own selection sheet has always treated **≤ 0.940 as the one
thing the board CAN catch**, and (b) the direction is *down*, on an arm whose whole justification was
a +0.10-weight blend of a specialist built on a `0.936` parent. There is no reading of it that
supports spending more.

## 2. The routing hedge is measured unhelpful, which retires the 2026-10-04 argument

`0943cfmsft2flat` = `0.942`, below its own baseline `cfmsft2` = `0.943`. The arm was **verified to
have applied exactly the intended change** (one-line source diff; the run's own log showing
`preset=parent` and all twelve findings at `0.650` instead of the non-uniform map; a different
submission SHA), so this is a measurement of the *routing-flattening* effect, not of a failed edit.

⇒ **`R ≲ 0`.** The "route hedge" that `docs/FINAL-SELECTION-2026-10-04.md` §3 built its case around —
*"the frontier's own written remedy for a map tuned on a public split"* — **does not pay on our
stack.** `0943cflat` (`C + R`, 0.941) is therefore **no longer the right second slot.**

⚠️ Same caveat as §1: `0.001` is far inside the noise. But a hedge is only worth a slot if it is
believed positive **in expectation**, and the one measurement we now have points the other way.
**An argument that was reasonable without data stops being reasonable once the data arrives.**

## 3. Revised final pair — and it is a hedge on `S`, the only axis still uncertain

With `M ≲ 0` (measured: cfmsft2m moved the in-block percentile 13.67 % → 14.31 %, i.e. slightly worse)
and `R ≲ 0` (measured: §2), the three candidate pairs are:

```
(cfmsft2, cfmsft2m) = C + S + max(0, M)      hedges M only, and M measured <= 0
(cfmsft2, cflat)    = C + max(S, R)          hedges R, and R measured <= 0
(cfmsft2, cf)       = C + max(S, 0)          hedges S   <-- RECOMMENDED
```

**The third strictly dominates the first two whenever `M ≤ 0` and `R ≤ 0`**, because it is the only
one that protects against `S < 0` — and `S` is the one effect in this whole repository whose evidence
is a **public-rank gain** (in-block 38.89 % → 13.66 %) rather than a displayed score, so its sign on
the *private* set is genuinely unknown. If `S > 0` privately, the pair returns `C + S`; if `S < 0`, it
returns `C`. Neither of the other pairs can say that.

| slot | ref | arm | public |
|---|---|---|---|
| **primary** | **`56814081`** | `0943cfmsft2` — `C + S` | 0.943 |
| ~~**second**~~ | ~~**`56599932`**~~ | ~~`0943cf` — `C`, and deliberately **without** `S`~~ | ~~0.943~~ |

> ## ⛔ THE SECOND SLOT ABOVE IS SUPERSEDED — it is `56613303` (`0943cflat`)
>
> **The primary is unchanged.** The second slot was reverted on 2026-10-06 by
> `docs/REVISION-2026-10-06c-second-slot-reverts-to-cflat.md`, and the reason is worth stating here because
> it is a defect in the reasoning *below*, not merely a change of mind:
>
> **The argument below compares the two candidates on PUBLIC numbers, and on public numbers they are
> EXACTLY EQUAL** (`S > 0`, `R ≲ 0` ⇒ both pairs return `C + S`). A comparison that cannot separate its
> options cannot choose between them — so the choice has to be made on something else, and the competition's
> own **Dataset Distribution Notice** supplies it: *"prevalence … is not guaranteed to be the same across the
> training, public leaderboard, and final evaluation datasets."* The frontier's per-label routing map was
> fitted on the **public** split, so the second slot should be the arm that **drops** that dependence —
> `cflat`.
>
> Case check: `(cfmsft2, cflat) = max(P(C+S), P(C+R))` beats `(cfmsft2, cf) = max(P(C+S), P(C))` in every
> case **except** `S < 0` **and** `R < 0` **and** `R > S`. **On our measurements the two are identical, so the
> switch costs nothing.** `0943p` remains dominated by `cflat` either way.
>
> **Authoritative current state: the `CURRENT STATE` block at the top of `STATUS.md`.**

Superficially this looks like "dropping a positive change to buy diversification" — the exact thing
`FINAL-SELECTION-2026-10-04.md` §3 argued against. **It is not.** That argument was about deleting `C`,
the best-measured claim we hold; **`C` is in both arms here.** What the second slot gives up is `S`,
whose private sign is exactly what is in doubt, and giving it up *is* the hedge.

## 4. State of the board at the time of writing

Snapshot `2026-10-05T16:01:59Z` (`state/lb_snapshot_2026-10-05T160159Z.json`):

| | |
|---|---|
| teams | **5226** |
| our rank / score | **551** / 0.943 |
| bronze line (top 10 %) | **522** |
| **margin** | **−29** |
| `0.943` block | 414..1419 (1006) |
| in-block percentile | **13.63 %** |

⚠️ **The margin has collapsed across the day: +1 (07:59) → −4 (09:14) → −29 (16:02).** And note the
instruments again disagree in direction — the in-block percentile *improved* from 13.92 % to 13.63 %
while the margin fell 25 places, because the bronze line is absolute. **For a medal, read the margin.**

## 5. Where this leaves the campaign

Every axis this repository has tried is now measured, and the ledger is unambiguous: **fourteen arms
across five axes, one measured gain (`S`), everything else at or below baseline.** The structural
boundary `docs/STRATEGY-2026-10-04-how-far-can-finetuning-go.md` §4 named — *"to reach 0.944 needs
1–2 successful new axes; to reach 0.95 needs something structurally different (more data / external
data), not more fine-tuning"* — has survived every attempt to cross it cheaply.

**Three submissions remain before the 2026-10-22 deadline** (5/day cap; 2 used today). Any further arm
must clear a higher bar than any arm in this round did: **it must target a new axis, not a variation
of one of the five that are now closed.**
