# FINAL SELECTION — current sheet, 2026-10-04

> ## ⛔ SUPERSEDED 2026-10-05 — do not tick from this sheet
>
> **Use `docs/FINAL-SELECTION-2026-10-05.md` instead.** This sheet was written before the readout of
> `56814081` (`0943cfmsft2`) and **never mentions `msft2` anywhere** (verified by grep).
>
> **Its PRIMARY is wrong. Its SECOND SLOT is right.** The amended recommendation below
> (`56791368` primary + `56613303` second) becomes
> **`56814081` (`0943cfmsft2`) primary + `56613303` (`0943cflat`) second** — because `0943cfmsft2`
> carries the **only change in this repository with a measured gain on the rank instrument**
> (rank 708 → 487, in-block 38.89 % → 13.66 %), while `0943cfm`'s only extra content (`MCL`) then
> **measured flat** (in-block 13.67 % → 14.31 % on `56826372`).
>
> **Everything below about `0943cflat` — including §3's case check and the finding that it is a strict
> superset of `0943p` — still holds and is not retracted.** Only the primary changes.
>
> The text is kept verbatim so the reversal stays auditable.

**This supersedes nothing; it is a *current* reading of `docs/FINAL-SELECTION.md` (2026-09-28), which
remains the reasoning of record.** That document was written before the last readouts landed and then
read against its own pre-registered table. This sheet only updates the candidate list, because three
more arms exist now.

Deadline **2026-10-22 23:59 UTC**. Two submissions may be selected.

---

## ⛔ The selection is a BROWSER action — verified three ways, do not look for a CLI route
* `kaggle` CLI (2.1.2 and the 2.2.2 wheel both inspected) has **no `select` command**. Its
  `competitions` verbs are list/files/download/submit/submissions/leaderboard/episodes/replay/logs/
  pages/topics.
* `ApiSubmission` exposes **no `selected` field** — only `ref`, `public_score`, `private_score`,
  `status`, `description`, `file_name`, `date`, … **so the current tick state cannot even be read
  programmatically.**
* The legacy `www.kaggle.com/api/v1/...` endpoints return **401** to API-key Basic auth.
* ⚠️ **The auto-ticked default is NOT the recommended pair** (Kaggle auto-selects `0943` + `0943f`,
  A123). It must be changed by hand.

## The arms that are actually in play

| ref | arm | public | what it changes | evidence for that change |
|---|---|---|---|---|
| **`56599932`** | `0943cf` | **0.943** | our `cn0` CoAtNet as a 5th Raptor view **plus** the C81 fracture fix | ⭐ **two independently measured changes**; `cn0` is the only member this repo has ever measured **above** the frontier's own DINOv2 seed members (0.8769 vs 0.8676–0.8697, A130) |
| `56791368` | `0943cfm` | **0.942** | `0943cf` **+ `MCL` added to `_RAD_EXCLUDE`** | +0.000904 macro on n=3802/210pos, **35/35** source×assembly sign-stable, direction pre-registered by the 58-study audit (which named MCL worst of 12 and was ignored) |
| ⭐ `56613303` | **`0943cflat`** | 0.941 | `0943cf` **+ outer routing fully flattened** — **differs from `0943cf` on ONE axis** | same flattening as `0943p`, **plus** the two measured changes `0943p` does not carry (A161). ⚠️ **Was missing from this table; see the amendment at the end.** |
| `56570144` | `0943f` | 0.943 | `_RAD_EXCLUDE` drops `Fracture` | C81: +0.0118 on that column, +0.00099 macro, n=4185 |
| `56570151` | `0943s` | 0.943 | Raptor windows 94 → 62/42 | measured neutral; **an efficiency lever, not a score lever** |
| `56599159` | `0943p` | 0.941 | outer per-label routing **fully flattened** | the frontier's *own* written remedy for a map tuned on "a public split that is a fraction of the test set" — the axis it calls "the likeliest place to give back points privately" |
| `56568524` | `0943h` | 0.942 | same axis, softened halfway | same direction, weaker dose |

## Recommended pair

> ⚠️ **SUPERSEDED 2026-10-03 by the AMENDMENT at the end of this file.** The readout it was waiting
> for (`56791368`) landed, and the amendment changes the **second slot** from `0943p` to
> **`0943cflat` = `56613303`** — an arm that was missing from the table above. Read the amendment
> before acting; the text immediately below is kept only so the reversal is auditable.

**Primary `0943cf` = `56599932`.** Unambiguous: joint-highest, and the only arm carrying two locally
measured changes.

**Second slot — one of two, and this is the only real judgement call:**

* **`0943p` = `56599159`** (0.941) — the pre-registered recommendation. It sits on the **opposite side
  of the one knob whose provenance is documented as public-fitted**, so the pair *hedges* that axis
  instead of doubling it.
* **`0943h` = `56568524`** (0.942) — the **pre-registered alternative**, same hedge direction one notch
  weaker. `FINAL-SELECTION.md` itself notes this is "now the more defensible alternative" because
  `0943p` returned the lowest score of the nine.

⚠️ **The 0.002 between them is NOT evidence** — see below.

**If `56791368` (MCL) lands ≥ 0.942**, it becomes a legitimate third candidate for the primary slot on
the same footing as `0943cf` (two measured changes each). If it lands **≤ 0.940**, treat that as a real
regression signal and drop it — the board can catch that, even though it cannot rank 0.941–0.945.

**If `0943cfm2eff` lands with a stage file that DIFFERS from `0943cfm`'s**, the EffB0 leg is live and it
is the strongest candidate measured so far (+0.00403 out-of-fold over four independent folds,
CI [+0.00285, +0.00491], 4/4 positive). ⚠️ **Check this by diffing `_pipeline_stage.csv` against
`56791368`'s — if they are byte-identical the leg did not apply and the kernel is a DUPLICATE that must
not be selected.** That is exactly how the v1 run was caught.

## ⚠️ Why the scores are not the basis for this — the board proved it cannot rank these arms

Nine arms spanning four axes came back across a total spread of **0.941 → 0.943 = 0.002**, against a
public-LB **SE ≈ 0.008**. The 2026-10-03 full-leaderboard pull (`ryanholbrook/rsna-knee-abnormalities-efficiency-lb`,
4745 teams) shows **558 teams sitting at exactly 0.943** and **none of the 240 at ≥0.944 publishing
code**. So the selection is made on **what each arm changes and what evidence that change has**, never
on the displayed score. The one thing the board *can* do is catch a regression: anything **≤ 0.940**
should be dropped even though the nominal gap is inside the noise.

## Steps — with the ref→description mapping VERIFIED against the live API

⚠️ **Do not pick by score. FOUR arms show 0.943 and three of them are not the recommendation.** The
description text is the only discriminator, and it is what the Submissions page displays.

Every row below was read back from `api.competition_submissions(...)` on 2026-10-04 — the refs and
descriptions are the live values, not transcribed from a note. (`ref` is an **int** on the API object,
which is why a naive string-keyed lookup silently returned nothing.)

| tick? | ref | public | description shown on the page |
|---|---|---|---|
| ✅ **TICK (primary)** | `56599932` | 0.943 | `0943cf v1 - our cn0 member (5th view) PLUS the fracture fix: the two changes with local meas…` |
| ✅ **TICK (second slot, pick ONE)** | `56599159` | 0.941 | `0943p parent routing v1 - flat outer 0.65 (the frontier's own insurance, A122)` |
| ✅ **…or this one instead** | `56568524` | 0.942 | `0943h halfway routing v2 - internet off, 20/20 fingerprints` |
| ❌ do NOT tick | `56568520` | 0.943 | `0943 speedy k94 v2 - internet off, 20/20 DINOv2 fingerprints, outer 0.65` ← the auto-ticked default |
| ❌ do NOT tick | `56570144` | 0.943 | `0943f fracture fork v1 - internet off, 20/20 fingerprints` |
| ❌ do NOT tick | `56570151` | 0.943 | `0943s sparse 62/42 v1 - internet off, 20/20 fingerprints` — **efficiency lever, see below** |
| ✅ **0.942** | `56791368` | 0.942 | `0943cfm - 0943cf + MCL added to _RAD_EXCLUDE; +0.000904 macro on n=3802/210pos, 35/35 sign-s…` |
| ✅ **TICK (amended second slot)** | `56613303` | 0.941 | `0943cflat v1 - cn0 member + C81 fracture fix + FLATTENED outer routing: one line from 0943cf, so …` |

1. Open the competition's **Submissions** page.
2. **Untick the default** — its description begins `0943 speedy k94 v2 …` (ref `56568520`).
3. Tick the row whose description begins **`0943cf v1 - our cn0 member (5th view) PLUS the fracture fix`**.
4. Tick **one** of: description begins **`0943p parent routing v1 - flat outer 0.65`** (0.941, the
   pre-registered hedge) **or** **`0943h halfway routing v2`** (0.942, same direction, weaker dose).
5. If `56791368` (`0943cfm … MCL …`) has scored **≥ 0.942** by then, it is a legitimate third candidate
   for the primary slot — it carries two measured changes like `0943cf` does, plus MCL. If it scored
   **≤ 0.940**, drop it (the board can catch a regression even though it cannot rank 0.941–0.945).
6. **Re-check before the deadline**: the tick state is not readable through the API (no `selected`
   field on `ApiSubmission`; the CLI has no `select` verb; legacy `/api/v1` returns 401 to API-key
   auth — all three verified), so this page is the only place that state exists.

⚠️ **Optional, costs nothing, and only you can weigh it**: `56570151` (`0943s sparse 62/42`) also scored
**0.943**, i.e. the same AUC as the primary with a shorter runtime, and this competition scores
`Efficiency = AUC/(0.5 − maxAUC) + RuntimeSeconds/32400` with an independent **$18,000 / 3-place
efficiency track** that the *same* submission feeds (`docs/EFFICIENCY-PLAN.md`). You have said the
efficiency board is not a goal, so this is recorded and not recommended — but ticking `0943s` in place
of a 0.943 arm would be free on the accuracy axis.

---

# ⭐ AMENDMENT 2026-10-03 — the pending rule FIRED, and it changes the second slot

## 1. The readout

`56791368` (`0943cfm`) came back **COMPLETE, public 0.942** (ledger row now reads 0.942; it had been
`PENDING`). By §"Steps" step 5 that is **≥ 0.942 ⇒ a legitimate candidate for the primary slot**.

## 2. ⚠️ But the bigger finding is an arm this sheet had DROPPED from its candidate table

`56613303` = **`0943cflat`**, scored **0.941** on 2026-09-27, and it is **absent from the six-arm table
above and from the nine-arm table in `FINAL-SELECTION.md`**. It should not be: it was built
*expressly* to be this pair's second member (`scripts/build_0943_addmember.py` L126–129, FACTS **A161**).

```
0943cf     = base + cn0 member + C81 fracture fix,            published per-label routing
0943cflat  = base + cn0 member + C81 fracture fix,            FLATTENED outer routing   <- 56613303
0943p      = base +                                          FLATTENED outer routing   <- 56599159
0943cfm    = base + cn0 member + C81 fracture fix + MCL,      published per-label routing <- 56791368
```

**A161 already recorded the conclusion in words** — *"它对 `0943p` 还是**弱占优**（同样的拉平，外加
+0.0028 成员与 +0.00099 骨折修复）"* — and then declined to act on it, on the grounds that `0943cflat`
is *"集中下注"* (a concentrated bet). **That objection does not survive being written out.**

## 3. Why "concentrated bet" is the wrong way round

Let the private-set effects of the three measured changes be `C` (cn0 + fracture fix), `M` (MCL) and
`R` (routing flattening), measured against the 0.943 baseline, and let the final score be the **better
of the two selected arms** (which is the only reading under which hedging makes sense at all).

| pair | better-of-two |
|---|---|
| **A** = (`0943cfm`, `0943cflat`) | `base + C + max(M, R)` |
| **B** = (`0943cf`, `0943p`) ← pre-registered | `base + max(C, R)` |

Case check on `A − B`, for every ordering of the three:

| case | `A − B` |
|---|---|
| `R ≥ C + M` | `C` ≥ 0 |
| `C ≤ R < C + M` | `C + M − R` > 0 |
| `R < C`, `R ≥ M` | `R` ≥ 0 |
| `R < M` | `M` ≥ 0 |

⇒ **`A ≥ B` pointwise whenever `C ≥ 0` and `M ≥ 0`.** Pair B wins only under `C < 0` — i.e. only if the
cn0 member and the fracture fix are *both* harmful privately, which is the single best-measured claim
this repository holds (C81: +0.0118 on that column, +0.00099 macro, n=4185, five label sources, gold-58
the sole dissenter).

**The intuition is the part worth keeping: including a positive-evidence change in *both* arms is not
concentration — it is refusing to give away expected value.** Diversification bought by *deleting* a
measured improvement is not diversification. `R` can be as bad as it likes and B still cannot win,
because in B the routing hedge **replaces** `C` instead of **stacking on** it.

## 4. Amended recommendation

| slot | ref | arm | public |
|---|---|---|---|
| **primary** | **`56791368`** | `0943cfm` — cn0 + fracture fix + MCL | 0.942 |
| **second** | **`56613303`** | `0943cflat` — cn0 + fracture fix + flat routing | 0.941 |

⚠️ **The primary moved off `56599932` (`0943cf`, 0.943) — and this is the weakest step in the
amendment.** `0943cfm` is a strict superset of `0943cf`'s changes, so it dominates on the sheet's own
criterion ("what each arm changes and what evidence that change has"). But MCL is the **thinnest** of
the three evidences — its own write-up says so: CI only just excludes zero, **3/5 folds**, and the
instrument's parent is stage-1 (public20) rather than the deployed stage-2, *and MCL is exactly the
marginal case that a stronger parent could flip*. The board's 0.942 vs 0.943 is **not** evidence
either way (SE ≈ 0.008, and §"Why the scores are not the basis" forbids ranking on it).
**So: if you would rather not take the MCL bet, tick `56599932` as primary and keep `56613303` second —
that pair is `A` with `M = 0`, and by the same case check it still dominates B.** The one arm that must
change is the **second slot**.

### Browser steps (refs verified against the live API)

1. Untick the auto-ticked default — description begins **`0943 speedy k94 v2`** (ref `56568520`).
2. Tick the row whose description begins **`0943cfm - 0943cf + MCL added to _RAD_EXCLUDE`** (ref `56791368`).
   *(Or, if skipping the MCL bet: **`0943cf v1 - our cn0 member (5th view) PLUS the fracture fix`**, ref `56599932`.)*
3. Tick the row whose description begins **`0943cflat v1 - cn0 member + C81 fracture fix + FLATTENED outer routing`** (ref `56613303`).
4. Do **not** tick `56570144` (`0943f`), `56570151` (`0943s`) or `56568520` (`0943 speedy`), all 0.943.

## 5. What this amendment does NOT claim

* It does **not** claim `0943cflat` will score better than `0943p` on the board — both are **0.941**, and
  0.001 against ±0.008 is nothing. The claim is about **which bets the pair holds**, not about scores.
* It does **not** upgrade the underlying evidence. Every effect here is ≤ 0.003 on a board whose noise
  is ±0.008; this whole decision is made **inside** the resolution floor, exactly as §"The rule" says.
* It does **not** assume `C`, `M` are certainly positive — only that they are positive *in expectation*,
  which is what their measurements say and what pair B implicitly denies.

⚠️ **Still a browser-only action.** Re-verified 2026-10-03 on kaggle 2.1.2 **and** the 2.2.2 wheel: no
`select` verb exists in either (`2.2.2` exposes `team-submissions` but **not** `submission-limits`
either — so A135/A162's `submission-limits` reading is **no longer obtainable from the CLI at all**,
and quota must be judged another way). `ApiSubmission` still exposes no `selected` field.
