# STATUS — 2026-09-27 22:30Z

**One screen. Rewritten every session.** If this grows past a screen, the overflow belongs in a
`HANDOFF.md` section, not here.

## 🚨 The premise changed today — read this before anything below

**We are NOT at the public ceiling.** The live leaderboard reads **0.959**, we are at **0.943**, and the
gap is **+0.016 — exactly 2× the ±0.008 noise band**. Our row: **rank 411, score 0.943, 36 submissions**.

The old belief ("ceiling ≈ 0.941–0.943, so we hold it") came from **A87.1, which read other
competitors' *claims* off a discussion thread — not the leaderboard.** The leaderboard is free,
authenticated and one command away, and it had never been read. ⇒ **A quantity that is cheaply and
authoritatively measurable must never be carried as a remembered belief.** ⚠️ I then violated that same
rule once more, in the same hour: `lb_ceiling.py` **derived** our rank as "teams strictly above + 1" and
reported **219**, when the file it had already downloaded contained a `Rank` column saying **411**. Every
team gets a **distinct rank inside a block of equal scores**. Both bugs are now covered by `--selftest`.

**Re-read it with `python scripts/lb_ceiling.py`.** Full plan:
[`docs/REVIEW-2026-09-27.md`](docs/REVIEW-2026-09-27.md).

| | |
|---|---|
| ⚠️ Is the gap real? | **Mixed.** Teams above us have median **83** submissions vs **15** in our band, so part is public overfitting. ⚠️ **But `daimaru` reached 0.955 on 8 submissions** — that cannot be hill-climbing ⇒ **a real recipe exists.** |
| ⚠️ What it is NOT | **Not a missing checkpoint.** `kernel-0943` already mounts essentially the entire published model zoo (maxspan, native384, native384dense, coat-resgated, d4-depthzone, global96, e11, e9, llm-labels, v52, radimagenet, pilkwang + DINOv2-small). |
| ⭐⭐ **The actionable target is +0.001, not +0.016** | ⚠️ **Live 2026-09-28**: our rank is **430**, and 0.943 is a **481-team block spanning ranks 230–710**. **Silver (top 5% = rank ≤224) sits in the 0.944 block (50 teams, ranks 180–229); bronze (top 10% = rank ≤448) falls INSIDE our block — 18 ranks below us.** ⇒ at 0.943 **whether we medal at all is decided by an opaque tie-break, not by score**, while **+0.001 (→0.944) escapes the block into silver.** ⚠️ Cutoffs are the standard Kaggle rule and medals are awarded on the **private** LB, so this is indicative, not decisive. |
| ⭐ The unlock | Training was closed on **"our arm cannot beat A5"** (A106, true). That is **not** the question: an ensemble member needs to be **strong AND decorrelated**, and `cn0` (0.8769, *below* A5) proved it adds **+0.0028, CI [+0.0011, +0.0043]** on a parent-quality base. **Two different questions were conflated.** |
| ⇒ The plan | **Phase 1 (no quota):** ✅ member sweep **done** — our existing stock is exhausted (only `cn0` adds). Still open: per-finding specialists; weekly frontier watch. **Phase 2:** train members in families the stack does *not* have (TPU, 1.28 h/fold) — **now the only path**. |
| ⚠️ Not relaxed | **A96 stands** — 0.002 is still invisible on a *single* arm. But a **real +0.003** on the private split is exactly the size that decides this competition, and the +0.001 escape from the tie block is the concrete objective. |

## 🚨🚨 And the headline of the 2026-09-28 全盘调查: we have been playing ONE track

**There is a `$18,000` Efficiency Track (3 places) that rewards accuracy-per-second, and the same
submission counts for BOTH tracks.** We are on its leaderboard at **efficiency rank 3343 / 4363**
(vs main-LB rank 430 / 4479).

| | |
|---|---|
| How it was missed | **The repo recorded it on 2026-09-13** (§72.6: *"$18,000 Efficiency Track… the same submission can win both tracks"*) and **flagged it unverified — then nobody mentioned it again for two weeks.** The whole project optimised the main track only. |
| Independent confirmation | A public repo tracking this competition: *"$77,000 total, including a separate Efficiency Track ($18,000 over 3 places)… entry/merger deadline **2026-10-15**"*, and its own locked decision was **"target both the main and efficiency tracks."** |
| ⭐ What it is | **Almost independent of score.** eff#1 = 0.958, eff#3 = 0.948, **eff#14 = 0.943 — the same score we hold** — while some 0.959 teams sit at eff#266/#655. ⇒ **a runtime contest, not an accuracy contest.** |
| ⇒ Where the headroom is | main LB: rank **430**, worth **+0.001** (silver). **Efficiency: rank 3343, worth thousands of places — and it is free on the main track.** |
| ⚠️ Not yet verified | the exact metric (§72.6's formula is second-hand and dimensionally odd), what runtime Kaggle measures, what the 2026-10-15 entry/merger deadline implies, and **our own runtime — never measured.** |
| ⇒ Next | **measure our runtime**, then verify the formula. If runtime dominates, *"3 h → 20 min"* is worth far more than *"0.943 → 0.944"*. The material already exists: `0943s` (62/42 sparse windows, **main-LB neutral at 0.943**) and the whole "speedy raptors" lineage. |
| ⚠️ Trap recorded | I first searched the efficiency board for **`ppxl16`** and found nothing — the board lists us as **`p pxl16`** (with a space). **A wrong search string nearly produced "we are not on the board".** |

## ✅ Phase 1, first result: our existing checkpoints are exhausted

**P1.1 member sweep is done** — all **24** arms through the honest gate (`member_sweep.py`, parent-quality
base, every weight fitted on one half and scored on the other). Full detail: HANDOFF §215 / A158.

| arm | Δ | 95% CI | verdict |
|---|---|---|---|
| `cn0@f0` | **+0.0028** | [+0.0011, +0.0041] | ⭐ **PASS** — already shipped in `0943cf` |
| `d64cn0@f0` | **+0.0016** | [+0.0006, +0.0027] | passes alone ⚠️ but see below |
| `nx3361` · `nx3360` · `d3f0` | +0.0009 · +0.0005 · +0.0003 | all straddle 0 | no signal |
| 17 others | 0.0000 (weight 0 chosen in **both** halves) | [0, 0] | no signal |
| `r3360` · `r3362` | **−0.0004** | both exclude 0 | ⚠️ **hurts** |

⭐ **Only the CoAtNet family passes; none of ~15 DINOv2/other arms do, and two actively hurt** — consistent
with the mechanism (our DINOv2 arms duplicate the frontier's own seeds; CoAtNet is the decorrelated one).
The positive control reproduced `cn0`'s +0.0028 exactly, so the sweep is trustworthy.

⚠️⚠️ **And the single-member gate is not sufficient.** `d64cn0` passes alone, but it is the *same family*
as `cn0`, so I added a **joint** test (`--arm A --arm B`, one grid dimension per member; the single-arm path
verified bit-for-bit unchanged):

| combination | Δ | 95% CI |
|---|---|---|
| base + `cn0` | **+0.0028** | [+0.0011, +0.0041] |
| base + `cn0` + `d64cn0` | **+0.0026** | [+0.0011, +0.0039] |

⇒ **Adding `d64cn0` makes it no better, and the joint fit gives it weight 0 on one half. They are
substitutes, not complements.** ⭐ Method lesson: *a single-member gate will pass two members that are
substitutes; for same-family members the only valid question is the joint one.*

⇒ **Consequence for the plan:** the member line is **exhausted with what we hold**. Climbing the 0.016
requires **training new, decorrelated members** (families the stack lacks) — that is Phase 2, and it is now
the only path. ✅ **Today's submission slot is therefore not used**: the one new candidate is redundant with
the shipped `cn0`, and nothing else met the standing rule *"confirm it can raise the score before
submitting"*. Slots reset per UTC day and cannot be banked, but adding an arm we would never select is a net
negative at the deadline.

## Where we stand

| | |
|---|---|
| Deadline | **2026-10-22 23:59 UTC** (~3.5 weeks) |
| Best ledger score | ⭐ **0.943**, a **four-way tie**: `56568520` (`0943`) · `56570144` (`0943f`) · `56570151` (`0943s`) · **`56599932` (`0943cf`)**. Then `0943h`/`0943o`/`0943c`/`0943r` at **0.942** and **`0943p` alone at `0.941`**. ✅ **All nine in; ledger 36 rows, 36 scored.** |
| Public LB noise | **±0.008** ⇒ the board cannot rank *these* arms (all nine span just **0.002**), but it **can** see the 0.016 to the top. ⭐ The `0943p` dose-ladder and why the board's preference there is non-discriminating: [`docs/FINAL-SELECTION.md`](docs/FINAL-SELECTION.md). |
| ⭐ What the readouts bought | **All nine arms completed the hidden rerun on 1322 studies** — including **our `cn0` member twice** (`0943c` 0.942, `0943cf` 0.943), 0 failures, which **retires the largest unmeasurable risk on it** (§200). **That de-risking is independent of the score.** |
| GPU quota | **17.49 h / 30 h** used (**12.51 h left**), refreshes 2026-10-03. TPU **1.88 h / 20 h** (18.12 h left). ⚠️ The counter moves in **20-minute quanta** (A121). ⇒ to the deadline (3 more refreshes) ≈ **100 GPU-h + 78 TPU-h (≈60 folds) + ~120 submission slots**: **compute-rich, idea-poor.** |
| Cost model | **1 h wall clock = 1 GPU hour even on the 2×T4 shape.** Scoring reruns are **not** billed — re-confirmed over a 7-hour idle window (A121). |

## 🔴 The one action left — and it is not something the CLI can do

The recommended pair is **`56599932` (`0943cf`, 0.943) + `56599159` (`0943p`, 0.941)**.
**Kaggle's auto-tick is not it** — it took `56568520` (`0943` v2) + `56570144` (`0943f`) (A123).

⇒ **Untick `RSNA knee 0943 - Version 2`, and tick `RSNA knee 0943cf` + `RSNA knee 0943p`.**
Full reasoning, and what leaving the default costs:
[`docs/FINAL-SELECTION.md`](docs/FINAL-SELECTION.md).

⚠️ `kaggle` 2.2.4 has **no select command** (verified) ⇒ this needs the browser. And the default is
not predictable: four of our arms tie at 0.943 and auto-select took two of them, **not** the ones we
would choose (A123). Deadline **2026-10-22 23:59Z**.

⚠️ **This is the one decision where your judgment legitimately overrides the rule.** The pair is
deliberate — opposite sides of the one knob documented as public-fitted (A122) — but `0943p` is the
**single lowest of the nine (0.941)**, and the pre-registered alternative **`0943cf` + `0943h`** holds
the **same hedge direction at a weaker dose**, scoring 0.942. Neither choice is supported or refuted by
the board (the whole nine-arm spread is 0.002 against ±0.008 noise).

⭐ **The routing hedge is no longer an analogy.** The frontier notebook's own source comment says the
per-label `coatnet_w` overrides were set "on the strength of a public split that is a fraction of the
test set… It may be real. It is also the likeliest place to give back points privately, which is what
the `parent` preset is for" (A122, §194.3). That covers the *direction*; the *dose* is still
unresolvable — see the caveats in the selection doc.

**The choice set is NINE arms, all nine read back, and the decision was pre-registered** — see
[`docs/FINAL-SELECTION.md`](docs/FINAL-SELECTION.md), written **before** the readouts landed so they get
read against a rule rather than fitted to a feeling.

| | |
|---|---|
| New since §196 | **`0943c`** = `0943` + **our rescued `cn0` as a fifth Raptor view** (`56599163`, 0.942) and **`0943cf`** = the same **plus** the fracture fix (`56599932`, **0.943**) — the two changes this repo actually has measurements for |
| Recommendation | **`0943cf` + `0943p`** — the published per-label routing against the flattened one, i.e. it hedges the single knob whose provenance is documented as public-fitted, and covers three of the four axes between them |
| ⚠️ The board cannot rank them, and that is now **measured** | all nine in, four different axes, and the **entire spread is 0.002** against ±0.008 — the readouts can only **disqualify** (a ≤0.940 return, or a failed run), and that rule is written down in the doc |
| ✅ Every arm is submitted | all nine went through a full hidden rerun on 1322 studies ⇒ **all nine are selectable** |

## Live right now

| Thing | State |
|---|---|
| Submissions | ✅ **all nine read back; nothing pending.** Ledger **36 rows, 36 scored**, best **0.943**. ⭐ **`56599932` (`0943cf`) = `0.943`, tied for best** — the recommended primary, carrying the only two changes with local measurements, and it **completed** the hidden rerun. ⭐ Our rescued `cn0` member has now survived the hidden rerun **twice** (`0943c` 0.942, `0943cf` 0.943), 1322 studies each, 0 failures. ⚠️ `0943p` is alone at `0.941`. |
| ⭐ The A-lite paired arms | **both QUEUED** on GPU: `a5fa` (the frontier's A5 architecture) and `rfa` (our head) — same script, fold, epochs, bs, res, and **the same augmentation on both**. This is the transfer check that gates the 48 GB cache (§210.3). |
| The augmented readout | **pre-registered before the numbers landed**: `scripts/a5lite_pair.py`, threshold **+0.02**. It reproduces A153 exactly (0.7941 / 0.7366, Δ +0.0575, P=1.00) and **asserts the self-pair invariant** (0.0000, degenerate CI) before printing any delta. |
| Density rung `a5lite-n8` | ⚠️ **built, deliberately NOT pushed** — both GPU slots are held by the queued pair. See the reopened-training section below. |
| `0943c` integration | **proven to run**: `raptor_raw.npz` came back **(5, 3, 12)** — five arms, not four — with 0 failed studies and 0 neutral fills, and our view correlates most strongly (0.792) with the one sharing its geometry. Cost measured: Raptor stage **28.9 s → 33.2 s (+15%)** after moving it to the lighter GPU (A143). |
| `0943p` / `0943r` runs | **both `COMPLETE`, both passed the acceptance test** (20/20 fingerprints, fail-closed gate PASS, valid `submission.csv`, 0 tracebacks). `0943p`'s log even shows `outer CoAtNet weight per finding: flat 0.65` — the flattening demonstrably took effect. |
| Kept for later | the pilot's log also revealed that the `[config]` line prints `coatnet_w` but **not** `rad_cal_w` — so for `0943r` the log cannot certify its own change; the proof is the artifact + the code path instead (A134) |

## Closed — do not reopen without new evidence

| Line | Why it is closed |
|---|---|
| **Training line — backbone swap** | Our best arm scores **0.8731** against A5's **0.9143** on the 181 studies where both are genuinely held out — a gap whose CI excludes zero (A106). Below the archive's own ~0.90 member bar. **And swapping in DINOv3 directly went backwards** (`d3f0` 0.8530, §203). ⚠️ The *architecture* half is separately live under A153 — the backbone is ruled out, the head is not. |
| **Champion folds** | The checkpoints carry a config, but in a **different schema**, and **neither gate can see a wrong geometry** (A108). Forcing it would produce a confident, wrong result. |
| **E2 parent OOF** | The 20 published members are **not held out** (A93): parent 0.9295 vs a genuine OOF's 0.8323 on the same studies. |
| **Fusion knobs on the frontier stack** | No valid parent OOF exists, so they could only be judged on the ±0.008 board. `rad_cal_w`→0 was "locally contradicted" by A117 — **that basis is retracted** (A127): the probe measured the calibrator repairing a mis-set blend, not adding information. |
| **Image+metadata arm** | Measured honest gain on top of our own ensemble: **−0.0033** (A112). Plan marked NOT RECOMMENDED. |
| Dense corpus · member additions · capacity scaling · full retrain · gold-58 for small deltas · CoAtNet on XLA · multi-TPU queues · rental box as a scoring route | Each closed by measurement, not assumption. |
| Scheduled tasks | Deleted (§164) — they were idle and were the black console window the user saw. |

## Where the remaining weeks go — 复盘 of 2026-09-27

**The 2026-09-26 review was superseded the next day, because its premise was false.**
See [`docs/REVIEW-2026-09-27.md`](docs/REVIEW-2026-09-27.md) (and keep the 09-26 one for its
*instrument* map, which is still correct). It said "the score is at the public ceiling, so what is
left is unmeasurable selection risk". **The ceiling is 0.959; we are 0.016 below it.**

| | |
|---|---|
| 1. ✅ What still holds | the instrument map, the five public-fitted knobs staying unjudgeable (A93), and the pre-registered hedge — **`0943cf` + `0943p`** |
| 2. ⭐ **Climb** | Phase 1 (no quota): member sweep + per-finding specialists + weekly frontier watch. Phase 2: train members in families the stack lacks |
| 3. ⚠️ **Not "reach 0.959"** | part of that gap is public overfitting with unknown private transfer, and we cannot see the leaders' methods. Target = **climb past 0.943 with non-public-validated changes** |
| 4. **Do not spend the weeks turning fusion knobs** | still true — closed four independent ways, and the parent has no honest OOF |

### ⭐ And one line that reopened: training (2026-09-26, §197)

The review's "don't turn knobs" applies to *fusion* knobs. **Training was closed on a measurement
that turned out to compare us to the wrong arm**, and that is now fixed:

| | |
|---|---|
| **A106 said** | our arm 0.8731 vs A5 0.9143 — "training line closed, −0.0412" |
| **`arm_parity.py` measured** | A5 0.9089 · **our `cn0` 0.8769** · the frontier's own four DINOv2 seed-members **0.8676–0.8697** · Rad 0.8424 |
| **⇒** | **we are +0.007 ABOVE the frontier's own DINOv2 members.** The 0.041 gap is *entirely* "A5 is a DINOv3 model" — not a property of our training. |
| **`member_add_test.py` measured** | adding our arm to a 0.917-quality blend: **+0.0019, CI [+0.0002, +0.0035]**, at rank correlation **0.8008** — below the **0.8061** A5 and Rad already share |
| **⇒** | `member additions` (§159.6) is not closed on this instrument either: our arm carries information the existing members do not. |

**And the payoff holds up on a parent-quality base (§199.2).** With the two Rad heads added to the base and
**every** blend weight (including the member's) chosen on one half and scored on the other: honest base
**0.9173 → 0.9201, +0.0028, CI [+0.0011, +0.0043]**. The gain did *not* shrink as the base strengthened —
and the mechanism is the interesting part: **both halves independently gave our arm weight 0.2 and took it
out of `public20`'s weight, leaving A5 untouched.** So the arm displaces the frontier's own DINOv2 seeds,
which is exactly what `arm_parity.py` predicted from the other direction.

⚠️ One fold, 870 studies, one arm; the base is still 0.917, not the parent's ~0.94. And **this does not yet
mean score** — converting it requires putting our member into the validated submission notebook.

**Scoping that step (§200) changed the picture: the mechanics are easy, the *quality* is the blocker.**
The submission notebook adopts **each member's own config** (`adopt_config_globals(cfg)`, per-member
`crop_mm`/`band`/`rules`), and our manifest is in that same schema with the same six slot names and the same
`RULES_NATIVE` — so a compatible member is close to drop-in, not a new inference path. But:

| | |
|---|---|
| `cn0` — the arm that produced **+0.0028** | ⭐ **weights RECOVERED and now DURABLE (§201–§202).** They were on the rented box after all — `/root/autodl-tmp/kaggle/working/`, which the first search missed because `-xdev` prunes that separate mount. Six checkpoints (1.68 GB) are local (git-ignored) **and published** as the private dataset **`ppxl16/rsna-knee-cn0-weights` (v2, ready)** — the only copy independent of both the box (now shut down) and this laptop. Identity confirmed by `best_gold_auc = 0.877929` against the recorded 0.8779; integrity by CRC-clean archives plus byte counts identical to the box's listing. ⚠️ Still not *shippable* into the submission until the notebook edit, which is a separate, risky step. |
| the 22 fine-tuned checkpoints on disk | all below the frontier's own seeds (best available `r3362` 0.8663 vs their 0.868–0.870) ⇒ adding them would be a negative — `cvg0` already demonstrated that |
| the published `rsna-knee-train-weights` | 5 × DINOv2-small@336 @ crop 130 mm, 20 epochs; their blend scored LB 0.842 on 09-06 |

### 🔴 The DINOv3 pilot landed and **the kill condition fired** (§203)

`kernel-tpu-d3f0` (DINOv3-small @336, k=12, 12 epochs, fold 0, 1.85 TPU-h) finished and converged
normally — then scored **0.8530** against a pre-registered stop line of 0.88:

| | macro-AUC (same labels, same studies) |
|---|---|
| A5 — the **frontier's** DINOv3 | **0.9089** |
| our `cn0` — CoAtNet@384 | **0.8769** |
| public20 — the **frontier's** DINOv2 seeds | 0.8697 |
| **our `d3f0` — DINOv3-small@336** | **0.8530** |

⇒ **Not merely below the line: below our own best arm by 0.024 and below the frontier's own seeds by 0.015.**
gold-58 agrees (0.8572 vs cn0's 0.8779), so two metrics say the same thing. **No folds 1–4, and `d3p0`
(small-plus) is not pushed either** — small did not match our DINOv2 arms, so capacity is not the story, and
reaching `cn0` would need +0.024 against a family slope of +0.01–0.02.

⭐ **The diagnosis is the deliverable: swapping in the backbone did not solve it.** The frontier's ~0.05 edge
on the DINOv3 family is **not the backbone** — it is their architecture (cross-series attention-MIL and
slots), their recipe, and probably much longer training. The *backbone-swap* route is closed by two
independent measurements (A106's 0.0412 gap to A5; and this direct swap that went backwards) — but see
**A153 below**, which reopened the *architecture* half of that sentence with a paired measurement.

⇒ **The only live scoring question left is whether to integrate `cn0`** — the one member that beats the
frontier's own seeds (0.8769 vs 0.8697), already published to Kaggle, with local support (+0.0019 / +0.0028 on
a 0.917 base) and a compatible member schema — but also with the unmeasurable shrink-at-0.94 risk and this
repo's history of changes that pass every gate and quietly degrade. **That is a user decision.** It has
now been **built and submitted** as `0943c` / `0943cf` (see the nine-arm choice set above).

✅ **The weights are no longer a risk.** Both suggested unblocks are done: the AutoDL box was powered on,
`cn0`'s six checkpoints (1.68 GB) were pulled from `/root/autodl-tmp/kaggle/working/`, verified by CRC and
by `best_gold_auc = 0.877929`, and **published** as the private dataset `ppxl16/rsna-knee-cn0-weights`
(6 × 279.3 MiB, ready). The box is shut down and the copy is now durable independent of it.

⚠️ **The diagnosis above needs one correction from A153.** "It is their architecture, their recipe, and
probably much longer training" was right, but A152/A153 split it further: the architecture **does** carry a
measurable advantage (+0.0575 paired) — the DINOv3 swap failed because it changed the *backbone* while
leaving the head, which is not where the edge is.

### ⭐⭐ A153: the paired arms came back, and they **reversed** the surface reading (§210)

| macro-AUC — same 881 studies, same script / fold / 12 epochs / bs=2 / **no augmentation** / same backbone | |
|---|---|
| `a5f0` — the **frontier's** A5 architecture | **0.7941** |
| `rf0` — **our** head (the control arm) | **0.7366** |
| **Δ** | **+0.0575, 95% CI [+0.0411, +0.0763], P(a5>r)=1.00** |

⭐ **The control arm is why the conclusion is right.** `a5f0` vs `d3f0` (the real trainer, same head as
`rf0`) is **−0.0584, CI [−0.0706, −0.0455]** (A152) and reads as "their architecture is *worse*". But
`rf0` and `d3f0` are the *same head*, and the **−0.116** between them is *my* simplified setup (no
augmentation, bs=2, fixed windows) — not architecture. ⇒ **Without the control arm I would have written
the opposite conclusion**, which is the empirical case for paired arms.

⚠️ **But +0.0575 was measured at a degraded operating point**, and this repo has twice measured such
gaps shrinking as the base strengthens (A109, A112). ⇒ **The augmented pair now in flight is the
transfer check**, read against a rule fixed in advance (§210.3): **≥ +0.02 ⇒ the advantage is real and
the 48 GB faithful cache is justified; collapse ⇒ route A stops**, and the ~0.05 lives in the 16-slice
density + the frontier's recipe rather than in their checkpoint.

⭐ **And "48 GB or nothing" was wrong on one point.** The corpus holds 44 slices in five groups of
**12/10/8/6/8**, and `a5_study` windows *inside* each group — so `n_slice` was capped at 6 by the
**smallest** group, not by the corpus. A rung at **`n_slice=8`** takes 8 **distinct** slices from the
12/10/8/8 groups and pads only coronal-no-fluid (6→8): **four of five groups fully distinct**. Measured
profile, so the rungs are not interchangeable (6 = zero padding; 10/12 are padding-dominated and would
measure *channel capacity*, not density — which is why only 8 is built):

| rung | distinct/total per group (12/10/8/6/8) | what it measures |
|---|---|---|
| `n_slice=6` | 6/6 · 6/6 · 6/6 · 6/6 · 6/6 | the honest ceiling of the corpus |
| **`n_slice=8`** | 8/8 · 8/8 · 8/8 · **6/8** · 8/8 | **a real density rung** |
| `n_slice=10` | 10/10 · 10/10 · 8/10 · 6/10 · 8/10 | half capacity |
| `n_slice=12` | 12/12 · 10/12 · 8/12 · 6/12 · 8/12 | mostly capacity |

⇒ It costs **one** arm, not two: the raptor control's input path never reads `n_slice`, so the queued
`rfa` is a valid control for **every** rung. `kernel-a5lite-n8` is built and verified (`--check` OK for
all three payloads) and **held back on purpose** — push it only if the augmented pair passes.

⚠️ **Kept honest:** the +0.0019 was measured on a **0.917** base and the parent is ~0.94, so by the
A109/A112 pattern the real gain is smaller, and that side is unmeasurable. This is a route worth
trying, not a +0.002 promise.

⭐ **The review's useful result is a negative one.** The instrument was first upgraded to the best we
have ever had — a Rad-free honest surrogate at **0.9151 on 4406 studies** (previous stand-in: 0.8823
on 870) — and on it the calibrator looked worth **+0.0075** on the gated labels with an optimal dose
of 0.7–0.8, **chosen identically in all five folds**. Then the rival explanation killed it: the
calibrator's inputs contain *both sides* of the blend it corrects, and the full `rad_alpha ×
rad_cal_w` surface peaks at **`rad_alpha≈0, cal_w=0`** with the calibrator's best dose rising
monotonically in `rad_alpha` — the signature of a correction term, not of information (A127). So
**`rad_cal_w=0.8` is not submitted**, and **`0943r` reverts to "a bet with a public source,
unverifiable locally"** (A128).

**The rule to carry: a probe asking "does X add information on top of Y?" must first show that Y is at
its own optimum — otherwise it measures repair.**

⚠️ **Every knob in the 0.943 recipe is gated on the same missing instrument.** `coatnet_w` was set on
the public split (A122 — the author's own words); the Rad family on 58 gold labels; `a5_w`, the view
weights and `coat_private_alpha` would each need the DINO base's honest OOF, and **the twenty
published DINOv2 members are not fold-trained, so that OOF does not exist** (A93).

## Two rules worth carrying forward

**How much tuning data a per-label knob needs (A111).** Optimism of per-label tuning at n_fit =
100 / 200 / **400 (≈ the public board)** / **870** / 1600 is **+0.0055 / +0.0032 / +0.0028 / +0.0016 /
+0.0014**, and per-label tuning beats one shared weight only from **n≈870** up (−0.0033 / −0.0015 /
**−0.0013** / +0.0002 / +0.0001). ⇒ **below ~870 studies, per-label tuning transfers *worse* than a
single shared weight** — the quantified case for keeping a softened/flattened hedge among the final
two. Our A5 OOF (4407) clears the bar; **the public board (~400) does not.**

**A submission confirms a locally-supported improvement; it never discovers one (A96).** Effects we
chase are 0.001–0.003 against ±0.008 noise, so "submit and see" is multiple comparisons on noise —
it can neither kill a good idea nor establish a bad one. This is why the final two were chosen on
evidence that is **not** the displayed score.

## Our own ensemble OOF (A112)

Fold 0 is the only fold with a rich arm set (13 arms; folds 3–4 have none), so the ensemble is
fold-0 only: n=870, which is exactly A111's safety line. Best single arm **0.8717**; rank-mean of
top-3 **0.8823**; **all 11 arms 0.8765** ⇒ ensembling helps (+0.005 to +0.011) and **pruning is real**
— the first local reproduction of D14. Metadata on top: **−0.0033 honest**.

## Known truth about our own instruments

- Our OOF fold discipline is **clean** (held-out fold removed from training; gold-58 never trained on).
- **Epoch selection uses gold-58** (`--topk 3`), so our gold-58 numbers are optimistic and the OOF is
  computed at gold-selected weights. **The whole published lineage shares this** (A98).
- **Report-group leak: 2 of 49 groups, 4 studies** (A100). **Scanner/site: measured, and it is a
  covariate, not a leak** (A101).
- Consequence to respect: **we cannot claim generalisation to new sites**; the data cannot express
  that claim either way.
- **Every `§` citation in FACTS is now machine-checked** (`scripts/check_facts_refs.py`): 24 citations
  are recorded gaps (§168–§191 were never written), everything else resolves (A119, §193).
