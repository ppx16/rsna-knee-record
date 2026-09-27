# STATUS — 2026-09-27 06:30Z

**One screen. Rewritten every session.** If this grows past a screen, the overflow belongs in a
`HANDOFF.md` section, not here.

## Where we stand

| | |
|---|---|
| Deadline | **2026-10-22 23:59 UTC** (~4 weeks) |
| Best ledger score | ⭐ **0.943** — a **three-way tie**: `56568520` (`0943` speedy) · `56570144` (`0943f` fracture) · `56570151` (`0943s` sparse). Then `56568524` (`0943h`) and `56570148` (`0943o`) at **0.942**. ⚠️ Eight of nine now read back: `0943c` and `0943r` also **0.942**, and **`0943p` alone at `0.941`** — the lowest of the nine, and a **recommended** pair member. That does **not** disqualify it (the pre-registered line is ≤0.940) — see the dose-ladder analysis in [`docs/FINAL-SELECTION.md`](docs/FINAL-SELECTION.md): the public board mildly preferring the *fitted* routing is exactly what A122 predicts, so it cannot discriminate. Ledger **36 rows, 35 scored**. |
| Public LB noise | **±0.008** — so **the board cannot rank these five arms** and no "which is better" conclusion is drawn from it (A96). All five sit inside the noise band. |
| ⭐ What the readouts actually bought | **All five arms completed the hidden rerun on 1322 studies.** Not just the baseline: both non-default presets and the reduced-window arm too. The submission path had only ever been exercised on the 3 visible studies. **That de-risking is independent of the score.** |
| GPU quota | **10.33 h / 30 h** used (**19.67 h left**), refreshes 2026-10-03. TPU **1.85 h / 20 h** (18.15 h left). ⚠️ The counter moves in **20-minute quanta** (A121), so it cannot resolve anything smaller. |
| Cost model | **1 h wall clock = 1 GPU hour even on the 2×T4 shape.** Scoring reruns are **not** billed — re-confirmed over a 7-hour idle window (A121). |

## 🔴 The one action left — and it is not something the CLI can do

**Kaggle has auto-ticked `56568520` (`0943` v2, 0.943) + `56570144` (`0943f`, 0.943).**
The recommended pair is **`56570144` (`0943f`) + `56568524` (`0943h`)**.

⇒ **Untick `RSNA knee 0943 - Version 2`, tick `RSNA knee 0943h - Version 2`.** Full reasoning, and
what leaving the default costs: [`docs/FINAL-SELECTION.md`](docs/FINAL-SELECTION.md).

⚠️ `kaggle` 2.2.4 has **no select command** (verified) ⇒ this needs the browser. And the default is
not predictable: three of our arms tie at 0.943 and auto-select took two of them, **not** the ones we
would choose (A123). Deadline 2026-10-22 23:59Z.

⭐ **The routing hedge is no longer an analogy.** The frontier notebook's own source comment says the
per-label `coatnet_w` overrides were set "on the strength of a public split that is a fraction of the
test set… It may be real. It is also the likeliest place to give back points privately, which is what
the `parent` preset is for" (A122, §194.3). That covers the *direction*; the *dose* is still
unresolvable — see the caveats in the selection doc.

**The choice set is now NINE arms, and the decision is pre-registered** — see
[`docs/FINAL-SELECTION.md`](docs/FINAL-SELECTION.md), which was written **before** the last readouts
land so they get read against a rule rather than fitted to a feeling.

| | |
|---|---|
| New since §196 | **`0943c`** = `0943` + **our rescued `cn0` as a fifth Raptor view** (built, smoke-tested, submitted `56599163`) and **`0943cf`** = the same **plus** the fracture fix — the two changes this repo actually has measurements for |
| Recommendation | **`0943cf` + `0943p`** — the pair puts the published per-label routing against the flattened one, i.e. it hedges the single knob whose provenance is documented as public-fitted, and covers three of the four axes between them |
| ⚠️ The board still cannot rank them | ±0.008 noise vs 0.001–0.003 effects; the readouts can only **disqualify** (a ≤0.940 return, or a failed run), and that rule is written down in the doc |
| ⚠️ An arm that was never submitted cannot be selected | eight are submitted now; `0943cf` follows its smoke run |

## Live right now

| Thing | State |
|---|---|
| Submissions | **one PENDING**: `56599932` (`0943cf`). ⭐ **`56599163` (`0943c`) read back `0.942`** — our rescued `cn0` member's **first hidden rerun**, and it **completed clean** on 1322 studies. `56599159` (`0943p`) `0.941`, `56599162` (`0943r`) `0.942`. |
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

## Where the remaining weeks go — 复盘 of 2026-09-26

**The review changed the answer: [`docs/REVIEW-2026-09-26.md`](docs/REVIEW-2026-09-26.md).**
**The score is at the public ceiling, and what is left is selection risk on knobs fitted to data we
cannot see** — unmeasurable *by construction*, not by lack of effort.

| | |
|---|---|
| 1. Hold the ceiling | done — five arms submitted, all ran the hidden rerun end to end |
| 2. Keep the final choice open | decide at the deadline, hedging different unmeasurable axes (`FINAL-SELECTION.md`) |
| 3. Look at the frontier, rarely | the only source of a genuinely new idea; the last sweep found nothing (A116) |
| 4. **Do not spend the weeks turning knobs** | 2026-09-26 is the demonstration — see below |

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
