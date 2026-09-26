# STATUS — 2026-09-26 12:30Z

**One screen. Rewritten every session.** If this grows past a screen, the overflow belongs in a
`HANDOFF.md` section, not here.

## Where we stand

| | |
|---|---|
| Deadline | **2026-10-22 23:59 UTC** (~4 weeks) |
| Best ledger score | ⭐ **0.943** — a **three-way tie**: `56568520` (`0943` speedy) · `56570144` (`0943f` fracture) · `56570151` (`0943s` sparse). Then `56568524` (`0943h`) and `56570148` (`0943o`) at **0.942**. |
| Public LB noise | **±0.008** — so **the board cannot rank these five arms** and no "which is better" conclusion is drawn from it (A96). All five sit inside the noise band. |
| ⭐ What the readouts actually bought | **All five arms completed the hidden rerun on 1322 studies.** Not just the baseline: both non-default presets and the reduced-window arm too. The submission path had only ever been exercised on the 3 visible studies. **That de-risking is independent of the score.** |
| GPU quota | **2.24 h / 30 h** used, refreshes 2026-10-03. TPU 0 h / 20 h. ⚠️ The counter moves in **20-minute quanta** (A121), so it cannot resolve anything smaller. |
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

**The choice set is now seven arms, all preflighted.** Two more were built and pushed on 2026-09-26
(§196) so that the deferred decision has something real to choose from: **`0943p`** (preset `parent` —
flatten the outer routing completely, the author's own registered insurance) and **`0943r`**
(`rad_cal_w` 0.40 → 0.00, the author's v38 move). Both are one-token forks of the same notebook; the
five existing arms rebuilt **byte-identical**, and preflight is 0 problems. ⚠️ Creating them widens
the choice set, it does **not** raise the expected score — and **an arm that was never submitted
cannot be selected**, so each needs one submission (free, 5/day) before the deadline.

## Live right now

| Thing | State |
|---|---|
| Submissions | **none pending** — all five read back, ledger is 32 rows. ⚠️ **`Remaining today: 0`** (5 used) ⇒ the two new arms below cannot be submitted — and therefore cannot be selected — until 00:00Z. |
| `0943p` / `0943r` runs | **both `COMPLETE`, both passed the acceptance test** (20/20 fingerprints, fail-closed gate PASS, valid `submission.csv`, 0 tracebacks). `0943p`'s log even shows `outer CoAtNet weight per finding: flat 0.65` — the flattening demonstrably took effect. |
| `kernel-tpu-d3f0` (DINOv3 fold 0) | **QUEUED** on TPU (queueing spends no quota; 0.00 h / 20 h used) |
| Kept for later | the pilot's log also revealed that the `[config]` line prints `coatnet_w` but **not** `rad_cal_w` — so for `0943r` the log cannot certify its own change; the proof is the artifact + the code path instead (A134) |

## Closed — do not reopen without new evidence

| Line | Why it is closed |
|---|---|
| **Training line** | Our best arm scores **0.8731** against A5's **0.9143** on the 181 studies where both are genuinely held out — a gap whose CI excludes zero (A106). Below the archive's own ~0.90 member bar. |
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
| `cn0` — the arm that produced **+0.0028** | ⭐ **weights RECOVERED (§201).** They were on the rented box after all — `/root/autodl-tmp/kaggle/working/`, which the first search missed because `-xdev` prunes that separate mount. Six checkpoints (1.68 GB) are now in `downloads/rent/` (git-ignored), verified by content (`testzip()` clean, 492 entries) and **identified by `best_gold_auc = 0.877929` matching the recorded 0.8779**. ⚠️ Still not *shippable* yet: they must be published as a Kaggle dataset, and the notebook edit is a separate, risky step. |
| the 22 fine-tuned checkpoints on disk | all below the frontier's own seeds (best available `r3362` 0.8663 vs their 0.868–0.870) ⇒ adding them would be a negative — `cvg0` already demonstrated that |
| the published `rsna-knee-train-weights` | 5 × DINOv2-small@336 @ crop 130 mm, 20 epochs; their blend scored LB 0.842 on 09-06 |

⇒ **A better member whose weights we hold is now the only thing that converts this line into score — which
makes the DINOv3 pilot the linchpin** (its checkpoint lands in the kernel's `/kaggle/working`).
⭐ And one cheap unblock, if you want it: **power the AutoDL box on for ~10 minutes** to pull `cn0`/`d64cn0`'s
checkpoints (≈ ¥0.5 at ¥2.88/h) — *if they still exist*; only the corpus is confirmed deleted.

⇒ **The live experiment (pre-registered, §197.5): train a DINOv3 arm on TPU.** Hypothesis: a DINOv3
backbone lifts a single arm from 0.877 to **≥0.90** (A5 = 0.9089). The zero-quota probe already
confirmed the backbone is reachable (timm 1.0.26, `vit_small_patch16_dinov3`, pretrained fetch and
336px forward both work, `num_features=384` so the head is unchanged). **Kill condition: fold-0
< 0.88 ⇒ stop and spend no more quota.** Trained on **TPU, not GPU** — the user's standing policy
reserves GPU for kernels that produce a submission (A132).

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
