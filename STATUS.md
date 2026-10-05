# RSNA Knee Abnormality Detection — STATUS

## ⭐ 2026-10-05 DELTA — the live state, and one suspicion checked and killed

### 1. Where we actually stand (snapshot 2026-10-05T07:59:16Z, persisted)

| | |
|---|---|
| teams on the board | **5192** |
| our rank / score | **518** / **0.943** |
| bronze line (top 10 %) | **519** |
| **margin** | **+1** — we are *one place* inside the bronze |
| `0.943` block | ranks **376..1389** (n=1014) |
| **in-block percentile** | **14.02 %** |
| top of board | **0.963** (2 teams), then 0.961, 0.960 |

Raw: `state/lb_snapshot_2026-10-05T075916Z.json`; appended to `state/rank_track.jsonl`.
Reproduce with `work/lb_snapshot.py` (it downloads the board, finds `ppxl16`, and persists both files).

**⚠️ The margin is not a buffer — it is a coin-flip.** The trajectory across 19 h:

| at | margin |
|---|---|
| 2026-10-04 12:44Z | **+23** (rank 487, line 510) |
| 2026-10-05 07:22Z | **+0** (rank 518, line 518) |
| 2026-10-05 07:59Z | **+1** (rank 518, line 519) |

⇒ **Without a further real gain we fall out of the bronze well before 2026-10-22.** ⚠️ The drift figure
quoted at the end of the 2026-10-04 DELTA ("≈ −2/day") was a **unit slip** and is corrected in place:
the true rate is ≈ **−40 ranks/day**. This does not change any conclusion, but it changes every
*deadline* estimate built on it.

### 2. One pending arm, and the rule it fires

`56845973` `0943cfmsft2m-men` (the author's public0033 DINov2-Base meniscus specialist at 90:10 on
Medial+Lateral Meniscus only) was submitted 07:22Z and is **still PENDING** — code reruns queue ~13 h.
`docs/PREREG-2026-10-05-meniscus-arm.md` fixes the reading in advance:

* in-block percentile **better than 14.07 %** ⇒ real gain, promote to final candidate;
* **flat or worse** ⇒ the "specialist" axis is dead too, and since the public-artifact, new-member and
  stacking axes are already falsified, **this round has no axis left and we should move to wrap-up.**

### 3. The Raptor crop was suspected mismatched, and it is NOT — checked, falsified

`CROP_MM = 130.0` is the first hit a grep finds, and `raptor-knee-native384`'s description says **140 mm**.
That would have been a silent mis-framing of every Raptor view. It is a **false alarm**: the notebook has
**per-block** preprocessing globals, and the Raptor block assigns `CROP_MM = 140.0` itself. No `ARMS`
entry carries a crop override, so all views correctly inherit 140 mm. Full text, plus the three dataset
descriptions that now independently pin our geometry: `docs/FIND-2026-10-05-raptor-geometry-confirmed.md`.

⇒ **Rule left behind: `CROP_MM`/`IMG`/`SLOTS` are not globals here — answer "what geometry do we use?"
per family, from the code path, never from a grep.** Same disease as `FIND-2026-10-04-stale-log-string.md`.

### 4. Evidence AGAINST the "add more backbones" intuition

`dreaddevelopment/raptor-knee-arms` = ConvNeXtV2-Base@336 + CoAtNet@384 + EfficientNetV2-L@480, same
2.5D window scheme and same head. Rank-mean blend = **0.915** public; **the CoAtNet arm alone = 0.914**.
⇒ the author added two architectures and an extra input resolution and bought **+0.001**, from a family
that sits ~0.028 *below* our current 0.943. **That axis is not the lever.**

### 5. ⚠️ A premise from this session that was WRONG: "v9 finespacing is an unbuilt lever"

`docs/FIND-2026-10-05-unused-strong-checkpoints.md` §2 read the **current** `0943*` `ARMS` table, found
no `finespacing-v9` entry, and concluded the gap was open and worth building. **The table observation is
correct; the conclusion was not.** `state/FACTS.md` records the prior:

* **`ref=56389731`** — `rsna-knee-ens-add-finespacing-arm` v1, submitted **2026-09-20**, ran at exactly
  80 slots / 78 windows / span 0.02–0.98 / img 336 / res 384 (the log confirms every field), and read
  **0.941 — identical to its own baseline** (**C49**, **C54**).
* **D14 closes the axis**: *"给 Raptor 臂加第 5 个视角 — 两次独立尝试、两次 0.941，关闭"* — the second was
  `add-widedense-arm` (v4 widedense), also geometry-verified.

⇒ **Re-adding v9 at the same ~0.20 dose would be a re-run of a measured-flat experiment.** The only never-run
variant is the dose test `add-finespacing-heavy` (v9 = 0.50), built and unpushed since **A32**. D14 says do
not spend on it; this file now agrees rather than treating it as the next step.
**Lesson, applied: before calling anything an "unfilled gap", grep `FACTS.md` for the artifact — not just
the current table for the string.** A gap in a table is not a gap in knowledge.

### 6. ⚠️⚠️ The GPU budget was mis-read by ~20×, and the error was in the DENOMINATOR

For this entire session I treated GPU as nearly spent and **used that to reject experiments**.
The figure I quoted was `19664 s / 21600 s` ⇒ "≈ 32 min left". Live, 2026-10-05:

| | used | remaining | total | refresh |
|---|---|---|---|---|
| **GPU** | 5.46 h | **24.54 h** | 30.00 h | 2026-10-10 |
| **TPU** | 3.79 h | 16.21 h | 20.00 h | 2026-10-10 |

```
19664 / 3600 = 5.46 h  == GPU used, live      ✓   (the numerator was right all along)
21600 / 3600 = 6.00 h  != GPU total, 30 h     ✗   per-SESSION cap, not the weekly allowance
```

⇒ remaining should have been `108000 − 19664 ≈ 24.54 h`, not 1936 s. **Same numerator, wrong total.**
⚠️ **This is why "refresh the page" would not have fixed it — a stale number and a wrong unit look
identical from outside.** It also reached `docs/PREREG-2026-10-04-convnext-member.md`, which framed the
ConvNeXt member as affordable **only as a fold-0 probe** when the full 5-fold (~13 h) was always in budget.
`docs/FIND-2026-10-05-quota-denominator-error.md`. Tool: `work/quota.py` (appends to `state/quota_log.tsv`,
which had been stale since 2026-09-26 — the log existed but nothing wrote to it).

**Rule: never record or quote an accelerator budget as a bare seconds count or single percentage. Read
`kaggle quota` at the moment of decision and always print `remaining / total` together** — the pairing is
what makes a unit error visible. ⚠️ Run it via `D:\kaggle\venv-kaggle\Scripts\kaggle.exe`; the PATH
`kaggle` failed 6/6 on the OAuth path while the venv one succeeded first try.

### 7. The "F_dim 1024 vs 768" blocker on `raptor-knee-arms` was a DEAD DEFAULT — resolved

`docs/FIND-2026-10-05-unused-strong-checkpoints.md` §1 warned that `coatnet384`/`cnv2b336` carry F_dim
**1024** while our v5/v8/v10 load through **768**, and that this had to be settled before wiring.

**Measured (`work/resolve_fdim.py`): our v5/v8/v10 are ALSO 1024.** Head tensors are identical across
every CoAtNet/ConvNeXtV2 checkpoint here — `att.0.weight (256,1024)`, `clsW (12,1024)`, `norm.weight (1024,)`
— and timm reports `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k num_features = 1024`. **Both sides were 1024.**

`768` occurs **once** in the notebook: `def __init__(self, backbone, F_dim=768, ...)` — a **default
parameter**. The only call site is `RaptorClassifier(bb, F_dim=bb.num_features)`. It is never reached, and
the load is `rsna_strict_load` (strict), so a real mismatch would have **raised**, not failed silently.

⇒ **The family is safely wireable.** What genuinely differs: `arch` must match per-arm (`load_model` raises
on arch drift, so `cnv2b336`/`effv2l480` need their own arch strings), and `effv2l480` really is **1280**
wide with an EfficientNet tail. `docs/FIND-2026-10-05-fdim-768-was-a-dead-default.md`.

⚠️ **This is the THIRD instance of one disease this session** — `CROP_MM` (per-block, not global), the
stale `[raptor-fast] ... cuda:0=[0,2]` log line, and now a dead default argument. Each wrong answer came
from reading **a declaration** instead of **the path that executes**.
**A grep finds where a value is written; it does not tell you whether that value is ever used.**

### 8. Built, NOT fired: a self-trained hard-label specialist (`msft3-spec`)

`docs/PREREG-2026-10-05-self-trained-specialist.md` + `scripts/build_msft3_specialist.py`
(`--check` passes for both tiers: anchors unique, compiles, 11 probes present). **Nothing pushed.**

**⛔ Hard entry gate: it may only be fired if the pending meniscus arm (`56845973`) reads a REAL gain**
(in-block percentile better than 14.07 %). If meniscus is flat, the label-specialist hypothesis is dead
and this is compute spent on no hypothesis.

It is not "a third same-corpus CoAtNet" (which the stop rule forbids) because it differs in three places
at once: the **loss is masked** to six labels, the **epoch selection is on that group's AUC**, and the
**blend target is that group only**. ⚠️ The second is not optional — `evaluate()` returns the all-label
macro and the loop selects on it, so a masked loss without group-aware selection would train one
objective and select on another, silently. The six labels are the ones the base's **own log** names
(*"Medial Meniscus / Medial OA / Lateral OA / Effusion / Synovitis / Contusion 是各臂真正不同…"*) — the
same sentence the meniscus prereg rests on, so the two experiments share one hypothesis.

**My stated expectation: probably flat-to-negative** (3 reasons in the prereg; the 11× decay was measured
on this corpus, and msft3-spec is still a fine-tune of it).

⚠️ **One deliberate deviation from A167, flagged:** A167's member gate (solo ≥ ~0.90) is an **all-12-label**
bar. A model trained on six labels cannot clear it, so applying it here would reject the experiment with
the wrong ruler. The gate is therefore computed **on the six-label group**. This is a real judgement call,
not an oversight.

Unverified and stated as such in the prereg: it has **never run a forward pass**; the rebuilt optimizer
(needed because the author's `AdamW([{bb},{head}])` raises on an empty group when the whole backbone is
frozen) is the likeliest place for a surprise.

---

## ⭐⭐ 2026-10-04 DELTA — READ THIS BEFORE ANYTHING BELOW

**This section supersedes several statements further down, which are now known to be WRONG.
The biggest one: `msft2` is NOT closed and its measured value is NOT zero.**

### 1. The public board has a sub-resolution instrument we were not using: the RANK

`0.943` is displayed for **953 teams**, yet they occupy **contiguous distinct ranks 357..1309**.
⇒ **Kaggle ranks on precision finer than the 3 displayed decimals.** Consequences:

* **the target is NOT "+0.001 to reach 0.944"** — it is to move up ~21 % of the block width, i.e. **≈ +0.0002**;
* the repo's premise **A96 ("the board cannot rank these arms, only disqualify") is true for the
  displayed SCORE and false for the RANK**.
* ⚠️ **The rank is confounded by field drift** — always record a snapshot BEFORE and AFTER a submission
  and compare the **in-block percentile** `(rank-lo)/(hi-lo)`, never the absolute rank.
  Measured drift with no improving arm: **−31 ranks in 16 h**.
  ⚠️ **CORRECTED 2026-10-05 — the "≈ −2/day" that used to end this line was a unit slip**
  (31/16 = 1.94 per **hour**, not per day). Two measurements now agree:
  −31 ranks in 16 h ≈ −47/day, and the 2026-10-04→10-05 margin collapse (+23 → 0 in 16.3 h) ≈ **−34/day**.
  **Working figure ≈ −40 ranks/day ≈ −1.7 ranks/hour.**

Full text: `docs/FIND-2026-10-04-rank-resolution.md`.

### 2. ⭐ `msft2` was built, submitted, and **measurably improved us** — killing §13.2's expectation

| arm | ref | result |
|---|---|---|
| `0943cfmsft2` = `0943cf` + our msft2 CoAtNet SWA as a **6th Raptor view** (w=0.20, share 14.3 %) | **`56814081`** | public **0.943** — **but rank 708 → 487** |

| | before 04:13Z | after 12:44Z |
|---|---|---|
| our rank | 708 | **487** |
| `0.943` block | 344..1280 (937) | 357..1309 (953) |
| bronze line (top 10 %) | 506 | 510 |
| **in-block percentile** | **38.89 %** | **13.66 %** |

**⇒ +221 ranks while the field grew 0.8 %; the date confound was excluded
(`pearson(rank, submission_time) = −0.13`, dates scattered at both ends).**
**⇒ rank 487 < line 510: we are currently INSIDE the bronze, by 23 places.**

⚠️ **This falsifies, in this case, HANDOFF §13.2's "a 0.910 member at 10–15 % weight → 0.000".**
msft2 is 0.9069 at a 14.3 % share and it moved us. That "0.000" was a **single noisy LB reading**,
and both I and this file treated it as a conclusion. **My own pre-submission expectation of "≈0" was
wrong.** Full text: `docs/RESULT-2026-10-04-msft2-arm-rank-move.md`.

### 3. Two other arms landed, both flat — and one of them closes a long-open question

* **`56798408` `0943cf448`** = `0943cf` + the author's PUBLIC one-fold M448 ConvNeXt fallback at w=0.10
  → **0.942**, in-block percentile 39.5 % → 38.9 % ⇒ **no measurable change.**
  ⚠️ The author's **trainer is not public** ⇒ folds 1–4 cannot be reproduced ⇒ **that line is dead.**
  `docs/PREREG-2026-10-03-m448-arm.md`.
* **`56791368` `0943cfm`** (+MCL) → **0.942**, and the pre-registered rule fired (≥0.942 ⇒ legitimate
  primary candidate). The amendment it triggered changes the **second slot** to
  **`56613303` `0943cflat`**. `docs/FINAL-SELECTION-2026-10-04.md` (amendment at the end).

### 4. What the 356 teams above us actually use — third exhaustive sweep

`docs/FIND-2026-10-04-what-is-above-us.md`. Headline: **the top 27 publish nothing for this
competition**; 188/464 of the usernames above us publish *something*, but only ~21 publish knee work,
and those are the same public-lineage authors whose ceiling is 0.943. **Top submission counts 26–298
vs our 41.** The one complete public recipe found was **a faithful replica of the pipeline we are
already on** — including the answer to a question this repo had left open:
**our `_RAD_CAL`'s "unknown provenance" is the public recipe's frozen 12-output calibration table.**

⚠️ **We are NOT missing DINOv3** — `mattiaangeli/knee-mri-fold-weights` (our `a5_w` member) is
**byte-identical** to the bundle's `dinov3/m_f0.pt`, backbone `vit_small_patch16_dinov3.lvd1689m`.
An earlier note in this session claiming "we use DINOv2, they use DINOv3" was **wrong**.

### 5. Open right now

| | |
|---|---|
| **The two final slots** | still a **browser-only** action. `0943cfmsft2` (`56814081`) is now the only arm with a **measured** gain. |
| **`ppxl16/rsna-knee-msft2-cx`** (ConvNeXt fold-0 fine-tune) | **QUEUED on TPU.** Pre-registered kill rule: **SWA gold < 0.88 ⇒ stop, do not run folds 1–4.** `docs/PREREG-2026-10-04-convnext-member.md` |
| **Drift** | ≈ **−40 ranks/day** (−34 to −47 measured twice, 2026-10-05). ⚠️ Earlier text here said "−2/day" — that was a **unit slip**, corrected. **The margin is now +1 (see the 2026-10-05 DELTA at the top), so we are one place from falling out.** |
| Strategy | `docs/STRATEGY-2026-10-04-how-far-can-finetuning-go.md` — same-corpus fine-tuning is **measured** to have decayed ~11× (cn0 +0.0028 on a 0.917 base → msft2 +0.00025 on a 0.943 stack). Stop rule proposed there. |

---

# (historical) STATUS — 2026-10-02 08:05Z

**One screen. Rewritten from scratch; the previous version was dated 2026-09-27 and had drifted.**
Deep detail lives in `FACTS.md` (A-numbers) and `docs/`; the recent session records are
`docs/RESULT-2026-10-02-msft2-finetune.md`, `docs/RESULT-2026-10-01-exp085-gate.md`,
`docs/FIND-2026-10-01-exp085.md`, `docs/FIND-2026-10-01-rsna-sweep.md`.

⚠️ **Everything from here down is the 2026-10-02 view and is retained only so the reversal is
auditable. In particular: the "msft2 is D7-closed" and "expectation ≈0.000" statements below
were REFUTED on 2026-10-04 (see the DELTA at the top).**

## ⛔ READ THIS FIRST — 2026-10-02: the msft2 / ConvNeXt plan was pursuing CLOSED directions

`state/EXPERIMENTS.md` §D is a ledger of every lever already **tested and closed**. It closes most of
what this file has been treating as open:

| closed | what it kills |
|---|---|
| **D7** | **"把我们的 raptor 成员塞进集成" is CLOSED.** The *author themselves* measured "adding a different-architecture backbone" = **+0.001** (C14), and we independently A/B'd it (`add-widedense-arm`) = **0.941 flat**. **⇒ the msft2 member is D7.** |
| **D5 / D6 / D17** | self-trained encoders with **solo <0.90 are net harmful**; the TPU route closed after predict crashed natively; TPU-trained coatnet closed. **⇒ the ConvNeXt arm is D5/D17.** |
| **D14** | "add a 5th view to the Raptor arm" — **two independent attempts, both 0.941 flat.** |
| **D1 / D3 / D13** | fusion-weight / routing / calibration tuning, incl. the community's "one weight" claim: **five readings all 0.941**; mattiaangeli's own metadata says *"sub-resolution blend tweaks should be treated as noise"*. |
| **D11 / D12** | turning the routing or calibration switches OFF is **negative** (monotone 0.941 / 0.940 / 0.939 / 0.938 four-corner). |

**And the instrument cannot resolve the target.** Public-LB **SE ≈ 0.008** ⇒ **differences ≤0.003 are
unjudgeable here**, while our target is +0.001. Confirmed too: `exclude-fracture` — the one
externally-claimed "+0.001" — was **already submitted (`56166772`) and read 0.941, same as canonical.**

⚠️ **RESOLVED 2026-10-03 — what bought the +0.002, and it is not a lever we still hold.**
`state/FACTS.md` **C79** records that the public frontier wrote down its own 0.942→0.943 changes, and
**we forked that finished artifact**. The three changes are:

1. a **third CoAt reader, `Global96`** — same 96-slice stack as D4 but a **different head** (full-width
   gated head vs D4's Rank8 finding-to-slot attention). Their words: *"共享几何、不同读者，这才是
   集成真正用得上的多样性"*;
2. the **reduction**: from *rank-each-then-average-the-ranks* to **average-the-probabilities-then-rank**
   (the former throws away confidence);
3. the outer CoAtNet default **0.60 → 0.65**.

Plus the Raptor window count **62 → 94** (with the speed engineering that made it affordable), and
**C81**'s removal of `'Fracture'` from `_RAD_EXCLUDE` — a decision that had been made on **gold-58**
and was **wrong**: re-measured on 4185 studies it is Fracture AUC 0.9187 → **0.9305**, reproduced
independently by five label sources, with **gold-58 the only dissenter at −0.0097**.

⇒ **All of it is already inside our kernels** (`kernel-0943f` is the fracture fork; A118 records
`3×0.943 + 2×0.942` across the five arms). **We are not below the public recipe — we are on it.**

⇒ **And the frontier itself annotated this whole change as "排行榜只动了 0.001"** — a three-part
structural edit plus 32 extra windows moved the public LB by 0.001. Combined with **LB SE ≈ 0.008**,
that is the clearest possible statement that **our +0.001 target is below this instrument's
resolution**, and that a null LB result cannot distinguish "no gain" from "gain we cannot see".

⚠️ **One recorded barrier is now stale and should not be quoted:** **D17** says CoAtNet cannot be
trained on `torch_xla` (it died in `timm/layers/pos_embed_rel.py` → `torch_xla/csrc/helpers.cpp`
broadcast check). **Our own msft2 v4 falsified that** — it trained
`coatnet_rmlp_2_rw_384.in12k` on TPU for 4/4 epochs to `partial: false` once the XLA activation-
checkpoint shim was in place. So "convnext is only a substitute" is no longer the reason to avoid
CoAtNet on TPU. (This does **not** revive the msft2 member: that is still closed by **D7** on
ensemble-value grounds, and by **D5** on the sub-0.90 rule — a different and better reason.)

## Where we stand

| | |
|---|---|
| Competition | **RSNA Knee Abnormality Detection** — code competition, 12-label macro ROC-AUC |
| Reward / teams | **$77,000** (incl. an **$18,000** efficiency track) / **4,692** scored |
| **Deadline** | **2026-10-22 23:59Z** — 20 days |
| Daily submissions | **5** (verified from the API) |
| ⭐ Our board | **rank 578 / 4692 · score 0.943 · 38 submissions** (`p pxl16`) |
| Board max | **0.961** — gap **+0.018**, i.e. 2.3× the ±0.008 noise band |
| ⭐ **The target** | **+0.001 → 0.944.** Our 0.943 is a **599-team block (ranks 263–861)** and the bronze cutoff (rank ≤ 462) falls **inside** it, so at 0.943 whether we medal is decided by an **opaque tie-break**. 0.944 is a 47-team block at ranks 216–262 — **+0.001 escapes it entirely.** |

## ⭐⭐ The headline: the fine-tune WORKED and clears the member gate

`ppxl16/rsna-knee-msft2` v4 — **COMPLETE, 4/4 epochs, `partial: false`**:

```
[zeroshot] GOLD macro-AUC 0.8961 | VERDICT: PASS (parent bar 0.8963)   <- warm start verified
ep0 0.8903 | ep1 0.8946 | ep2 0.9007 | ep3 0.9048
SWA over epochs [3,2,1]  gold 0.9069   <- better than the best single epoch
```

| reference | gold58 |
|---|---|
| parent checkpoint reloaded through **our** pipeline | **0.8963** (the bar) |
| this run, zero-shot (warm start, no training) | 0.8961 — PASS |
| this run, best epoch | 0.9048 |
| ⭐ **this run, SWA** | ⭐ **0.9069** |

**+0.0106 over our own warm-start number, and 0.9069 CLEARS the A167 member gate (`solo ≥ 0.90`).**
A167 established that **nothing this project can train from scratch passes it** (our ceiling ~0.877), so
fine-tuning a published strong checkpoint was the only route — **this is the first artifact of ours that
passes.** Calibration: exp085's author reports `gold58 0.9062`; our SWA reaches **0.9069**.

✅ **DONE — the checkpoints are now a dataset we own**: `ppxl16/rsna-knee-msft2-ckpt` (private, 8 files),
published 2026-10-02. The exp085 failure mode (a public weights dataset withdrawn mid-session, then
silently dropped by `kernels_push` while it reported success) no longer applies to this asset.

⚠️⚠️ **BUT the premise that this is "paid-for headroom" is contradicted by our own recorded evidence.**
`docs/HANDOFF.md` §13.2 (2026-09-11, from others' LB-measured results) says:

> **加自训模型没用。** 0.910 的模型按 0.10/0.15 权重加进 0.936 的公开底 → 变化 0.000；
> **0.874 的模型加进去反而 0.936 → 0.935（更差）**。基于 Gold58 预测的 +0.004 实际交出了 −0.004。

> **58 例 gold 集不能当裁判。** 它的 bootstrap 95% CI 比要追的效果还宽…… **不要用 Gold58 调权重。**

**msft2 is 0.9069 — inside the band that was already measured at 0.000.** It is also a CoAtNet, i.e. the
same *family* as the CoAt readers already in the stack, and §13.2 says the ensemble lives on
**divergence, not quality** (the weakest member, solo 0.843, contributed +0.009; another of the
*strongest* DINO folds contributed −0.0007). So **the honest expectation for msft2 is ≈0.000, possibly
slightly negative** — and the "0.943 → 0.944 is untested" framing below must be read with that in mind.
The arm is therefore built to **measure divergence first** and defaults to weight 0.0.

## ⚠️ Where the headroom is, and where it is NOT

### Closed: better public artifacts
Full sweep of newest + top-voted notebooks, datasets and the forum
(`docs/FIND-2026-10-01-rsna-sweep.md`). **The public frontier IS 0.943 and we are on it:**

| name claims | the artifact's own evidence |
|---|---|
| `…-d4-public**0946**` family | **0.943** — *"the 0946 … overstates what the public pipeline alone achieves"* |
| `0-biohub-…-**0947**-lb` | private assets |
| `parent**0957**-routing` | **0.941** — *"Public score 0.941"* |
| `the-**2965**-master-hybrid-engine` (Kaggriculture, 09-29) | **byte-identical to `frontier-top2`** |

⇒ **A notebook name on this competition is systematically inflated, always upward.** Only the
artifact's own text and a local gate are admissible.

### Closed: exp085 as a member
Reconstructed from its checkpoint alone (weights + config, code private). Module tree **proven**
(`strict=True`, 245/245), **head order A** recovered (+0.13), **`eval_group_agg: mean`** recovered
(+0.02, max/median ruled out), `slice_select` and `crop_mm` **eliminated**. Best **0.8191 against the
author's 0.9062** — **below the 0.90 gate**, and the dataset was then made private. Recorded and closed.

### ⭐ OPEN — the one thing that matters, and it is not built yet
**The msft2 member has never been added to the ensemble and never submitted.** +0.0106 on gold is a
*solo* number; **A163 says a member must buy DECORRELATION, not just strength.** Until an arm carries it
and the board is re-read, `0.943 → 0.944` is untested. **This is the whole of the remaining headroom
that is already paid for.**

## Are there OTHER arms worth fine-tuning? — yes, but only ones we can still attach

The msft2 recipe is: take a **published strong checkpoint**, carry the head, **verify the warm start at
zero training**, then fine-tune with a low backbone LR and a higher head LR for ~4 epochs, keeping the
best epoch **and** an SWA. Applying it to a **different family** is how a second, *decorrelated* member
would be produced — which is what A163 demands.

| candidate | accessible? | why it is interesting |
|---|---|---|
| `kawaharataishi/rsna-knee-coatnet-distilled-checkpoints` (543 MB, 09-29) | ❌ **403** | the D4/CoAtNet distilled family — the most on-family candidate |
| `tbongc/rsna-knee-raptor-v8geo/v10geo` (271 MB each) | ❌ 403 | Raptor variants, same family as msft2 → **low decorrelation value anyway** |
| `apoorvadixit/rsna-knee-weights-v4cx` (5 × 199 MB) | ✅ | **ConvNeXt-small** — a *different architecture*, so the best decorrelation candidate on the list. ⭐ **CHOSEN — `v4cx0` is running now as `ppxl16/rsna-knee-msft2-cx` (QUEUED, TPU).** |
| `soyaoki/rsna-knee-my-models` (20 files, 89 MB each) | ✅ | a 5-fold set with `_epoch5`/`_epoch10` variants |
| `lixin73/rsna-knee-train-checkpoints` (82 MB) | ✅ | unchecked |

⚠️ **Budget check before starting one**: TPU remaining ≈ **17 h** and a 4-epoch run takes **2.6 h** — so
several are affordable, but the **GPU quota is 0.00 h until 2026-10-03**, and **no arm can be pushed or
read before then.** A second fine-tune can be *run* today; it cannot be *used* today.

## The instrument, and its honest limits

* the metric is **macro ROC-AUC over 12 binary findings** — only **ordering per column** counts, so any
  monotone transform is score-neutral *alone* and matters only inside a blend;
* **ties are expensive**: a study nobody can score gets a constant, and a block of identical constants
  earns half credit against everything it ties with — **the same mechanism as the 599-team block that
  decides our medal**;
* **58 gold studies is a small set.** The +0.0106 is real but has poor precision, and it is a *solo*
  screen, not a leaderboard prediction;
* local OOF cannot settle member value: **A84** established the public artifacts are checkpoints rather
  than features, so `coat_private_alpha` and the individual CoAtNet members **cannot be priced on OOF**.

## Blockers, both with dates

1. ⚠️ **GPU quota 0.00 h account-wide until 2026-10-03** — `enable_gpu` kernel versions cannot be
   created, so **no new arm can be pushed or scored for ~18 h.** (ARC is blocked by the same quota.)
2. **The efficiency track needs a browser action**, and the board has no page (it is a staff notebook
   output). ⚠️ The user has said plainly they cannot see it — **treated as out of scope**; the main
   track is the target.

## Next actions, in order

> **2026-10-04 delta — current sheet is `docs/FINAL-SELECTION-2026-10-04.md`.**
> * ⛔ **EffB0-MIL LINE CLOSED 2026-10-04 (user decision).** v3 COMPLETE: the per-term pin fixed
>   (**7/7 PASS**), dose **0.150** confirmed, and the safety net fired cleanly (`_EffOutOfTime` →
>   `_pipeline_stage.csv` = exactly `0943cfm`'s bytes ⇒ **duplicate, not submitted**). **The real
>   blocker was never the budget: the rebuild produced `0 slot-series` for all four TRAIN studies, so
>   the gate would REJECT even with unlimited budget.** Closed on four reasons — falsified rationale,
>   effect below the board's resolution, cost, and nothing new left to learn. Full text:
>   `docs/RESULT-2026-10-04-effb0-v3.md` (closure section).
> * 🔄 **`0943cfm2eff` v3 was the last EffB0 run** — nothing further will be pushed for this line.
> * ✅ **`0943cfm` = `56791368` (+MCL) LANDED: COMPLETE, public 0.942.** The pre-registered rule fired
>   (§"Steps" step 5 of `docs/FINAL-SELECTION-2026-10-04.md`: ≥0.942 ⇒ legitimate primary candidate).
>   ⭐ **The amendment it triggered is in that doc** — the second slot changes from `0943p` to
>   **`0943cflat` = `56613303`**, an arm the current sheet had dropped from its candidate table even
>   though **A161** had already recorded that it *weakly dominates* `0943p` (same flattening, plus the
>   +0.0028 member and +0.00099 fracture fix). The "concentrated bet" objection that stopped A161 from
>   acting **does not survive being written out**: over the better-of-two aggregator, pairing
>   (`0943cfm`, `0943cflat`) is **pointwise ≥** (`0943cf`, `0943p`) whenever the member/fracture effect
>   and the MCL effect are ≥ 0. **This is now the recommendation; the click is still browser-only.**
>   ⚠️ New: **`submission-limits` is not a verb in kaggle 2.1.2 *or* 2.2.2** (`2.2.2` has
>   `team-submissions` instead), so A135/A162's quota reading is **no longer obtainable from the CLI**.
> * ✅ **Reusable from the closed line**: `scripts/verify_effb0_arm.py` (acceptance test, battle-tested),
>   `scripts/harden_effb0_budget.py` (built, verified, never pushed), and the honest-dose method
>   (`work/eff_min.py`, `eff_dose.py`, `eff_dose_parents.py`, `eff_convention.py`).
> * ⭐ **EffB0 findings that stand regardless**: honest OOF +0.00403 at w=0.10, **4/4 folds**, CI
>   [+0.00285,+0.00491], and **the OOF estimate EXCEEDS the in-sample one**; **w=0.15 is the largest
>   dose unanimous under BOTH soft and binary conventions**. ⛔ But "helps most where we are weakest"
>   is **FALSIFIED** (weakest-3 +0.00414 vs strongest-3 +0.00544; Fracture +0.0013; Baker's negative).
>   `docs/FIND-2026-10-04-effb0-honest-test.md`.
> * ⭐ **MEDAL BAND, measured from a third-party leaderboard scrape** (`docs/FIND-2026-10-04-medal-band-density.md`):
>   the bronze line is **0.943 with 794 teams in that single 0.001-wide bin** (16.4 % of the board), and
>   **the line has been FLAT at 0.943 since 2026-09-25** while the field grows (`bronze_rank` 455→482).
>   ⇒ the medal is decided **inside a dense band**, so **this corrects my "below resolution so pointless"
>   framing**: it was right about *measurement* and wrong about *ranking*. A sign-stable +0.001 has real
>   expected rank value even though the board cannot see it — which **strengthens the MCL bet**.
>   ⚠️ The panel's `our_*` fields are the *collector's* account (0.948), not ours; the band is public
>   while the medal is private, so it is a proxy for density, not for ordering.
> * ✅ **Provenance AUDIT (`docs/AUDIT-2026-10-04-provenance.md`)**: `kernel-0943` is a **byte-identical**
>   fork of the frontier source (**0 of 51 cells differ**); `0943cf` = +cn0 view +fracture fix;
>   `0943cfm` = +MCL. No accidental deviations anywhere.
> * ✅ **`AMP_PREF='bf16'` CLOSED on provenance** — byte-identical in the frontier source, so switching
>   would deviate from a validated config on a foreign anecdote (§5a of the artifact scan).
> * ⚠️ Also recorded: `EXPERIMENTS.md` D1 says "all pre-registered axes" but **`rad_alpha` was never
>   submitted** — closed by *category*, not by a completeness proof.
> ⚠️ **Acceptance test for `0943cfm2eff` is a script: `scripts/verify_effb0_arm.py`** — it decodes the
> **UTF-16** log (which `kaggle kernels logs … *> file` writes under PS 5.1 and which made a successful
> run look empty), parses the gate verdict, and compares the arm against its parent.
> ⛔ **Do NOT judge a duplicate by `_pipeline_stage.csv` alone.** That file holds **ranks**, and the
> public run covers only **3 visible studies** (6 possible orderings per label), so a genuinely
> applied change can leave it byte-identical. **Verified against a known-applied change**: `0943cfm`
> (+MCL) and `0943cf` have the *same* 3-study stage file and `submission.csv`, while their
> `v558_rad_receipt.json` and `_coat_arm_receipt.json` **differ** — so MCL did apply, and
> **`56791368` is NOT a duplicate**. The **receipts are the primary signal**; the log arbitrates; the
> stage hash is corroboration only. An earlier version of this rule would have rejected a working arm.

0. ⭐⭐ **THE ONLY ACTUAL DECISION LEFT: the two final submission slots — and it is a BROWSER job.**
   `docs/FINAL-SELECTION.md` is the pre-registered decision (written before the last four readouts
   landed, then read against its own table). Its recommendation is **`0943cf` + `0943p`**:
   * **primary `0943cf` = ref `56599932`** (public 0.943) — the only arm carrying **two** locally
     measured changes (`cn0` as a 5th Raptor view *and* the C81 fracture fix), and the only member
     this repo has ever measured **above** the frontier's own DINOv2 seed members (0.8769 vs
     0.8676–0.8697, A130);
   * **second `0943p` = ref `56599159`** (public 0.941) — outer routing **fully flattened**, i.e. the
     frontier's *own* written remedy for the per-label map it describes as tuned on "a public split
     that is a fraction of the test set" and as "the likeliest place to give back points privately".
     Pre-registered alternative at the same hedge direction, one notch weaker: **`0943h` = `56568524`**
     (0.942).
   ⚠️ **This supersedes A118's `0943f` + `0943h`.** ⚠️ **The auto-ticked default is NOT this pair**
   (Kaggle auto-selects `0943` + `0943f`, A123) — so it must be changed by hand.
   ⚠️ **No CLI/API route exists**: `kaggle` has no `select` command, and `ApiSubmission` exposes no
   `selected` field (re-verified 2026-10-03, kaggle 2.1.2). Only the browser can do this.
   ⚠️ The doc explicitly leaves one thing to human judgment: the recommended pair deliberately carries
   the **single lowest-scoring** arm, because the public board *cannot* rank these arms — nine arms
   spanning four axes came back across a total spread of **0.941→0.943 = 0.002** against **±0.008
   noise**. If carrying the minimum is unwelcome, take `0943cf` + `0943h`.

0b. ✅ **`ppxl16/rsna-knee-msft2-cx` DELETED from the TPU queue** (2026-10-03). It was still `QUEUED`,
   so **nothing was destroyed**, and the single TPU slot (D18: concurrency = 1, a *queued* session
   already occupies it) is now free. Two independent reasons: the ConvNeXt member is closed on
   ensemble-value grounds (**D5** sub-0.90 rule, **D7** measured-flat), and TPU had only **1.88 h of
   20 h** left while the run needs ~2.6 h — so it could not have finished anyway.

0c. ⚠️ **`state/lb_ground_truth.csv` was STALE and has been regenerated** (2026-10-03, now 38 rows).
   It was missing `56669951` = **`0943s-2coat`**, public **0.942**. That arm is a clean single-axis
   A/B: `0943s` (sparse, **with** Global96) 0.943 → **minus Global96** 0.942, i.e. **Global96 measures
   at ≈ −0.001** — the same sign as C79's account, but well inside ±0.008, so it does not resolve.
   Its own description shows it is an **efficiency** play (≈50 min less hidden-rerun runtime against a
   break-even AUC cost of 0.034), not a score play.

1. ✅ **DONE — `ppxl16/rsna-knee-msft2-ckpt` published** (private, 8 files, `EXIT=0`). The exp085
   failure mode — depending on a dataset we do not control — is closed for this asset.
2. ⛔ **msft2 / ConvNeXt as ensemble members: CLOSED.** See the D-table at the top.
3. ⭐ **2026-10-03, GPU quota refreshed** — but there is **no member left to push.** GPU budget noted
   in `FINAL-SELECTION.md` was 21.85 h / 30 h at time of writing; selecting spends none.
4. **The only unmeasured axis left is corpus density**, and by the project's own pre-registered banding
   its best measurement (+0.0068 soft OOF, C84) lands in *"only worth upgrading the instrument, not
   building a new member"*. The layer-selection rule needed to build a denser corpus is **fitted and
   validated 34/35** (C82), so it is an engineering job on CPU, not a compute bet.

<details>
<summary>Superseded 2026-10-02 plan (kept only so the reversal is auditable)</summary>

1. ✅ `ppxl16/rsna-knee-msft2-ckpt` published (8 files, `EXIT=0`) — still true and still valuable.
2. ~~🔄 `ppxl16/rsna-knee-msft2-cx` IN FLIGHT on TPU~~ — **deleted, see 0b.** It was pursuing **D7**
   (our-member ensemble value measured flat) and **D5** (sub-0.90 members are net harmful). Its v1 had
   already died at the head guard; v2 was relaunched before the D-table was read.
3. ~~Build the msft2 arm into `kernel-0943s-2coat`~~ — **cancelled.** The arm was D7-closed before it
   was built; the subagent was stopped and left nothing behind (the 0.943 notebook is untouched,
   mtime 09-29).
4. ~~Push the arm and submit on 10-03~~ — **cancelled**, per the same D7/D5 closure.

⚠️ **Why the reversal happened:** this file's premise was "the msft2 member is the whole of the
remaining headroom that is already paid for". That premise came from reading **A167/A163** (a member
gate and a decorrelation rule) without reading **`state/EXPERIMENTS.md` §D**, which had already
*measured* the answer. The lesson recorded for next time: **before designing an experiment, read
§D of the ledger — it is the list of experiments already run.**
</details>
