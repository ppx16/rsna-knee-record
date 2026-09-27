# Final submission selection — the decision, pre-registered

**Deadline: 2026-10-22 23:59 UTC.** Two submissions may be selected, and it is a **two-click job in
the browser**: the Kaggle CLI cannot do it (verified on 2.2.4 — `kaggle competitions` exposes
`list/files/download/submit/submissions/leaderboard/team-submissions/submission-limits`, no `select`).
The recommendation below is written **before** the last four readouts land, so that whatever they
show gets read against a rule instead of being fitted to a feeling. Every claim carries its FACTS row.

---

## The rule that governs everything below

**The public board cannot rank these arms.** Its noise is ±0.008 and the effects in play are
0.001–0.003 (A96). Three of our arms scored 0.943 and two 0.942, and that ordering carries no
information. So the choice is made on **what each arm changes and what evidence that change has**,
never on the displayed score.

⚠️ Corollary that cuts the other way: **a readout can still catch a regression.** ±0.008 is the
resolution for *ranking*; an arm returning 0.938 or below is a real signal that something broke.
That is what the pending readouts are for.

---

## The nine candidates, by the axis each one moves

| arm | ref | what it changes | evidence for that change |
|---|---|---|---|
| `0943` | 56568520 | — the published recipe | — (the control) |
| `0943f` | 56570144 | `_RAD_EXCLUDE` no longer excludes `Fracture` | ⭐ **measured, external, large n**: +0.0118 on that column, **+0.00099 macro**, 4/5 folds positive, n=4185 (C81) |
| `0943h` | 56568524 | per-label outer routing softened halfway | the public split is that map's tuning set; A111's iso says per-label transfers **worse** than one shared weight at n≈400 |
| `0943p` | 56599159 | outer routing **fully flattened** | the frontier's **own written statement** that this map is "the likeliest place to give back points privately" and that flat "is what the `parent` preset is for" (A122) — plus A111's direction |
| `0943o` | 56570148 | outer **default** 0.65 → 0.60 | as `0943h`, weaker dose, no independent support |
| `0943r` | 56599162 | `rad_cal_w` 0.40 → 0.00 (the frontier's v38) | the author's own revision, motivation unstated; our A117 basis for *rejecting* it was **retracted** as confounded (A127), and the trend we did measure — calibrator value falling as the base strengthens — leans that way |
| `0943s` | 56570151 | 94 → 62/42 windows | **measured neutral** (both 0.943) ⇒ a cost lever, not a score lever |
| `0943c` | 56599163 | ⭐ **+ our rescued `cn0` as a fifth Raptor view** | ⭐ **measured by us**: +0.0019 at a 0.917 base, **+0.0028** on a parent-quality base, CIs excluding zero, and **both halves chose weight 0.2 out of `public20`'s share** (A131/A137). Integration proven to run: five arms in the blend, 0 failures (A143) |
| `0943cf` | **56599932** | the member **plus** the fracture fix | the two changes this repository actually has measurements for, on different parts of the pipeline (the CoAt view set vs the Rad blend's label set), so at least plausibly additive |

**Four axes, two slots.** Rad exclusion set · outer routing sharpness · calibrator weight · our member.

---

## Recommendation: **`0943cf` + `0943p`**

**Why `0943cf` is the primary.** It is the only candidate carrying **two independently measured**
improvements, and one of them — `cn0` — is the only member this repo has ever measured **above the
frontier's own DINOv2 seed members** (0.8769 vs 0.8676–0.8697, A130). Its machinery is proven rather
than assumed: five arms in the blend, no failed studies, no neutral filling, and the Raptor stage
cost measured at +15% (A143).

⚠️ One honest dilution: inside the CoAt arm our view's share is 0.167, and that arm is itself
weighted ~0.65 against the stack, so the **effective weight is ≈ 0.11, not the 0.2 that was
measured**. The measured base was also weaker (0.917 vs ~0.94), and a stronger base wants a *smaller*
dose — so a dose below 0.2 is the coherent direction rather than a fudge.

**Why `0943p` is the second slot.** It is the frontier's own registered remedy for the one knob
family they describe, in their own source, as fitted to a public split that is a fraction of the test
set (A122) — and our own tuning-transfer curve points the same way (A111: below ~870 studies,
per-label tuning transfers *worse* than a single shared weight). `0943h` moves the same axis at a
weaker dose; `0943o` barely moves it at all.

**Why this pair.** `0943cf` keeps the published per-label routing; `0943p` flattens it. So the two
picks sit on **opposite sides of the one knob whose provenance is documented as public-fitted** —
the pair hedges that axis instead of doubling it — and between them they cover three of the four
axes (Rad exclusion, our member, routing). The fourth is left alone deliberately: the calibrator's
only local measurement was confounded (A127), and `0943r` is a bet whose direction we cannot check.

**Alternatives, documented rather than hidden:**
* `0943cf` + `0943r` — swaps the routing hedge for the calibrator bet; defensible, less supported.
* `0943cf` + `0943h` — the milder routing dose. `h` vs `p` rests on A122 versus C57, and A122 is a
  primary source while C57 was measured on the fitting set.
* `0943f` + `0943c` — if you would rather not stack two changes in one arm: same two axes, no stacking.

**If only one selection is allowed, or only one is scored: `0943cf`.**

---

## Pre-registered rules for the pending readouts

**Four** submissions are in flight (`0943p`, `0943r`, `0943c`, `0943cf` — refs `56599159`, `56599162`,
`56599163`, `56599932`). **They do not rank anything** — but they can disqualify, and that call is made
here rather than later:

| observation | action |
|---|---|
| any comes back **≤ 0.940** while `0943` sits at 0.943 | treat as a **regression signal** and drop that arm from the pair, even though the gap is nominally inside the noise — a 0.003+ move in the *wrong* direction is the one thing the board can show |
| `0943c`/`0943cf` comes back 0.941–0.945 | **no information**; proceed with the recommendation above |
| any **fails** (FAILED, or no `submission.csv`) | drop it — the hidden rerun is the only test that matters for a code competition |
| `0943c`/`0943cf` **completes** | ⭐ keep it as a candidate regardless of score: completing **1322 studies inside the 8-hour fail-closed budget** is the same de-risking A110 got from the five arms, and without it the arm is not selectable in good conscience |
| the frontier publishes something new | re-open the whole table; that is the only source of a genuinely new idea |

### Readouts so far, read against the table above

| ref | arm | result | reading |
|---|---|---|---|
| `56599163` | `0943c` (our `cn0` member) | **COMPLETE, 0.942** | falls in the **0.941–0.945 ⇒ no information** band, so it neither promotes nor demotes the arm — **and it completed**, which is the row that matters: our member survived the hidden rerun on 1322 studies with a fifth view in the blend, 0 failures. ⚠️ **Not evidence the member helps.** 0.942 vs `0943`'s 0.943 is 0.001 against ±0.008 noise, and two pure-knob arms (`0943h`, `0943o`) also sit at 0.942 — so that value is not a fingerprint of the member. Equally it is **not** a regression: the pre-registered line was ≤ 0.940. |
| `56599159` `56599162` `56599932` | `0943p` · `0943r` · `0943cf` | **PENDING** | — |

⚠️ The standing rule this repository paid for: **never spend a slot to "see if it helps"** (A96).
Each of the nine submitted arms was submitted either to confirm a change with prior evidence or to
create an option for this deadline — never to discover whether an idea works.

⚠️ **Selecting is a browser action.** `kaggle` 2.2.4 has **no `select` command** (verified), and the
auto-ticked default is not the recommended pair: Kaggle auto-selected `0943` + `0943f`, three of our
arms tie at 0.943, and the default took two of them rather than the two we would choose (A123). At the
deadline, **untick `RSNA knee 0943 - Version 2` and tick the two arms named above by their refs.**

---

## What is already banked (no action needed)

- **Five arms completed the hidden rerun** end to end on 1322 studies (A110); `0943c` now tests
  whether that holds **with a fifth view** (A143).
- **Ledger:** `state/lb_ground_truth.csv`, regenerated by `python scripts/lb_ledger.py`.
- **Quota:** GPU 10.33 h / 30 h, TPU 1.85 h / 20 h, refresh 2026-10-03. Selecting spends no quota.
- **The weights:** `ppxl16/rsna-knee-cn0-weights`, six checkpoints, private, ready — the only copy
  independent of the rented box, which is now shut down.
- ⚠️ **The live research line (`STATUS.md`) is not expected to feed this decision.** The A5-architecture
  transfer check and the `n_slice=8` density rung produce *single-fold surrogate* numbers on 881
  studies; converting one into a selectable submission would need a full notebook integration on top,
  and no arm in this repo has ever beaten the current stack by a margin the board could see. Treat the
  table above as the decision, and the research line as what it is: an attempt to find a *future* edge.
