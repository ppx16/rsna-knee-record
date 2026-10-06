# STRATEGY 2026-10-06 — a fresh sweep found two new levers, one of them externally verified

**Trigger:** user asked for a full re-look on 2026-10-06. The last exhaustive public sweep was
2026-10-04, so the public surface was re-scanned before proposing anything. The sweep found new work
posted **today**, including a notebook titled *"…LB 0.944"* with 106 votes.

---

## Lever 1 — ⭐ a public, independently-trained model family that is VERIFIED to add +0.001

`goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944` (2026-10-06 01:29Z, 106 votes) is **the
community 0.943 stack, unchanged, plus one independently trained reader, rank-blended at 30 %.**

Its own reported evidence:

| | public LB |
|---|---|
| community stack alone | 0.943 |
| **their reader alone (5 folds)** | **0.929–0.930** |
| **stack 70 % + reader 30 %** | **0.944** |

and its own weight sweep:

```
(0.00, 0.943 stack alone)  (0.15, 0.944)  (0.30, 0.944, two runs)
(0.30, 0.943, one run)     (0.45, 0.942)
```

**The mechanism is exactly the shape we have never used** (cell 29 of their notebook):

* **one global weight** `OWN_W = 0.30`, chosen *before* submitting, not tuned on the leaderboard;
* the stack's own submission is **copied aside as a safe fallback** first;
* their reader runs as a **separate subprocess**, writes `_own.csv`;
* the two are **rank-blended**;
* **if the reader fails for any reason, the stack's submission is kept unchanged.**

⚠️ **This is the difference that matters.** Our `cn0` and `msft2` members are added *inside* the Raptor
sub-blend — a small share of one family. Theirs is a **separate 30 % leg** of the final rank blend, from
a **different architecture family** (2.5D ConvNeXt-Tiny + a 2-layer study transformer, 12 attention
pools, 256 px, their own 0.4 mm / 154 mm canonical preprocessing). We have never tested that
*composition*, and the public evidence says it is worth +0.001.

**And the artifacts are fully public** — `goodpjw2008/rsna-knee-2-5d-convnext-reader`:

```
cnxt_v0_fold0.pt  cnxt_v0_fold1.pt  cnxt_v0_fold2.pt     <- 3 fold checkpoints
infer.py  knee.py  preprocess.py                        <- inference
train.py  make_labels.py  oof.py                        <- TRAINING code
pylibjpeg wheels
```

⚠️ **Runtime is the real constraint, and it is worse than we have been assuming.** Their notebook
states the stack needs **6.5–8 h of the 9 h allowed** on the hidden test set, and the reader adds ~20 min
on top. That also explains why our own submissions sit PENDING for hours: the local 3-study preview runs
in 8 minutes, but the **server-side re-run is hours**. Any arm we build must leave room.

⚠️ **And the gain is one tick, with their own reproducibility at 0.944 / 0.944 / 0.943.** A fork landing
on 0.943 is normal by their own account.

## Lever 2 — ⚠️ TESTED AND **DID NOT REPRODUCE** on our data (recorded as a negative result)

Their panel 3, against the 58 official labels: report labels **0.896**, their model **0.912**, a 50/50 mix
**0.921**. The test was run on our data before spending anything (zero GPU).

**⚠️ But the first attempt returned AUC = 1.0000 for `report_labels_v5.csv`, and that was not a result —
it was a tautology.** Elementwise comparison (`work/check_gold_leakage.py`) found:

```
exact equality, elementwise : True  (696/696 cells)
binarised agreement         : 696/696 cells
per-label positive counts   : identical on all twelve
```

⇒ **`report_labels_v5.csv` — the table every member in this repository trains on — contains the 58
official gold labels VERBATIM on those 58 rows.** It therefore can never be used to measure
report-label quality; it IS the gold there. Any future comparison must use a gold-clean table.

⚠️ **This is not a training leak, and that was checked too**: the trainer holds the 58 out separately
(`train 3479 | OOF held-out 870 | gold-val 58`, and 3479 + 870 + 58 = 4407), so they never enter training
and `gold-58` remains a clean holdout. **The 0.9003-vs-0.8493 gap between gold-58 and 870-OOF is
therefore a genuine instrument difference, not leakage** — which we had been asserting and can now
support.

Rerun against clean tables:

| label table | = gold(58)? | macro AUC vs gold |
|---|---|---|
| `report_labels_v5.csv` | **YES** | 1.0000 *(tautology)* |
| `report_labels_v4hybrid.csv` | no | **0.8991** |
| `outputs/report_labels_consensus.csv` | no | 0.8860 |
| `outputs/report_pseudo_labels.csv` | no | 0.8617 |
| `report_labels_gpt56sol.csv` | no | 0.8352 |
| **a trained member (msft0)** | — | **0.8968** |

⇒ **The premise does not reproduce here.** Their labels were worse than their model (0.896 vs 0.912);
ours are marginally *better* (0.8991 vs 0.8968). Label refinement is a lever for their pipeline, not
evidently for ours. **Dropped, and recorded rather than quietly forgotten.**

⚠️ Caveat kept honest: `v4hybrid` is not the table we actually train on (`v5` is, and it is a 4-source
merge that is uncontaminated on the other 4349 rows), so this is a close comparison rather than an exact
one. It is however decisive against the *large* gap their panel shows.

## Lever 1 — ⭐ the actionable one, and it needs no new training at all

Everything in the Lever 1 section above stands unchanged, and the label investigation **strengthens**
rather than weakens it: **their reader scores 0.912 on the 58 gold as a single model family, against our
member's 0.8968** — an independent architecture that is at least as good on the one set of official
labels we have, and is published with its full training code.

⚠️ For the record, the label hypothesis *they* built on — that report labels are worse than a model, and
retraining on a report/model mix fixes it — was **tested on our data and did not reproduce** (see the
Lever 2 section above). What survives is the simpler claim: the reader is an independently trained model
family, published with code, that their own weight sweep says is worth **+0.001** blended at 15–30 %.

## What this does NOT change

The structural boundary in `STRATEGY-2026-10-04-how-far-can-finetuning-go.md` §4 still holds: neither
lever reaches 0.95. Lever 1 is a **+0.001 composition**; lever 2 is a **label-quality** improvement whose
size is unknown but bounded by how noisy the labels actually are. Both are worth more than anything else
on the table, because everything else is measured shut (`STATUS.md` §13) and these two are not.

## Proposed order — cheapest evidence first

1. **Zero GPU: test Lever 2's premise on our own data.** Do our members' OOF predictions match the 58
   gold better than `report_labels_v5` does? One script, no accelerator. **If yes, Lever 2 is real; if
   no, it is a property of *their* pipeline and we drop it.**
2. **Then Lever 1, which needs no new training at all**: mount the public reader, blend at 0.30 with the
   stack's submission as fallback, snapshot before/after, read the **in-block percentile** and the
   **margin**. It is one arm and one submission, and it is the only change on the table with a
   *published* +0.001 behind it.
3. **Only if both land**, consider retraining the reader ourselves on mixed labels and/or more folds —
   the training code is public, so this is a real option rather than a guess.
