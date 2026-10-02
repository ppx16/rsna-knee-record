# RSNA Knee Abnormality Detection — STATUS (2026-10-02 08:05Z)

**One screen. Rewritten from scratch; the previous version was dated 2026-09-27 and had drifted.**
Deep detail lives in `FACTS.md` (A-numbers) and `docs/`; the recent session records are
`docs/RESULT-2026-10-02-msft2-finetune.md`, `docs/RESULT-2026-10-01-exp085-gate.md`,
`docs/FIND-2026-10-01-exp085.md`, `docs/FIND-2026-10-01-rsna-sweep.md`.

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

⚠️ **Checkpoints are still KERNEL OUTPUT, not a dataset we own** (`raptor_ft_msft0.pt`,
`raptor_ft_msft0_swa.pt`, 293 MB each). **Publish them under `ppxl16` before anything depends on them** —
the exp085 dataset was withdrawn mid-session and that is exactly how a working line dies.

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

1. ✅ **DONE — `ppxl16/rsna-knee-msft2-ckpt` published** (private, 8 files, `EXIT=0`). The exp085
   failure mode — depending on a dataset we do not control — is now closed for this asset.
   `publish_msft2_dataset.py` / `work/LOG-2026-10-02-dataset-and-cx.md`.
2. 🔄 **IN FLIGHT — `ppxl16/rsna-knee-msft2-cx` is QUEUED on TPU** (fine-tunes ConvNeXt-small
   `v4cx0` for 4 epochs at res 384, tag `cx0`). ⚠️ **Its log prints a zero-shot gold58 macro-AUC
   before training; if that is near chance (< ~0.75) the checkpoint did not load and the run must be
   killed, not left to burn the window.** `head tensors carried: 0` is expected and fine.
   Fallback if killed: `msft2-r2`, i.e. continue from our own `raptor_ft_msft0_swa.pt` (hist is
   monotone and unsaturated ⇒ ~0.910+ plausible).
3. ⭐ **BUILD THE ARM — cannot start yet, and correctly so.** Host is
   `kernels/kernel-0943s-2coat` (`rsna-knee-restructured-version-3.ipynb`), which is
   `enable_gpu: true` / `NvidiaTeslaT4`. **GPU is 0.00 h until 2026-10-03**, so writing the msft2
   blend now means writing **untestable** code into a 301 KB notebook. Start this on 10-03.
4. ⭐ **2026-10-03, when the GPU quota returns**: push the arm and submit, then read the board.
   ~19 days left.
5. **The ConvNeXt TPU run is the only productive compute available this round** — which is exactly
   why it, and not the arm, was started.
