# STATUS — 2026-09-26 05:45Z

**One screen. Rewritten every session.** If this grows past a screen, the overflow belongs in a
`HANDOFF.md` section, not here.

## Where we stand

| | |
|---|---|
| Deadline | **2026-10-22 23:59 UTC** (~4 weeks) |
| Best ledger score | **0.941** (silver). Public ceiling seen: **0.943** (the recipe we now replicate) |
| Public LB noise | ±0.008 — it cannot resolve the 0.001–0.003 effects we chase |
| GPU quota | **1.58 h / 30 h**, refreshes 2026-10-03. TPU 0 h / 20 h |
| Cost model | **1 h wall clock = 1 GPU hour even on the 2×T4 shape** (measured 05:43Z: +0.33 h over 20 min) ⇒ §158.5 answered: N hours is billed as N, not 2N. Scoring reruns are **not** billed at all. |

## Live right now

| Thing | State |
|---|---|
| `ppxl16/rsna-knee-a5oof` — A5 OOF over all 4407 training studies | **COMPLETE and verified.** Shape `(5, 4407, 12)`, 4407/4407 applicable, no degenerate fold slices. |
| A5 leakage check (pre-registered, A97) | **PASSED.** Fold discipline is real: the held-out model is worse than its siblings in **5 of 5 folds**, mean **−0.0278**, against the leaky parent's **+0.0022** (A104). |
| 5 submissions, all `PENDING` | `56568520` 0943 speedy · `56568524` 0943h halfway · `56570144` 0943f fracture · `56570148` 0943o outer060 · `56570151` 0943s sparse |
| Watcher | `pwsh-2` (scores) |

⚠️ **The five readouts cannot rank the arms.** Their expected differences sit below the leaderboard's
noise. They are useful for two things only: confirming the 0.943 recipe reproduces, and catching a
gross failure. Today's quota of 5 submissions is spent; it refreshes at 00:00Z.

## Next actions, in order

1. ✅ **Done — our best arm measured against A5 on a genuinely common held-out set (A106):** 0.8731
   against 0.9143, a gap of −0.0412 with a CI that excludes zero. That is below the archive's own
   ~0.90 member threshold, so **the training line is closed by measurement**, not assumption.
2. 🔴 **Closed — the champion-folds route (A107, A108).** The checkpoints do carry a config and a
   fingerprint, but in a **different schema** (`lat_from_geometry`, `pad_short_fov`, `n_group`,
   `layer_scale1/2`) that this pipeline does not implement, and `slices`/`band` are genuinely absent.
   Forcing it would read their weights against the wrong geometry, and **neither gate we have can see
   that**: `check_fingerprint` depends only on architecture and `img`, and the fold test validates the
   weights' fold property. Wrong geometry would pass both and look reasonable.
3. ⏳ **Read the five scores** (watcher `pwsh-2`). They cannot rank the arms; their job is to confirm
   the 0.943 recipe reproduces and to catch a gross failure.
4. **Final two slots** from the ledger (`scripts/lb_ledger.py`), once the scores land.
5. **Fold 4 caveat:** its held-out gap is ~0 while the other four are −0.026 to −0.049. Report results
   with and without it, or use folds 0–3 only.

## Where the remaining weeks should go

Every lever we can reach is now closed by measurement — **except one, and it now has a written,
pre-registered plan with kill conditions: [`PLAN-arm-image-metadata.md`](PLAN-arm-image-metadata.md).**

The reason it is the only survivor: acquisition metadata was **measured** to carry ordering
information the image arms do not have (+0.008 on a weak arm, +0.002 on a strong one, weights stable
across folds — A109), and **every arm in the four-stage pipeline is image-only**, so nothing uses that
channel. The plan's decisive test is cheap (~5.6 GPU-hours on a rented 24 GB card) and its control leg
— images plus **permuted** metadata — is what separates "the metadata's information helps" from
"extra capacity helped".

It is still likely to fail: the measured gain shrinks as the arm strengthens, extrapolating to
~0.000–0.001 at the ensemble's level, and the required bar is ≥0.90 solo with error correlation below
the **0.8061** that A5 and Rad already share.

What comes first is not this plan but the discipline of banking what exists: read the five scores
back, then choose the final two submissions from the ledger rather than from memory. The plan waits
until both are done.

## Closed — do not reopen without new evidence

| Line | Why it is closed |
|---|---|
| **E2 parent OOF** | The 20 published members are **not held out**: on identical studies the parent scored 0.9295 against a genuine OOF's 0.8323, a gap the frontier's own gold-58 numbers say cannot be real (A93). Their `fold` field does not demarcate a held-out split either. |
| **Fusion knobs on the frontier stack** (D11–D13) | No valid parent OOF exists, so they can only be judged on the ±0.008 leaderboard. |
| **`rad_cal_w` locally** | Needs the 88-feature calibrator plus three arms' ranks and protocol features; none published. |
| Dense corpus (C85/C86) · member additions (§159.6) · capacity scaling (D15) · full retrain (D19) · gold-58 for small deltas (D16) · CoAtNet on XLA (D17) · multi-TPU queues (D18) · rental box as a scoring route (A84) | Each closed by measurement, not assumption. |
| Scheduled tasks | Deleted (§164). They were idle and were the black console window the user saw. |

## Known truth about our own instruments

- Our OOF fold discipline is **clean** (held-out fold removed from training; gold-58 never trained on).
- **Epoch selection uses gold-58** (`--topk 3`), so our gold-58 numbers are optimistic and the OOF is
  computed at gold-selected weights. **The whole published lineage shares this** (A98).
- **Report-group leak: 2 of 49 groups, 4 studies** (A100). **Scanner/site: measured, and it is a
  covariate, not a leak** — only 46 scanner models cover 4407 studies and the hidden set draws from
  the same pool (A101).
- Consequence to respect: **we cannot claim generalisation to new sites**; the data cannot express
  that claim either way.
