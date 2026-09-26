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

**Tick two rows on the competition's Submissions page: `56570144` (`0943f`) + `56568524` (`0943h`).**
Full reasoning and the fallbacks: [`docs/FINAL-SELECTION.md`](docs/FINAL-SELECTION.md).

⚠️ `kaggle` 2.2.4 has **no select command** (verified) ⇒ this needs the browser. And **doing nothing
is not neutral**: Kaggle then auto-selects **by public score**, while three of our arms are tied at
0.943, so the resulting pair is not the one we chose. Deadline 2026-10-22 23:59Z.

## Live right now

| Thing | State |
|---|---|
| Submissions | **none pending** — all five read back, ledger is 32 rows |
| Background jobs | **none running** (the score watcher `pwsh-2` exited) |
| Accelerators | **idle, deliberately** — every reachable lever is closed below |

## Closed — do not reopen without new evidence

| Line | Why it is closed |
|---|---|
| **Training line** | Our best arm scores **0.8731** against A5's **0.9143** on the 181 studies where both are genuinely held out — a gap whose CI excludes zero (A106). Below the archive's own ~0.90 member bar. |
| **Champion folds** | The checkpoints carry a config, but in a **different schema**, and **neither gate can see a wrong geometry** (A108). Forcing it would produce a confident, wrong result. |
| **E2 parent OOF** | The 20 published members are **not held out** (A93): parent 0.9295 vs a genuine OOF's 0.8323 on the same studies. |
| **Fusion knobs on the frontier stack** | No valid parent OOF exists, so they could only be judged on the ±0.008 board. `rad_cal_w`→0 was then **locally contradicted** (A117). |
| **Image+metadata arm** | Measured honest gain on top of our own ensemble: **−0.0033** (A112). Plan marked NOT RECOMMENDED. |
| Dense corpus · member additions · capacity scaling · full retrain · gold-58 for small deltas · CoAtNet on XLA · multi-TPU queues · rental box as a scoring route | Each closed by measurement, not assumption. |
| Scheduled tasks | Deleted (§164) — they were idle and were the black console window the user saw. |

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
