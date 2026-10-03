# HANDOFF 2026-10-04 — this session, condensed

Scope: one autonomous session on **RSNA Knee Abnormality Detection** (deadline **2026-10-22 23:59 UTC**).
Written so a fresh session can act without re-deriving anything. **Read this first, then
`STATUS.md`, then `state/EXPERIMENTS.md` §D.**

---

## 1. Where we are, in numbers

* **Our score 0.943.** Public board ~4829 entries; the **medal line is 0.943** and it has been **flat
  there since 2026-09-25** while the field grows.
* ⭐ **794 teams sit in that single 0.001-wide bin** (16.4 % of the board at the 2026-10-02 scrape), and
  the bronze line cuts through them at ~rank 482. **The medal is decided INSIDE a dense band.**
* The public frontier's own recipe is what we run: our `kernel-0943` is a **byte-identical** fork of
  the frontier notebook (**0 of 51 cells differ**).
* Public board **SE ≈ 0.008**; nine 0943-family arms spanned only 0.941→0.943.

## 2. The single most important correction this session made

I had written, and acted on, the claim that **a change below the public board's resolution is
"pointless"**. **That is right about *measurement* and wrong about *ranking*.** With 794 teams in one
0.001-wide bin, our rank is `1 + #{teams whose private score exceeds ours}`, so a genuine private gain
of δ removes a densely-populated interval from that count. **A small, sign-stable improvement has real
expected rank value even when the board cannot confirm it.**

⇒ This retroactively **strengthens** the MCL arm (`56791368`, +0.000904 macro, **negative in 0/35**
source×assembly combinations). It does **not** revive EffB0 — that is closed for operational reasons
(see §5), not because its effect was small.

## 3. In flight — check these two refs first

| what | ref | state |
|---|---|---|
| **MCL arm submission** | `56791368` | **PENDING**, created 2026-10-03 06:32Z, no error flags |
| EffB0 kernel v3 | `ppxl16/rsna-knee-0943cfm2eff` v3 | COMPLETE — **not submitted, and must not be** (duplicate) |

**MCL decision rule, pre-registered**: ≥ 0.942 ⇒ legitimate third candidate for the primary slot;
**≤ 0.940 ⇒ treat as a real regression and drop it** (the board can catch a regression even though it
cannot rank 0.941–0.945).

## 4. ⛔ What only the user can do — the final two slots (browser action)

Verified three ways this session: the `kaggle` CLI has **no `select`** verb (2.1.2 and the 2.2.2 wheel
both inspected); `ApiSubmission` exposes **no `selected` field**, so the current tick state **cannot
even be read**; legacy `www.kaggle.com/api/v1` returns **401** to API-key auth.

⚠️ **Four arms display 0.943 and three of them are NOT the recommendation — do not pick by score.**
Full verified ref→description table: **`docs/FINAL-SELECTION-2026-10-04.md`**. Summary:

* **TICK** the row beginning `0943cf v1 - our cn0 member (5th view) PLUS the fracture fix` (`56599932`)
* **TICK one of**: `0943p parent routing v1 - flat outer 0.65` (`56599159`, 0.941) **or**
  `0943h halfway routing v2` (`56568524`, 0.942)
* **UNTICK the default**: the auto-ticked row begins `0943 speedy k94 v2 …` (`56568520`)

## 5. What is CLOSED, and why — do not re-open without reading these

| line | closed because |
|---|---|
| **EffB0-MIL leg** | three runs (v1 spelling-mapped slots, v2 an always-False pin term, v3 ran and the **rebuild produced `0 slot-series` for all four TRAIN studies, so its own gate rejects regardless of budget**); **and its headline rationale was FALSIFIED** — "helps most where we are weakest" is false (weakest-3 +0.00414 vs strongest-3 +0.00544; Fracture +0.0013; Baker's negative). `docs/RESULT-2026-10-04-effb0-v3.md` |
| **msft2 fine-tune** | the artifact itself is real and is the **first of ours to clear the A167 solo gate (SWA gold58 0.9069)** — published as `ppxl16/rsna-knee-msft2-ckpt`. But putting our raptor member into the ensemble is **ledger D7, measured flat**. Closure is on ensemble value, **not** on the artifact's quality. |
| **`AMP_PREF='bf16'`** | byte-identical in the frontier source that reaches 0.943, so changing it would deviate from a validated config on a foreign anecdote. Closed **on provenance**. |
| **ConvNeXt / TPU-trained coatnet** | ledger D5/D17 territory; the TPU run was deleted from the queue. ⚠️ **D17 is itself stale** — msft2 v4 proved CoAtNet trains fine on `torch_xla` with the activation-checkpoint shim. |
| **`rad_alpha`** | never submitted (`v2src-radalpha35/65` are "built but not pushed"), so `EXPERIMENTS.md` **D1's "all pre-registered axes" is an over-claim**. Still closed by *category*; `0.50→0.55` had LB backing. Do not "discover" it as a fresh lever. |
| **new public legs** | three preprocessing datasets checked this session — all caches, **no labels, no predictions**. The only prediction-publishing artifact found (`codeandodev/rsna-knee-run-base`) is a 500-study toy at MACRO 0.561/0.733. `docs/FIND-2026-10-04-preprocessing-datasets.md` |

## 6. Tooling built this session (reusable, do not rewrite)

* **`scripts/verify_effb0_arm.py`** — the arm acceptance test. Decodes the **UTF-16** log that
  `kaggle kernels logs … *> file` writes under PS 5.1 (that trap made a *successful* run look empty).
  **Decides: log arbitrates, receipts are the primary signal, the stage hash is only corroboration.**
  ⚠️ It had **three silent-wrong-answer bugs**, all caught by one test — **run it against a kernel
  compared with itself**; it must say INCONCLUSIVE.
* **`scripts/harden_effb0_budget.py`** — built and compile-verified, **never pushed**. If the EffB0 line
  is ever revived, push it with v3's pin.
* **Honest-dose method**: `work/eff_min.py`, `eff_dose.py`, `eff_dose_parents.py`, `eff_convention.py` —
  how to price a member on an **outer** split instead of in-sample. This is the transferable result.
* **`scripts/build_mcl_arm.py`**, **`scripts/build_0943_arms.py`**, **`scripts/lb_ledger.py`**.

## 7. Traps paid for this session — all of them mine

1. **Soft labels vs a binary metric.** `train_y.npy` is a soft label (7 values, mean ≈0.35). A metric
   comparing `yj==1`/`yj==0` matches **nothing**, silently returns 0.5 everywhere, and printed a
   confident fictitious "gain +0.00000".
2. **A swallowed exception.** `sklearn.roc_auc_score` raises `continuous format is not supported` for
   soft `y_true`; my `except ValueError: return 0.5` reproduced the same all-0.5 output.
3. **An AUC above 1** (1.00001) from using a global histogram where a running one was needed.
4. **A `while` loop that could not advance** on NaN, hanging a script for 7+ minutes.
5. **`None` (a failed download) compared as a value** → "LEG LIVE" for a kernel compared to itself.
6. **Storing a `Path` instead of its hash** → every file read as "DIFFERS".
7. **Counting goal rounds instead of reading timestamps** — I twice said a job had "been running for
   N rounds"; the actual elapsed time was **0.08 h**.
8. **Nearly publishing a statistic with a broken denominator** — a manifest with **one series per
   study** cannot support corpus-level percentages.
9. **Editing a file I had also delegated to an agent**, creating a race that left the local notebook
   disagreeing with the running kernel. Resolved by `kaggle kernels pull`, **not by assumption**.
   **Rule: never delegate an edit and then make it yourself.**

## 8. The honest strategic read

**There is no known lever that raises the AUC leaderboard score.** The public recipe is exhausted, the
frontier is frozen (last run 2026-09-14), the medal line is flat at exactly our score, and every
conventional axis is closed by measurement in `EXPERIMENTS.md` §D.

What remains is therefore: **land the MCL score, and make the tick decision well.** The tick is a
browser action; the recommendation and its rationale are in `docs/FINAL-SELECTION-2026-10-04.md`.

⚠️ **Also recorded but explicitly not recommended**: this competition has an independent **$18,000 /
3-place efficiency track** that the *same* submission feeds, and `0943s` (`56570151`) scored **0.943**
at a shorter runtime. The user has said the efficiency board is not a goal — so it is noted, not pushed.
