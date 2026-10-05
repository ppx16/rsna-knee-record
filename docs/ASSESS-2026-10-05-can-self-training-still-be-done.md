# ASSESS 2026-10-05 — can self-training still be done? Yes, and the strongest form has never been run

**Short answer: yes.** Self-training is the only axis in this repository with a **positive** measurement
(`msft2` moved our rank 708 → 487, in-block 38.89 % → 13.66 %), and the specific run that produced it was
**fold 0 of a five-fold design**. The other four folds were never trained. That is not "another
same-kind member" — it is *the same member, finished*.

What follows separates what is still open from what is genuinely closed.

---

## 1. The fact that reframes the question

`kernels/kernel-tpu-msft/knee-mri-training-the-twelve-finding-model.py` prelude:

```
'--epochs', '4', ... '--k', '12', '--k_eval', '24',
'--seed', '42', '--tag', 'msft0', '--folds', '5', '--fold', '0',
```

**`--folds 5 --fold 0`.** The trainer is a five-fold design; we ran one fold and shipped its SWA
checkpoint as a 6th Raptor view (`56814081`). `kernel-msft2-cx1..4` are the **ConvNeXt** variant's
folds — not this recipe, and not pushed.

⚠️ And the trainer already writes per-fold OOF (`[cv] wrote OOF ... fold=... nfolds=...`), so a five-fold
run yields **a full 4407-study OOF for `msft2`**, where today the one-convention table carries
`msft2 = 0.8847` from fold 0's held-out slice alone.

**Cost:** ~2.6 h per fold × 5 ≈ **13 h GPU**, against **~24 h remaining** (refresh 2026-10-10).

## 2. Why this is not the thing the stop rule forbids

`STRATEGY-2026-10-04-how-far-can-finetuning-go.md` §5 stops *"同语料同配方的 CoAtNet 成员"* — **another
member of the same kind** — on the measured ~11× decay. A five-fold average is a different object:

* it is the **same** member, so it does not add a correlated sibling to the blend;
* fold-averaging **reduces variance** rather than adding a new hypothesis;
* and it **produces the full OOF we do not have**, which converts "msft2 helped" from a one-shot rank
  observation into a measurable, re-usable member.

⚠️ **The honest counter, stated plainly:** the 11× decay is real. Fold-averaging makes a member more
*reliable*, not more *informative*, and the blend may not care. The expected gain is small. What is
different from every other option is that its **prior is positive rather than unknown**.

## 3. What is genuinely closed (so this is the last self-training option of its kind)

| option | status |
|---|---|
| **Specialist** (`msft3-spec`, label-subset loss) | ⛔ **gated off.** Its prereg §0 made the meniscus result a hard entry gate; meniscus scored **0.940** vs a **0.943** baseline. Axis falsified. |
| **External / more data** | ⛔ **closed, proven.** The corpus is the competition train set **id for id** (`FIND-2026-10-05-corpus-is-the-competition-train-set.md`), and no public Kaggle-mountable knee MRI labels more than ACL + meniscus. |
| **Competition-specific Kaggle Models** | ⛔ **closed, measured.** The only evaluable one scores **0.7992** macro on 4407 studies (`RESULT-2026-10-05-models-axis-closed.md`); the rest are repackages or undocumented. |
| **A third same-kind CoAtNet** | ⛔ stop rule: expected in the noise. |
| **Our own ConvNeXt member** | ⛔ measured and failed — fold 0 gold-58 **0.9003** but 870-study OOF **0.8493**; two instruments disagreeing by 0.051 is a disqualification, not a borderline. |
| ⭐ **Five-fold `msft2`** | ✅ **open, and the only one with a positive prior.** |

## 4. The design question a five-fold run forces, before any GPU is spent

Today `msft2-ours` enters `ARMS` as **one view at w = 0.20** (14.3 % of the Raptor sub-blend). Five
checkpoints cannot each take 0.20 — that would quadruple the family's share and change two things at
once, which is how a result becomes uninterpretable. So the arm has to choose, and the choice must be
pre-registered:

* **(a) fold-mean → one view.** Average the five SWA checkpoints' *predictions* (rank-mean) and insert
  that single member in the existing slot at the existing w = 0.20. **Isolates "is fold-averaging
  better than fold 0".** Recommended: it changes one thing.
* **(b) five views at w = 0.20/5 each.** Keeps the family share fixed but makes five members where one
  was. Harder to read, and the sub-blend already rank-means its views, so it is close to (a) anyway.

⚠️ **Measured against what:** the arm must take a **before-snapshot and an after-snapshot** and be read
on the **in-block percentile**, with the **margin** reported alongside — because for a medal the bronze
line is absolute. Current: rank **551**, line **522**, **margin −29**, in-block **13.63 %**
(`2026-10-05T16:05Z`).

## 5. ⚠️ CORRECTION — this belongs on TPU, and the earlier "13 h GPU" was wrong

The user asked *"自训练不应该用tpu做吗"*. **They are right, and the repository already says so** —
`scripts/corpus_manifest.py`: *"The 2026-09 GPU-quota policy is: GPU only for submissions, experiments
on TPU"*, which is `FACTS` **A31** (the user's own rule from 2026-09-21).

Checked, not assumed:

```
kernel-msft2         tpu=True   gpu=False     <- the run that produced the winning SWA checkpoint
kernel-msft2-cx1..4  tpu=True   gpu=False
kernel-tpu-msft      tpu=False  gpu=True      <- the documented mistake: named "tpu", pushed as GPU,
                                                 ran on a T4 at 3.7 h/epoch
```

**⇒ `msft2` fold 0 already ran on TPU**, and the trainer is fully TPU-ported (XLA device selection,
single process rather than `xmp.spawn`, XLA-safe activation checkpointing, deps installed in-kernel).

⇒ **The five-fold estimate is ~13 h of TPU quota, not GPU.** And that distinction is operational, not
cosmetic: the 0943 submission kernels run on `NvidiaTeslaT4`, so spending GPU hours on training would
eat the budget the submissions need.

### But the binding constraint is not the 13 hours — it is concurrency

| | GPU | TPU |
|---|---|---|
| remaining | 24.40 h | **16.21 h** |
| concurrency cap | 2 sessions | **1 session**, and a **QUEUED** kernel already holds it (`A28`, `A136`) |
| per-session cap | 9 h (rules) | **≈ 9 h** (`A29`) |
| queueing | — | **can be very long** (measured once at ~26 h, `A33`) |

Three consequences that change the plan's shape:

1. **13 h of compute fits inside 16.21 h of quota — with ~3 h to spare.** Tight but feasible.
2. **13 h > the ~9 h session cap**, so a single kernel that loops all five folds is **impossible**. It
   must be **five separate pushes**, trained serially.
3. **TPU concurrency is 1**, so those five pushes cannot be queued together — each waits its turn.
   ⚠️ Queueing does not consume quota (`A33`), so this is a **wall-clock** cost, not a budget one. With a
   17-day window that is acceptable; if the queue behaves like the 26 h observed once, five serial waits
   is the real risk to the plan, not the 13 h.

⚠️ **Also revisit the per-fold estimate before committing**: "≈ 2.6 h per fold" is inherited from
`docs/PREREG-2026-10-04-convnext-member.md` §预算, i.e. it was sized for the **ConvNeXt** member. The
`msft2` log on disk is truncated to 3.5 KB and carries no per-epoch timing, so **the CoAtNet fold cost has
never been measured directly** — the first fold re-run should be treated as the measurement of it.

## 6. Recommendation

**Yes — run the five folds, on TPU, and pre-register option (a).** It is the last option in this
repository whose prior is positive rather than unknown, it leaves the GPU budget untouched for
submissions, and its by-product (a full 4407-study OOF) is worth having regardless of whether the arm
improves the board.

⚠️ **And state the ceiling honestly:** even a perfect five-fold `msft2` does not cross the structural
boundary the strategy document drew. It is worth **0.000x–0.001**, which the rank instrument can resolve
and the displayed score cannot. It is a *consolidation* move, not a breakthrough — and it should be
weighed against spending the same 13 TPU-hours on nothing at all, which is the actual alternative now
that every other axis is measured shut.
