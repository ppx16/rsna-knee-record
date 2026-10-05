# FIND 2026-10-05 — the corpus IS the competition train set, so "more data" means external patients only

**Decisive, and it narrows the last open axis before any engineering is spent on it.**

---

## 1. What was assumed, and what is now proven

The strategy document's structural boundary (`docs/STRATEGY-2026-10-04-how-far-can-finetuning-go.md`
§4) says the only way past `0.944` is *"more data / external data — not more fine-tuning"*. That
sentence is ambiguous in a way that matters: **"more data" could have meant data we already have access
to and are not using.** If the competition's own train split contained studies the corpus omitted,
those would be the cheapest possible new axis.

Both counts read **4407**, which is suggestive and is not evidence. So the ID sets were compared
(`work/corpus_vs_competition_ids.py`):

```
competition train.csv      : 4407 rows, 4407 unique
corpus all_ids.npy         : 3200
corpus extra_ids.npy       : 1207
corpus total (union)       : 4407
corpus internal overlap    : 0
competition NOT in corpus  : 0
corpus NOT in competition  : 0
identical sets             : True
```

⇒ **`dreaddevelopment/knee-raptor-corpus` (+ `-ext`) is a preprocessed rendering of this competition's
own training set, study for study.** Not a subset, not a superset, not overlapping-but-different.

## 2. Consequences — three of them, and the third is the uncomfortable one

1. **There is no unused competition data.** Every training study the competition offers is already in
   the pipeline at 44 slices × 336 px. The 0.943 ceiling is not a data-quantity ceiling *within* the
   competition.

2. **"More data" is now unambiguous: it can only mean studies from a different patient population.**
   Which means a new source, a new download, a new preprocessing path, and — because submission kernels
   have **internet disabled** — that source must first be published as a Kaggle Dataset.

3. ⚠️ **And the labels are the hard part, not the images.** The twelve findings are supervised here by
   `flight0234/rsna-knee-hybrid-report-labels` — labels read out of free-text radiology reports by an
   LLM. An external dataset arrives with whatever labels *its* authors chose; the twelve findings are
   specific enough (six of them are OA-compartment / Effusion / Synovitis / Baker's / Contusion /
   Fracture) that most public knee-MRI corpora will overlap on **one or two** of them at best.
   **Images without matching labels cannot supervise the metric** — they can only serve as
   *pretraining*, which is a weaker and much less certain use of the same GPU hours that have already
   measured an 11× decay on in-domain fine-tuning.

⚠️ **Also worth stating plainly, because it bounds the whole idea:** all 16 datasets our best arm
mounts are **weights, not images** (§ `kernels/kernel-0943cfmsft2flat/kernel-metadata.json`). The
pipeline has never consumed a second image corpus. Adding one is not a variation of anything in this
repository — it is a new branch of the project, with the pipeline-engineering cost that implies.

## 3. What this does and does not decide

**It does not decide that the external-data axis is worth doing.** It removes the possibility that the
axis was already half-solved for free, which is the thing that had to be checked before spending on it.

Verdict pending the external-data availability report; if the label overlap is as thin as §2.3
suggests, the honest conclusion may be that **the remaining 17 days and 3 submissions are better spent
locking a safe final pair than opening a new corpus.**
