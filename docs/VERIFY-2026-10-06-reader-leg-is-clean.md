# VERIFY 2026-10-06 — the pending reader-leg measurement is CLEAN

The reader leg (`56872230`) is PENDING. Before reading it, one question had to be settled: **if it fails to
improve the leaderboard, will that be the idea failing, or a wiring bug?** A wiring bug would be fixable and
would make the negative reading worthless. Answered by reading the reader's own code and checkpoint, not by
assuming.

---

## What was checked, and what it showed

**`preprocess.py` is self-describing and reads the competition layout directly.**

```
MM_PER_PX = 0.4    SIZE = 384    MAX_SLICES = 64      -> 153.6 mm field of view
slices come from {DATA}/{split}_series/{study}/{series}
orientation from ImageOrientationPatient, order from ImagePositionPatient,
spacing from PixelSpacing, per-series intensity window at the [1, 99.5] percentile
```

⇒ No external cache, no assumed directory shape beyond the competition's own, and none of the
geometry has to be guessed. **The `data_dir` I inferred for the submission cell is the right one**, and the
preview log confirms it was found: `reader leg: test data dir /kaggle/input/competitions/rsna-knee-abnormality-detection`.

**The checkpoint's own `args` agree with the notebook's description and with how it is run:**

```
{'name': 'cnxt_v0', 'fold': 0, 'backbone': 'convnext_tiny.fb_in22k_ft_in1k', 'res': 256,
 'k': 8, 'k_eval': 12, 'epochs': 14, 'ema': 0.998, 'seed': 0}
```

`infer.py` passes `res=res` from the checkpoint into the forward pass, and `preprocess.py` produces 384 px
volumes that are interpolated to 256 — **so the model runs at the resolution it was trained for**, not at the
384 px storage resolution. 219 tensors; head = `cls_w/cls_b/proj.*/slot_emb/pos_emb.*/tf.*`, i.e. the study
transformer plus per-finding attention pooling described in its model card.

**And the preview proved the whole path end-to-end** (`state/reader_arm_preview.json`):

```
reader leg: 3 checkpoints from /kaggle/input/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader
reader leg: wrote /kaggle/working/_own.csv          <- the reader produced output
reader leg: BLENDED 3 checkpoints at w=0.30 in 27s  <- alignment + asserts + rank blend all passed
reader leg FAILED  <- ABSENT, so no fallback was taken
```

⚠️ Note what the two `assert`s in the blend cell would have caught, and did not: `_o_own[_o_lab].notna().all().all()`
(NaN) and `(_o_own[_o_lab].nunique() > 1).all()` (a constant reader). Both passed, and `.loc[_o_stack.StudyInstanceUID]`
would have raised on any study mismatch. **So the reader's predictions were aligned, finite, non-constant, and
blended.**

## Conclusion

**The measurement is clean: no wiring, geometry, resolution or alignment bug.** When `56872230` scores, the
in-block percentile change measures *the idea* — adding an independently trained model family as its own 30 %
leg — and nothing else.

⚠️ `0.0000 of label cells moved` on the 3-study preview is **not** an exception to that: on three items
`rank(pct=True)` yields {1/3, 2/3, 1}, and re-ranking any three distinct values returns the same three
values, so at w = 0.30 the stack's order is preserved unless the reader reverses it hard enough to overcome
the weight. On 1322 studies there are 1322 distinct ranks and reordering is essentially certain.

---

# PRE-REGISTRATION (not fired): reader leg with TTA + rank ensemble — the next step IF the base leg works

**Entry gate: only if `56872230` moves the in-block percentile by ≥ 3 points** (the same bar as its own
prereg). If it does not, this is not fired and the composition is dropped.

**What would change, and why it is not public-overfitting.** The reader currently aggregates its three
folds as a **probability mean** (`torch.stack([...]).mean(0)` in `infer.py`) and has **no test-time
augmentation**. The community's Stack++ variant changes exactly two things:

1. **fold × TTA rank ensemble before blending** — rank-space aggregation "matching the competition's AUC
   objective", rather than a probability mean;
2. **inference slice centres aligned to the training-bin centres**, with an extra original-grid TTA pass.

⚠️ **TTA and rank aggregation change no fitted parameter** — they are variance reduction, not tuning
against the public split. That distinction is the reason this is admissible while a msft2 weight sweep
(deliberately NOT done, see `FIND-2026-10-06b` §4) is not.

**Why it is affordable:** the stack's hidden-test inference is **< 30 min** (`FIND-2026-10-06b` §2, two
independent sources), so the 9 h cap is largely unused and an extra TTA pass costs minutes, not hours.

⚠️ **What is unverified:** Stack++ reports **no leaderboard score** — its own text says *"this Stack++ variant
is an improvement experiment, not a claimed leaderboard score until it is submitted"*. So the prior for it is
weak, and it is only worth firing **after** the base leg has earned its place.

**Read rule if fired:** identical to the base leg — before/after snapshot, **in-block percentile** as the
primary criterion (≥ 3 points), **margin** reported alongside, displayed score explicitly not the criterion.
