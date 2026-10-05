# FIND 2026-10-05 — external DATA is closed; the one "foundation weights" artefact contains no weights; and there IS an unexamined Kaggle **Models** axis

Three questions were left open by the external-data sweep because `web_fetch` is denied for non-public
IPs in this environment (every host resolves to `198.18.0.40`). **The Kaggle API is not affected by that
restriction and has worked all session, so all three were settled directly.** Two answers are negative
and the third opened a lead nobody had looked at.

---

## 1. External knee-MRI **data** with our twelve labels: CLOSED

The sweep's verdict — *"there is no publicly-available, Kaggle-mountable knee MRI corpus that labels more
than ACL and meniscus"* — is consistent with everything checkable here, and the structural reason is
worse than "we did not find one":

* MRNet: ACL + meniscus + generic "abnormal" — **2 of 12**, and per-study.
* OAI/MOAKS: genuinely relevant OA labels, but NDA registration + a DUA that **prohibits redistribution**,
  so it cannot be published as a Kaggle Dataset and therefore cannot be mounted with internet off.
* CoPAS (Nature Comms 2024) is the **exact twelve-finding task**, and its data is *"available upon
  request"* — unusable for the same reason.
* Everything else is X-ray (modality mismatch), unlabelled, or n=1.

⚠️ **And the label gap lands on exactly the wrong targets.** Public knee MRI is ACL + meniscus; the
twelve-finding task's unusual labels are the OA compartments, Effusion, Synovitis, Baker's, Contusion and
Fracture — **the ones no public corpus labels**, and (by the base's own log, `FACTS` and the 58-study
audit) among the ones where our arms actually disagree. Adding ACL/meniscus data would sharpen the two
findings we are already best at, and macro-averaging dilutes that.

⇒ **`docs/FIND-2026-10-05-corpus-is-the-competition-train-set.md` + this = the "more data" half of the
strategy document's structural boundary is closed, not just un-found.**

## 2. ⚠️ The artefact the sweep named its STRONGEST recommendation does not contain what its title says

The sweep's top pick was `prvsiyan/orthodiffusion-official-knee-mri-foundation`, titling it *"OrthoDiffusion
Knee MRI Foundation **Weights**"*. Its full file listing:

```
OrthoDiffusion-main/.../LICENSE          OrthoDiffusion-main/.../train.py
OrthoDiffusion-main/.../README.md        OrthoDiffusion-main/.../unet.py
OrthoDiffusion-main/.../config_finetune.yaml     finetune_classifier.py
OrthoDiffusion-main/.../config_linear.yaml       finetune_segmentation.py
OrthoDiffusion-main/.../config_segmentation.yaml linear_classifier.py
... (20 files, all source code, configs, LICENSE and README)
```

**Zero `.pt` / `.safetensors` / `.ckpt` files.** The dataset is the *code*, not the weights, despite the
title. ⇒ **That recommendation collapses.** It was the single most emphatic item in the sweep's verdict
and it rested on a title.

⚠️ **This is the FOURTH instance of the same disease in this session** — reading a declaration instead of
the artefact: `CROP_MM`'s winning assignment, the stale `[raptor-fast]` print, `F_dim=768`'s dead
default, and now a dataset whose title says "Weights" and whose contents are source code. **A title is
not a listing. Check the files.**

## 3. ✅ And the same probe found a genuinely unexamined axis: competitor **Kaggle Models**

The sibling datasets are confirmed weights-only, so no hidden data hides behind their names
(`raptor-knee-arms` → 3 `.pt`; `arms-x` → 1 `.pt`; `widedense` → 2 `.pt`; **`fullspan`'s title says
"Corpus" and it contains one `.pt`**). But while confirming that, `api.model_list(search="knee")` returned
something we have never touched:

| Model | instance slug | reads as |
|---|---|---|
| `fabotelli/pillar0-knee-finetune` | `atlas-336-2ep` | a medical foundation model fine-tuned for this task |
| `beraterolelk/rsna-knee-25d-attention-backbone` | `default` | a 2.5D slice-attention backbone |
| `yaolinghan/rsna-knee-pool-models` | `convnextv2-pool` | **ConvNeXtV2** — a different architecture family |
| `ruicompany/rsna-knee-a5-rad` | `pytorch` | A5 + RadImageNet lineage |
| `yunusgmsoy/rsna-knee-radhead-resnet50` | `5-fold-yunus-v5` | ResNet-50 RadImageNet head, 5 folds |
| `prvsiyan/rsna-knee-pulse-v6-checkpoints-public-20260813` | *(dataset)* | 15 ResNet-18 fold checkpoints, train-only |

**Our kernels already support this mechanism and mount exactly one model:**
`model_sources: ['metaresearch/dinov2/PyTorch/small/1']` — a *generic* DINOv2. **We mount zero
competition-specific models.** All 16 of our dataset sources are Datasets.

⇒ **This is the one lever that is simultaneously (a) permitted — the rules say *"Freely & publicly
available external data is allowed, including pre-trained models"*; (b) mountable with internet off,
which is exactly why the data route is closed; and (c) structurally different — a ConvNeXtV2 or a
Pillar-0/Atlas encoder is not a variant of anything in our stack.** It is what the strategy document
called *"somebody else's model"* when it named the way past the ceiling.

⚠️ **It is NOT a licence to expect a gain.** `C14`/`C16` measured *"view / architecture diversity ≈
+0.001"*, and `D14` closed view-adding after two flat attempts. What those did **not** test is a
*different family from a different author*, which is what these are. The honest prior is small and the
test is cheap — **one arm, one submission** — so it is worth exactly one shot, not a campaign.

## 4. What is now claimed, and what is not

**Claimed:** external labelled data for the twelve findings is not publicly mountable; the specific
foundation-weights artefact is mis-titled code; and competition-specific Kaggle Models exist, are
already supported by our kernel format, and are entirely unmounted.

**Not claimed:** that mounting them will help. Nothing here has been run, and the one measured
expectation we have for "more diversity" is ≈ +0.001, which the rank instrument can resolve but which is
inside the displayed score's noise.
