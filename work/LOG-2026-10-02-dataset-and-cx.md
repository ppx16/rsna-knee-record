# RUN LOG — 2026-10-02 (round: dataset publish + ConvNeXt TPU arm)

Machine gets shut down without warning. Everything needed to resume without re-deriving it
is below. Local paths are `D:\kaggle\rsna-knee\...`.

---

## 1. DONE — `ppxl16/rsna-knee-msft2-ckpt` published (was blocker #1)

**Why this was first and not later.** The exp085 line died on exactly this: its weights were a
*public* dataset, the author made it private mid-session, `dataset_list_files` started returning
`HTTPError 403`, and `kernels_push` **reported success while silently dropping the source**. A
submission must mount assets from a dataset we control. The checkpoints previously existed only
as kernel output.

- Built by `work/publish_msft2_dataset.py` (stages `work/msft2_out/` → `work/msft2_dataset/`,
  writes `dataset-metadata.json`, then `dataset_create_new(public=False)`).
- Log: `work/publish_msft2_dataset.log`. `EXIT=0`, final line `created: /datasets/ppxl16/rsna-knee-msft2-ckpt`.
- Uploaded 8 files: `raptor_ft_msft0.pt` (279 MB), `raptor_ft_msft0_swa.pt` (279 MB),
  `raptor_ft_msft0.json`, `raptor_gold_msft0{,_swa,_epens}.npz`.
- Datasets are versioned: a re-upload needs `dataset_create_version`, not `create_new`.

**Gate status of the asset it holds** (from `work/msft2_out/raptor_ft_msft0.json`, verbatim):
`best_gold_auc 0.9047828479633759`, `partial: false`, `epochs_done: 4`, `n_gold: 58`,
`res: 384`, hist `ep0 0.8903 → ep1 0.8946 → ep2 0.9007 → ep3 0.9048`; SWA over epochs [3,2,1]
= **0.9069**. Clears the A167 solo gate (≥ 0.90); our own measured reload bar was 0.8963.
⚠️ 58 gold studies is small — **solo-passing ≠ ensemble gain (A163)**. Still never blended, never submitted.

---

## 2. DONE — `ppxl16/rsna-knee-msft2-cx` pushed, **QUEUED** on TPU

Built by `scripts/build_msft2_cx_kernel.py`; log `work/push_cx.log`, `EXIT=0`.
Status check (`work/kk_status.log`): `{"status": "QUEUED", "failureMessage": null}`.

**What it is.** A byte-identical copy of the msft2 author body (`kernel-msft2/`), with **only the
prelude** changed — so the XLA grad-checkpoint shim that made msft2 v4 succeed still applies
unchanged. Diff vs `kernel-msft2`:

| | msft2 | msft2-cx |
|---|---|---|
| `--arch` | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` | `convnext_small.in12k_ft_in1k` |
| warm start | `raptor_ft_coatnet_v5_full_swa.pt` | `convnext_small.in12k__latL__v4cx0_best.pt` |
| `--tag` | `msft0` | `cx0` |

Everything else identical: `--res 384 --epochs 4 --bs 2 --k 12 --k_eval 24 --grad_ckpt
--seed 42 --folds 5 --fold 0 --workers 0`. Code file `knee-mri-convnext-finetune.py`.
`build_backbone()` in the author script already special-cases `convnext` (`global_pool="avg"`,
not treated as ViT), so no body change was needed — that was the signal this arch was supported.

**Why ConvNeXt and not a 2nd CoAtNet seed.** A163: a member earns its place by **decorrelation**,
not solo strength. The stack already carries four Raptor views plus CoAtNet readers; another
CoAtNet is nearly collinear with what is there. ConvNeXt-small is a different architecture and is
therefore the cheapest genuinely *new* axis available.

**Why TPU.** ~17 TPU-hours remain and refresh 2026-10-03 — use-it-or-lose-it. GPU is
**0.00 h remaining** until the same date, and TPU cannot run the stacking kernel anyway, so a
second member is the only thing this window can buy.

### ⚠️ KILL CRITERION — check the log before doing anything else
The script evaluates **zero-shot gold58 macro-AUC before training** and prints it. msft2 printed
`GOLD macro-AUC 0.8961 | VERDICT: PASS (parent bar 0.8963)`.
- `head tensors carried: 0` is **expected and acceptable** (v4cx is a foreign head) — the head
  trains during the 4 epochs.
- But if the printed zero-shot macro-AUC is **near chance (< ~0.75)**, the checkpoint did not load
  the intended semantics; **kill the run** rather than let it burn the window, and fall back to
  `msft2-r2` (continue from our own `raptor_ft_msft0_swa.pt`, `--seed 1337`, 4 more epochs —
  the hist is monotone with no sign of saturation, so ~0.910+ is the plausible follow-on).

---

## 3. NOT STARTED, and correctly so — the arm that adds msft2 to the 0943 stack

Host identified: **`kernels/kernel-0943s-2coat`** (`ppxl16/rsna-knee-0943s-2coat`,
notebook `rsna-knee-restructured-version-3.ipynb`, 301 KB). It already demonstrates the pattern
for adding a member: two CoAtNet readers are pulled in via `dataset_sources`
(`mattiaangeli/rsna-knee-coat-resgated-ep10-top3`,
`mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2`,
`mattiaangeli/rsna-knee-coatnet-global96-top3`), plus `kernel_sources` for E11/E13 training.

Adding msft2 needs (a) `ppxl16/rsna-knee-msft2-ckpt` appended to `dataset_sources` and (b) code in
the notebook to load `raptor_ft_msft0_swa.pt`, run inference, and blend it.

**Blocked, and this is a hard block not a choice:** that kernel is `enable_gpu: true`,
`machine_shape: NvidiaTeslaT4`. GPU quota is exhausted until **2026-10-03T00:00Z**. Writing the
blend now would be writing **untestable** code into a 301 KB notebook, so it was deliberately not
started. Resume this on 10-03 once GPU refreshes.

---

## 4. Verified this round (so it need not be re-derived)

Warm-start dataset inventory, via `work/ck_ds2.py` → `work/ck_ds2.log` (all readable, no 403):

| dataset | files |
|---|---|
| `apoorvadixit/rsna-knee-weights-v4cx` | **5** — `convnext_small.in12k__latL__v4cx{0..4}_best.pt` (one per fold) |
| `soyaoki/rsna-knee-my-models` | 20 — `fold{1..4}.pt`, `*_epoch{5,10}.pt`, 5 `raptor_ft_coatnet_*_gate.pt`, 3 `raptor_ft_coatnet_synovitis_spatial_full_fold{0,1,2}.pt` |
| `lixin73/rsna-knee-train-checkpoints` | 8 — `dinov2_seed17_fold0.pt` + OOF/test `.npy`/`.csv`, `folds.csv` |
| `dreaddevelopment/raptor-knee-maxspan` | 1 — `raptor_ft_coatnet_v5_full_swa.pt` (the msft2 warm start) |

`soyaoki` and `lixin73` are **unused fallbacks** if the ConvNeXt arm fails its kill criterion —
`soyaoki` is same-family as msft2 (low decorrelation); `lixin73/dinov2_seed17_fold0.pt` comes with
its own OOF, which makes it priceable without burning GPU.

---

## 5. Traps hit / rules reconfirmed

- `api.dataset_list_files(slug).files[i].totalBytes` **does not exist** in kaggle 2.1.2
  (`ApiDatasetFile` has no such attribute) — it raised `AttributeError` inside the loop and
  truncated the listing after the first file. Print `f.name` only.
- Kaggle derives the kernel slug from the **TITLE**. `kernels_push` reported
  `pushed -> ppxl16/rsna-knee-msft2-cx`, which is what the title `"RSNA knee msft2 cx"` implies —
  metadata `id` and title are deliberately aligned here.
- The build script asserts the substitution actually took, and asserts the old
  `coatnet_rmlp_2_rw_384` string is gone from the injected argv, because a silent no-op replace
  would have re-run CoAtNet under a ConvNeXt name and looked like success.
