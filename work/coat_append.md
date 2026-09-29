
## 7. ⭐ The CoAt family dominates the rerun — and the receipt says why

`downloads/radcal00/_coat_arm_receipt.json` is the child process's own receipt for **one** CoAt member
(the resgated arm), and it decomposes that member's runtime properly instead of leaving it as a single
`[phase]` number:

| field | value | reading |
|---|---|---|
| `studies` / `series` | 3 / 15 | the visible cohort |
| `elapsed_seconds` | **37.33** | the whole child, for 3 studies |
| `shards` | 2, on GPU 0 and GPU 1 | the member splits the cohort across both T4s |
| `model_load_seconds` | **7.93** and **7.90** | 3 checkpoints each, so **loaded twice per member** |
| `dicom_preparations_per_study` | **1** | each member prepares its own DICOMs |
| shard 1 | 2 studies, 16.96 s | ⇒ **(16.96 − 7.90)/2 = 4.53 s/study** |
| shard 0 | 1 study, 18.28 s | ⇒ 10.35 s/study — inflated by first-shard warmup, so **4.5 is the better estimate** |

The parent's own phase for the same member is **53.6 s** (`0943cf`), so roughly **16 s of every member is
parent overhead in the critical path** — a `pip install --no-deps --target` of its own `timm-1.0.22`
wheel, plus process spawn.

### Revised estimate

Per member at N=1322 ≈ 16 (parent, fixed) + 15.8 (both shards' loads, fixed) + (1322/2) × 4.53 ≈ **50 min**.
Three members ⇒ **~2.1–2.5 h for the CoAt family alone**, which is the dominant term:

| | @1322 |
|---|---|
| `a5` | 19.0 min *(measured)* |
| `dinov2` | 37–41 min *(measured)* |
| **CoAt family ×3** | **~2.1–2.5 h** *(estimated from the receipt)* |
| `radimagenet` + `public_raptor` | ~20–40 min *(unmeasured)* |
| **total** | **~3.5–4 h** |

⚠️ So the earlier **2.3–3.6 h** is revised **upward to ~3.5–4 h**, and the reason is that the CoAt family
is bigger than the proxy-ratio extrapolation assumed. At 4 h, `RuntimeSeconds/32400` ≈ **0.44** against an
AUC term of ≈ −2.06, and among the **308 teams tied with us at 0.943** runtime is the only differentiator.

### The waste, precisely

1. **Each member is passed only `competition_root`** — no cache path — so every member re-prepares the
   DICOMs. The receipt's own `dicom_preparations_per_study: 1` is a within-member claim ("we prepare each
   study once"), not a cross-member one. **Three members ⇒ the image preparation is done three times**,
   and it is per-study, so it scales with the 1322-study rerun.
2. **Each member loads its 3 checkpoints twice**, once per shard. That is 47 s across the family on the
   visible run — a fixed cost, so it matters much less at N=1322 than item 1.

### Why merging is not the fix, and what is

The three members run as **separate `python -c` children with privately pip-installed dependency trees**
(`pip install --no-deps --target <envd> <timm wheel>`, and each asserts its own `cv2.__version__`). That
dependency isolation is presumably *why* they are separate processes, so folding them into one process
risks exactly the kind of version-conflict failure the isolation prevents.

⇒ The natural fix is **not** one process but **one preparation**: render the pixels once and have all
three children read the result. That is the lever worth investigating next, and it is a code question,
not a GPU one — the thing to establish first is whether the children can consume a preparation cache at
all, since `run_submission` lives in the artifact (`input_resgated_runtime.py` is generated from the
Kaggle dataset, not from this repo) and its signature as called here passes no cache.
