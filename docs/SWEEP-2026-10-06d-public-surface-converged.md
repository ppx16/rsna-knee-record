# SWEEP 2026-10-06 (d) — the newest public activity is forks; two negatives worth keeping

Third sweep of the day (07:17Z), because the community is demonstrably active — the 0.944 notebook that
gave us the reader leg was posted at 01:29Z. Everything found is recorded, including the things that are
**not** levers, so a future session does not re-chase them.

---

## 1. ⭐ An independent party confirms the public ceiling is 0.943 — by correcting a 0.946 title

`sameerk2004/rsna-knee-sil-d2-mcl-latmen` opens with an unprompted correction:

> **"I did not write any of the code in this notebook."** Every code cell is taken verbatim from
> `ryokucha/rsna-knee-d4-blend-**0946**-ours10`; *"I only removed that author's six trailing cells, which
> depend on private datasets that are not attached in its public metadata."*
>
> **"## Correction: this scores 0.943, not 0.946. The `0946` in this notebook's URL is inherited from the
> parent's name and overstates what the public pipeline alone achieves. I submitted exactly this notebook
> to the leaderboard."**

⇒ **A notebook named 0.946 scores 0.943 once the private cells are removed.** That is the third independent
confirmation this session that the **public** pipeline's ceiling is `0.943` — and it locates the `+0.003`
in **private data**, not in any public trick.

⚠️ It also means the title-based sweeps this repository has been running are systematically **optimistic**:
`0946` in a slug is a claim, and in this case it was false.

## 2. The `sil-d2` family is forks, not an ablation study

The name suggested a systematic experiment set (`dino-v5`, `tony-dinov3`, `4arm-v55`, `mcl-latmen`,
`meniscus10`). It is not — each is a copy of somebody else's notebook with a trailing-cell edit:

| notebook | what it actually is | score evidence |
|---|---|---|
| `sil-d2-dino-v5` | a copy of `romantamrazov/rsna-knee-dinosaur-v5` (the unverified Stack++) | 0.942/0.944 literals only |
| `sil-d2-4arm-v55` | a standalone repack of `anhadmahajan06/...-take-care-of-your-knee` v46 | **0.902–0.916** — far below us |
| `sil-d2-mcl-latmen` | `ryokucha/...0946-ours10` minus its private cells | **0.943, corrected** |
| `sil-d2-meniscus10` | the V5 0.943 graph + `public0033` on **Medial Meniscus only** | claims 0.943, i.e. no gain |

## 3. ⭐ Worth keeping: the meniscus specialist's *scope* differs between our arm and the newest public one

| | ours (`56845973`) | `sil-d2-meniscus10` |
|---|---|---|
| labels touched | **Medial + Lateral Meniscus** | **Medial Meniscus only** |
| weight | fixed 90:10 | 5 % / 10 % / 15 % |
| other 11 labels | byte-identical | byte-identical |
| fallback | on any failure | on any failure |
| **result** | **0.940 — a 0.003 regression** | **0.943 at every dose — no gain** |

⇒ **A testable hypothesis exists that we cannot currently resolve:** our regression may have come from
applying the specialist to **Lateral** Meniscus as well as Medial. `docs/PREREG-2026-10-05-meniscus-arm.md`
fixed Medial+Lateral because that is what the author's recipe named.

⚠️ **But it stays closed.** Their Medial-only design reports **no improvement either**, so the hypothesis'
best case is *neutral*, and the axis was already closed by a measurement in the other direction. **A
variant that might be no-worse is not a reason to spend a submission.**

## 4. The two new dataset candidates are not levers either

| dataset | listing says | verdict |
|---|---|---|
| `barun2104/rsna-knee-mri-processed-3d-volumes` | *"24,371 normalized uint8 3D MRI volume arrays (.npz)"* — i.e. one per **series**, 24371 being the training series count | another preprocessing of the **same competition data**. A "new data view" is an allowed axis, but `C84`'s +0.0068 OOF for a denser view was followed by `C86`'s discovery that the dense corpus carried a min-max normalisation defect (member 0.862 → 0.869 even after fixing). Building a third corpus for a low prior is not worth it. |
| `lixin73/rsna-knee-train-checkpoints` | `dinov2_seed17_fold0.pt` + `_oof.csv`/`_oof.npy`/`_test.*` + `folds.csv` | **one fold** of a DINOv2 model, with its own OOF. Evaluable, but a single fold at this lineage's level is not a member candidate. |
| `alenic/...jpeg-224x224` | JPEG renders of the **test** DICOMs at 224 px | no labels, and we already have the DICOMs. Useless. |

## 5. Net result of the sweep

**No new lever.** What it produced is one confirmed fact (the public ceiling is 0.943, now confirmed by
three independent routes) and two recorded hypotheses. The honest reading is that the public surface is
**converged**: everything new is a fork, a reprocessing, or a title that overstates.
