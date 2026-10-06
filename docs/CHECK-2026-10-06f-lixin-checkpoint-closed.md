# CHECK 2026-10-06 (f) — a new checkpoint dataset, measured at zero GPU and closed

Fifth round of the autonomous goal. Both live measurements (`56872230` the reader leg, and the distilled
zero-shot gate) were still pending, so this round took a **new public artifact** and settled it with the
pattern that worked for `yunusgmsoy`'s head: **a shipped OOF is a free full-population measurement.**

---

## 1. The candidate

`lixin73/rsna-knee-train-checkpoints` ships:

```
dinov2_seed17_fold0.pt          89 MB
dinov2_seed17_fold0_oof.csv    173 KB      <- the OOF
dinov2_seed17_fold0_test.csv/npy
folds.csv
training_history_seed17_fold0.csv
```

i.e. **one fold** of a DINOv2 model **with its own out-of-fold predictions**, which is enough to place it on
our one-convention table before touching an accelerator.

## 2. ⚠️ The first measurement was n = 8, and was not a result

Scored against the 58 official gold studies it returned **0.8454** — but the join matched only
**8 of 58** studies, because the OOF's 882 held-out studies barely overlap the gold set.

**n = 8 is noise.** gold-58's paired CI is already ±0.021 *on all fifty-eight* (`FACTS` D16); at eight it
means nothing. Reporting 0.8454 as a finding would have been the same class of error as the
`2^S` subset maxima that `kiritohayate95`'s notebook explicitly warns about.

⇒ **The lesson worth keeping: before scoring a shipped OOF against gold-58, check the OVERLAP COUNT, not
just that the join succeeded.** A successful join on 8 rows looks exactly like a result.

## 3. The right instrument, and the verdict

The OOF covers **882** studies and we hold a label table covering all 4407 — the convention
`work/rank_all_members_one_convention.py` already uses for every member. Scored against
`report_labels_v5` with **the 58 gold rows removed** (that table equals the gold truth elementwise on those
rows, so including them would flatter the member and break comparability):

| label | AUC | | label | AUC |
|---|---|---|---|---|
| Lateral Meniscus | 0.7709 | | Medial OA | 0.8851 |
| MCL | 0.8225 | | Synovitis | 0.8876 |
| Lateral OA | 0.8416 | | ACL | 0.8921 |
| PF OA | 0.8417 | | Fracture | 0.9009 |
| Medial Meniscus | 0.8632 | | Baker's | 0.9134 |
| Contusion | 0.8635 | | | |
| Effusion | 0.8781 | | **MACRO (12 labels, n=874)** | **0.8634** |

Against the existing table:

| member | macro (same convention) |
|---|---|
| `surrogate_radfree` | 0.8960 |
| `msft2` | 0.8847 |
| `d64cn0` | 0.8679 |
| **candidate** | **0.8634** |
| `cn0` | 0.8631 |
| `cx0` | 0.8493 |

⇒ **0.8634 is `cn0` to within 0.0003 — and `cn0` is already in our stack.** It is also a **single fold**,
while our stack already carries **twenty** DINOv2 slot-attention members from `pilkwang/rsna-knee-weights`.

⇒ **Not a candidate. Closed at zero GPU cost.**

## 4. A secondary observation worth keeping

Its two weakest labels are **Lateral Meniscus (0.7709)** and **MCL (0.8225)** — the same two that the
rejected `yunus` head was worst on (0.7418 / 0.7417), and the same two this repository's audit named as the
hard cases. **Three independent models now agree on where the difficulty is**, which is mild evidence that
the difficulty is a property of the data rather than of any one architecture.

⚠️ And a caution against over-reading the member gate: `cn0` sits at **0.8631**, well below A167's nominal
"≈0.90", and it *is* in the stack and did contribute. The gate is a heuristic, not a law — which is why
`msft2` at 0.8847 was worth trying and this at 0.8634 is not: the difference is not the number, it is that
**`cn0`'s family was not already represented twenty times over.**
