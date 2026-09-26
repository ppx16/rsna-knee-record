# `ppxl16/rsna-knee-cn0-weights` — the rescued CoAtNet members

**Payload is not in git.** 1.68 GB lives in the Kaggle dataset; this directory holds only the
metadata and this note, following the `datasets/dataset-*/` convention.

## What is in it

| file | bytes | what |
|---|---|---|
| `raptor_ft_cn0.pt` | 292,823,739 | ⭐ **the arm `scripts/member_add_test.py` measured at +0.0028** on a 0.917-quality blend (fold 0) |
| `raptor_ft_cn0_swa.pt` | 292,825,771 | its SWA variant (C83: 0.8628 vs 0.8623, i.e. no gain — kept for completeness) |
| `raptor_ft_cn1.pt` / `_swa` | 292,823,739 / 292,825,771 | fold 1 (needed if a two-fold member is ever wanted) |
| `raptor_ft_d64cn0.pt` / `_swa` | 292,825,215 / 292,827,247 | the 64-slot dense-corpus variants (that line is closed, §C85) |
| `raptor_ft_*.json` | ~1.7 KB each | the training sidecars: per-label AUC and the per-epoch curve |
| `SHA256SUMS.txt` | 588 B | hashes of the six `.pt` files as they left this machine |

## Why it exists

`cn0` is the only arm this repo has measured **above** the frontier's own DINOv2 seed members
(0.8769 against 0.8676–0.8697, `scripts/arm_parity.py`), and it is the arm the member-addition test
used. It was trained on the rented 4090D, and until 2026-09-26 the checkpoints existed **only on that
box** — there is no `cn*` weight anywhere else. They were pulled off it (`downloads/rent/`,
git-ignored) and then published here, so this dataset is the only copy independent of both the rented
box (now shut down) and this laptop's disk.

## How the copy was verified

The box was shut down before a sha256 comparison against the source could be run, so the verification
rests on four other things (FACTS A140):

1. **byte counts identical to the remote listing** — the sizes above are also what the box reported;
2. **every zip entry passes CRC** — `zipfile.testzip()` returns `None` for all six (accidental
   corruption is therefore ruled out);
3. **identity** — `raptor_ft_cn0.json`'s `best_gold_auc = 0.877929` matches the 0.8779 this repo
   recorded for that arm;
4. **the six files are pairwise distinct** (6/6 distinct sha256), which rules out a copy mix-up.

⚠️ Not verified: sha256 against the original. CRC32 plus exact sizes plus a matching identity rules
out accidental damage, not a deliberate substitution — which is not in the threat model here.

## Geometry (from `state/EXPERIMENTS.md`)

`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, **res 384**, k 12, 12 epochs, `--grad_ckpt`, 44-slot corpus,
fold 0. ⚠️ **TPU cannot train this architecture** (D17 — XLA dies on CoAtNet's relative-position bias);
the frontier's submission *does* run CoAtNet@384 at inference, which is what makes this member
architecturally compatible with that stage.
