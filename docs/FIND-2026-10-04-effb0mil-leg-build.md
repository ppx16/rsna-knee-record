# FIND 2026-10-04 — `kernel-0943cfm2eff`: the effb0mil leg, built and gated

Companion to `FIND-2026-10-03-public-artifact-scan.md`. That document is right about the
*target* (the leg is real and the curve reproduces exactly) and wrong about two facts that
would each have cost a run. Both are corrected here, then the notebook and its convention
gate are described.

Kernel: `kernels/kernel-0943cfm2eff/` (id `ppxl16/rsna-knee-0943cfm2eff`), a byte-preserving
copy of `kernels/kernel-0943cfm` plus one stage. Nothing was edited in place;
`kernel-0943cfm` and its siblings are `git status`-clean. See
`kernels/kernel-0943cfm2eff/BUILD-REPORT.md` for the per-cell audit.

## 1. ⚠️ Correction: the 21 GB `tbongc/rsna-knee-cache336-v1` does NOT contain the 224 px cache

FIND-2026-10-03 §3 "Cost" says the mount for this leg is
`tbongc/rsna-knee-cache336-v1` (21 GB) and that it "holds `sl16_fix.csv`, `s_cache_e13_224.npy`,
the 4-slot … 16-slice 224 px train cache". **It does not.** That dataset was created
2026-09-11 and its complete file list is:

```
cache336_index.parquet        248,641
shard_000.npy .. shard_021.npy  1,354,752,128 each
shard_022.npy                  47,416,448
slot_pick_train.parquet        25,455,537
```

— a **336 px, 5-slot** cache (`m_sag_fluid / m_sag_t1 / m_cor_fluid / m_cor_t1 / m_ax_fluid`),
plus per-series metadata at 20 477 rows. There is no `sl16_fix.csv` and no
`s_cache_e13_224.npy` anywhere in it.

Where the 224 px cache actually is:

| file | actual home | size |
|---|---|---|
| `s_cache_e13_224.npy` (4×16×224×224 uint8, 4407 rows) | **`ngoanphamvan/sl16s0b-cache-variant`**, path `track_s_sl16/` | 14,152,040,576 |
| `s_cache_e13_224_meta.npz` | same | 1,148,046 |
| `s0b_audit.json` (the build's own rules record) | same | 624 |
| `best_effb0mil_sl16_tv7w_f0..f3.pt`, `effb0mil_lib.py`, 4× OOF npy + ids | `tbongc/rsna-knee-effb0mil-sl16-tv7w` | 61 MB |
| `sl16_fix.csv` | **nowhere public** — its declared home `tbongc/rsna-knee-labels` is **403** to the dataset API | — |

The kernel that builds this cache is public and its **cache-building code is byte-identical to
cells 19–26 of our own `kernel-0943cfm`**: `ngoanphamvan/rsna-knee-s0b-sl16`. That identity is
what makes a faithful test cache possible at all.

⚠️ One residual puzzle, recorded rather than explained: that kernel as published declares **three**
slots and `CACHE_SLICES = 8`, while the cache it produced has **four** slots and 16 slices, and
its audit json names the four `SAG_FS/COR_FS/AX_FS/SAG_NOFS` while the uploaded metadata names
them `sag_fluid/cor_fluid/ax_fluid/sag_t1`. The two names cannot both denote the same series with
the same fill rate (`SAG_NOFS` 96.5% in the audit vs `sag_t1` mask 26.7% in the metadata). The
notebook's gate resolves this empirically — see §3.

## 2. The measured curve is real, and it was measured on the 4349-study OOF, not on gold-58

Reproduced exactly by `scratch/verify_effb0b.py` from the published OOF and our own
`downloads/oof/oof_equal_prob.npy`, aligned on `StudyInstanceUID`, scored with
`valid_mask` and labels binarised at 0.5 (`scripts/oof_fusion_probe.py`'s own metric):

```
w        macro
 0.00   0.84962
 0.05   0.85290
 0.10   0.85587
 0.15   0.85851
 0.20   0.86083
 0.30   0.86440
```

Per label at w=0.10 — positive on all twelve, largest exactly where we are weakest:

| label | ours | leg solo | Δ @0.10 |
|---|---|---|---|
| Lateral Meniscus | 0.80143 | 0.83625 | **+0.01308** |
| Medial Meniscus | 0.86270 | 0.87485 | +0.00932 |
| PF OA | 0.81670 | 0.82775 | +0.00885 |
| ACL | 0.83933 | 0.81278 | +0.00718 |
| Lateral OA | 0.82403 | 0.81605 | +0.00659 |
| Baker's | 0.88823 | 0.89280 | +0.00663 |
| MCL | 0.82103 | 0.78896 | +0.00521 |
| Medial OA | 0.86968 | 0.87194 | +0.00539 |
| Fracture | 0.88870 | 0.86213 | +0.00427 |
| Contusion | 0.84970 | 0.84940 | +0.00375 |
| Effusion | 0.85494 | 0.84698 | +0.00317 |
| Synovitis | 0.87894 | 0.84526 | +0.00160 |
| **macro** | **0.84962** | **0.84376** | **+0.00625** |

**⚠️ But the 4407-study public OOF is not gold-58.** The leg's four published folds cover
4349 of our 4407 studies and **0 of the 58 fully-labelled ones** — the author held gold out. So
the leg is *invisible* to gold-58: there is no local labelled array on which it can be
independently adjudicated, and the table above is in-sample for the decision. That is the whole
reason the notebook's default weight is **0.10** rather than the 0.20–0.30 the curve peaks at,
and it is the same failure mode as our own closed item #8 (selection illusion: +0.011 in-sample
→ +0.0003 out-of-fold).

## 3. What the notebook does about the real risk (the flip / laterality convention)

`effb0mil_lib.py` applies a per-row `flip` from `sl16_fix.csv` to the frozen train cache before
its forward pass:

```python
x = self._arr()[int(r.row)]
if int(r.flip):                      # cache sl16 KHÔNG lật; đưa về quy ước cache336_v1
    x = x[:, :, :, ::-1]
```

`sl16_fix.csv` is not published, so the contents of `flip` are unknown. The notebook is written
so that it never needs them:

1. **The test cache is built by our own cache builder**, re-pinned to the frozen corpus contract
   (`RULES_LEGACY` = dominant_axis / corner_x / slot_fallback / decode_fill=zero; band
   0.12–0.88; 130 mm; 224 px; 16 slices). Cells 43 and 45 leave `RULES`/`IMG`/`CROP_MM`/
   `CACHE_SLICES`/`SLICE_BAND`/`SLOTS` mutated for other stages, so the leg cell re-pins all six
   and asserts the pin.
2. **The argument for skipping `flip` on the test side**: the only difference the library's
   `flip` variable can express is a per-row **mirror**. If the test cache comes out of the same
   builder as the frozen cache, the two cannot be in different conventions — the degrees of
   freedom are identical.
3. **That argument is tested, not assumed.** Before any prediction is used, the
   **convention gate** rebuilds real TRAIN studies from the raw DICOM, compares them to the
   published `s_cache_e13_224.npy` byte-for-byte, and accepts only identity or an axis mirror —
   and only if each study's pattern is *entirely* identical or *entirely* mirrored (a MIXED
   study would mean the frozen cache was built by a rule we do not have). Any UNEXPLAINED
   comparison, any majority-mirror corpus, any MIXED study → the leg is abandoned and
   `_pipeline_stage.csv` keeps `kernel-0943cfm`'s bytes.
4. **The one genuinely ambiguous input is discriminated, not guessed.** Slot 3 of the frozen
   cache is named both `sag_t1` and `SAG_NOFS` in the public artifacts, and our
   `SLOTS_RECOVERED` contains *both* as distinct acquisitions. The gate tries `SAG_T1` first and
   `SAG_NOFS` second, and keeps whichever hypothesis reproduces the frozen cache. Getting this
   wrong silently would build every test slot-3 from the wrong series.

Fail-safe everywhere else too: dataset not mounted, checkpoint/library missing, cache build
error, inference error, UID mismatch, and even an out-of-range weight all end in
`_effb0mil_ok = False` plus a loud `rsna_event('effb0mil_*')`, never in a half-built blend.

## 4. What could not be determined here

* **The test cache was not checked against `s_cache_e13_224.npy` on the dev box.** That needs the
  14.1 GB cache *plus* the raw competition DICOMs. The check exists in the notebook as the gate
  and runs before anything is used.
* **The contents of `sl16_fix.csv`.** Unknown and unobtainable.
* **The identity of slot 3** (see §3.4).
* Whether the leg's +0.0063 on soft labels survives on the hidden test. Given the macro is over
  12 columns on 3 test studies, and our own precedent, the honest expectation stays
  **0.943 flat with a chance of 0.944**, and the downside at w=0.10 is small.
