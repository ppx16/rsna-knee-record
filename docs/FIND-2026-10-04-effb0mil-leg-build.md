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

⚠️ One residual puzzle, now **resolved** rather than merely recorded: the build kernel as published
declares **three** slots and `CACHE_SLICES = 8`, while the cache it produced has **four** slots and
16 slices. The four are `SAG_FS` / `COR_FS` / `AX_FS` / `SAG_NOFS` — stated by the cache's own
`slots` array, and named `sag_fluid` / `cor_fluid` / `ax_fluid` / `sag_t1` only in the *reader's*
vocabulary (`effb0mil_lib.SLOT_NAMES`). The 26.7%-filled `sag_t1` in the 336 px dataset's index is
a different acquisition under a colliding name; the cache's slot 3 is `SAG_NOFS` at 0.9653 fill.
See §3a for the full mapping and for the v1 failure it caused.

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
4. ~~**The one genuinely ambiguous input is discriminated, not guessed.**~~ **RESOLVED — see §3a;
   it was never ambiguous.** Slot 3 is `SAG_NOFS`, and the mapping is purely positional.

Fail-safe everywhere else too: dataset not mounted, checkpoint/library missing, **slot vocabulary
unresolvable**, cache build error, inference error, UID mismatch, and even an out-of-range weight
all end in `_effb0mil_ok = False` plus a loud `rsna_event('effb0mil_*')`, never in a half-built
blend.

## 3a. ⚠️ v1 died on a SLOT VOCABULARY mismatch — recorded because it is the same class of failure

v1's first run passed everything expensive — library import, the 14.1 GB cache mount, all four
checkpoints — and then died with

```
[effb0] LEG FAILED - continuing with kernel-0943cfm behaviour EXACTLY: ValueError: 'SAG_FS' is not in list
```

**before the convention gate ever ran.** Cause: the leg did

```python
SLOTS.index("SAG_FS")        # v1: WRONG
```

`effb0mil_lib.py` names its four slots `SLOT_NAMES = ['sag_fluid','cor_fluid','ax_fluid','sag_t1']`
with `SLOT_PLANE = [0,1,2,0]` and `SLOT_FS = [1,1,1,0]`. Those are the **cache-build kernel's**
names for the same four slots — the vocabulary is the *cache's*, not ours. Our `SLOTS_RECOVERED`
spells them `SAG_FLUID_FS` / `COR_FLUID_FS` / `AX_FLUID_FS` / `SAG_FLUID_NOFS` / `COR_T1` /
`SAG_T1`, so `SAG_FS` genuinely is not in the list.

**The mapping is purely positional, and the cache's own metadata states it.** From
`ngoanphamvan/sl16s0b-cache-variant` → `track_s_sl16/s_cache_e13_224_meta.npz`:

```
slots      (4,)             ['SAG_FS' 'COR_FS' 'AX_FS' 'SAG_NOFS']   <- cache order
slot_mask  (4407, 4) uint8  per-study availability
crop_mm 130.0   img 224   slices 16
rules      {"order":"dominant_axis","lat":"corner_x","slot_fallback":true,"decode_fill":"zero"}
```

and `s0b_audit.json`: `slice_band=[0.12,0.88]`, `shape=[4407,4,16,224,224]`,
`slot_fill = SAG_FS 0.9349, COR_FS 0.9560, AX_FS 0.9857, SAG_NOFS 0.9653`. Therefore:

| cache index | cache vocabulary | library vocabulary | library `SLOT_PLANE`/`SLOT_FS` | our spelling |
|---|---|---|---|---|
| 0 | `SAG_FS` | `sag_fluid` | plane 0, fat-sat | `SAG_FLUID_FS` |
| 1 | `COR_FS` | `cor_fluid` | plane 1, fat-sat | `COR_FLUID_FS` |
| 2 | `AX_FS` | `ax_fluid` | plane 2, fat-sat | `AX_FLUID_FS` |
| 3 | **`SAG_NOFS`** | `sag_t1` | plane 0, **no** fat-sat | `SAG_FLUID_NOFS` |

**Slot 3 is `SAG_NOFS`** — 0.9653 is the `SAG_NOFS` row — and the 26.7%-filled `sag_t1` in the
336 px dataset's index is a *different acquisition under a colliding name*. `COR_T1` is excluded
by plane.

Two consequences:

* **`sl16_fix.csv` is not required.** The per-study availability the library merges at its line 45
  is `m_<slot>` for the four `SLOT_NAMES`, i.e. exactly `slot_mask (4407, 4)` in the meta npz —
  which *is* published. The only column the meta does not carry is `flip`; the notebook supplies
  `flip = 0` test-side and lets the gate's identity/mirror comparison be the proof.
* **A name-based mapping was the wrong shape.** v2 resolves the four canonical names in cache
  order, prints the resolution, cross-checks it against the library's own `SLOT_PLANE`/`SLOT_FS`,
  refuses a cache whose published vocabulary has moved, and lists every available slot name on
  failure. The convention gate is now a **check** on one fixed assignment, not a search.

⚠️ My earlier 2026-10-04 note calling slot 3 "ambiguous" was **wrong**: the ambiguity was an
artifact of comparing the 336 px dataset's `sag_t1` column against this cache's audit. The meta npz
ships with the cache and states the vocabulary outright.

## 4. What could not be determined here

* **The test cache was not checked against `s_cache_e13_224.npy` on the dev box.** That needs the
  14.1 GB cache *plus* the raw competition DICOMs. The check exists in the notebook as the gate
  and runs before anything is used.
* **The contents of `sl16_fix.csv`.** Unknown and unobtainable — but no longer needed (see §3a).
* Whether the leg's +0.0063 on soft labels survives on the hidden test. Given the macro is over
  12 columns on 3 test studies, and our own precedent, the honest expectation stays
  **0.943 flat with a chance of 0.944**, and the downside at w=0.10 is small.
