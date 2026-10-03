# FIND 2026-10-04 — three preprocessing datasets checked: no new pricable leg, one genuinely different corpus, and a denominator trap

Follow-up to `docs/FIND-2026-10-04-public-artifact-scan.md`, prompted by a bounded re-scan of the
newest `rsna-knee` datasets. Four datasets that the earlier sweep had not catalogued were inspected.

## 1. No new pricable leg exists among them

| dataset | what it actually is |
|---|---|
| `sachinhambar/rsna-knee-ft-cache-data` | a **JPEG image cache**: `ft_cache.json` = `{img_size 384, cache_slices 20, jpeg_quality 90, crop true, studies 4407, shards 32}`; `ft_cache_index.json` maps each study to `[shard, offset, byte_length]`. **No labels, no predictions.** |
| `vaprf5/rsna-knee-v11-preprocessed` | `manifest.csv` = `study_id, cache_path, num_slices` (4408 rows). A cache manifest, **no labels**. |
| `richyroseyiloveyou/rsna-knee-full-volume-288-npz` | one `.npz` per study, full volume, 288 px. Another geometry. |
| `codeandodev/rsna-knee-procesado-32x224x224` | `manifest_processed.csv`, **one series per study**, plus `.npy` per study. |

⇒ **Nothing here publishes predictions or OOF**, so none of them can be priced the way the EffB0 leg
was. They are **inputs**, and turning an input into a member means training — the axis §D closes.

⚠️ Also checked and **rejected on quality**: `codeandodev/rsna-knee-run-base` *does* publish per-study
predictions (`A_scratch_pred_dev.csv`, `B_resnet50_pred_dev.csv`) — but `comparacion_auc_oro.csv` shows
**MACRO 0.561 (scratch) / 0.733 (resnet50)** on a 75-study dev split from a 500-study toy corpus.
Far below the 0.90 gate, and below even our from-scratch ceiling. **Dead.**

## 2. ⭐ A genuinely different public corpus geometry

`yutoshimomura/rsna-knee-exp096-crop140-518-cache` → `cache_summary.json`:

```json
{"studies": 4407, "successful": 4407, "failed": 0,
 "slices": 597979, "windows": 556395, "bytes": 160458331331,
 "resolution": 518, "crop_mm": 140.0, "retained_range": [0.06, 0.94],
 "slots": [["Sagittal", 1, 18], ["Sagittal", 0, 14],
           ["Coronal", 1, 12], ["Coronal", 0, 8], ["Axial", -1, 12]]}
```

**Five slots (2 sagittal, 2 coronal, 1 axial) with per-slot counts 18/14/12/8/12 = 64 windows**, at
**518 px** and a **140 mm** crop with a **0.06–0.94** retained range.

Compare our stack: 4 Raptor views, **94 windows**, `crop_mm` 130 after re-pin, band (0.12, 0.88).
So this is a **different slot count, different band, different crop and a different resolution** —
and 518 is notably the **native `image_size` of the `facebook/dinov2-base` backbone** we used in the
exp085 reconstruction (`checkpoint position_embeddings (1,1370,768)` = 1+37²), where 224 was only the
input size applied by interpolation.

⚠️ **This is an input, not a member.** Using it means training something on it, which is the closed
axis (§D D5/D9/D14). Recorded because it is a real, public, fully-built geometry we did not know about
— and because "518 native" connects to a backbone already in this repo.

## 3. ⚠️ A denominator trap I walked into and backed out of

`manifest_processed.csv` has **one series per study** — verified, not assumed:

```
rows: 4408 (header + 4407)   studies: 4407   series per study: min 1, median 1, max 1
```

I first computed "**24.2 % of studies have no sagittal series**", "31.0 % have no sagittal fat-sat",
"85.3 % have no sagittal T2". **Those numbers are meaningless as corpus statistics** — the denominator
is *one series chosen by that dataset's author per study*, not the corpus's series. The check that
caught it was printing the series-per-study distribution, which I nearly skipped.

⇒ **The only defensible statement from this file** is about the *naming heterogeneity it displays*:
across its 4407 single-series rows the `descripcion` field contains **`DummySeriesDesc!` 558 times**
plus hundreds of scanner- and site-specific spellings — `pd_tse_fs_sag_320`, `pd_tse_fs_sag_384`,
`PDW_TSE_SPAIR_Sag`, `SAG DP SPIR`, `R- Sag PD FSE FS` / `L- Sag PD FSE FS`, `int_tse_sag_fs_p3_s2_2.5mm_drb_384`,
`RT.Sag_PDW_TSE_SPAIR_Sag`, `SG PD FatSat`, … That heterogeneity is **consistent with** (not proof of)
the series-matching difficulty that our pipeline and the EffB0 rebuild both ran into.

⚠️ **It does NOT establish** that the EffB0 gate's `0 slot-series` was caused by naming — the gate
sampled 4 studies and `manifest_processed.csv` is a different author's selection. The EffB0 post-mortem's
open question stands as *unresolved*; this is a plausible adjacent observation, not the answer.
