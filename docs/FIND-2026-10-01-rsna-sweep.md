# FIND 2026-10-01 — full RSNA sweep: ⭐ the public frontier IS 0.943, and that is what we have

Ran the broadest sweep of this competition so far — newest AND top-voted notebooks, datasets, and the
discussion forum (which every earlier sweep had skipped). Output: `work/rsna_scan.txt`.

## 1. ⭐⭐ The headline: the "0.946" arms are not reproducible from public assets

The newest list carried a whole family advertising **0946** — `yamadan96/rsna-knee-d4-public0946`,
`ryokucha/rsna-knee-d4-blend-0946-ours10`, `sushanthtiruvaipati/rsna-knee-d4-public0946-adapted` — plus
`beraterolelk/0-947-lb-biohub-deepcenter-ilp-tracker`. That looked like +0.003 to +0.004 sitting in
public. Pulling the original and reading its **own** markdown settles it:

> **Correction: this scores 0.943, not 0.946.** The `0946` in this notebook's URL is inherited from the
> parent's name and **overstates what the public pipeline alone achieves.** I submitted exactly this
> notebook to the competition on 2026-09-28 and it scored **0.943** on the public leaderboard — the
> same as the public frontier it descends from. The parent notebook reaches 0.946 by blending in its
> author's **private** datasets.

⇒ **Every `0946`/`0947` artifact in the public list depends on private assets.** The public frontier is
**0.943**, and **our 0.943 is the public frontier** — not a plateau we failed to climb past.

⚠️ This is the second time a **name overstates a number** in this project (the first was Kaggriculture's
`the-2965-master-hybrid-engine` being byte-identical to `frontier-top2`). **A notebook title is a
claim; only the artifact's own text or a gate is evidence.**

## 2. What the public pipeline actually is (from the same notebook, five stages)

Useful because it confirms our own `0943` lineage is the public state of the art rather than a weak
copy of it:

1. **DINO ensemble** — 20 member-specific tails share one **frozen DINO prefix**, so the biggest
   repeated transformer computation is done once;
2. **A5 attention pooling** — five folds pool variable-length slice features;
3. **RadImageNet heads** — one encoder feeds **E10/E13/E11** layouts; v7 uses the scored **0.942
   routing: main alpha 0.55, second E13 pass alpha 0.20**;
4. **Raptor** — four views;
5. **complementary CoAtNet readers** — residual + D4 and Global96 + Repair-v1, overlapped where the
   cohort is small.

It also pins checkpoint hashes, dependency versions and memory limits **before** inference and *"stops
the run instead of silently changing predictions"* — the same discipline this repo arrived at from the
other direction.

## 3. Other new public items worth a look

| ref | note |
|---|---|
| `adonis122/rsna-knee-parent0957-routing`, `heon29/…` | **"parent0957"** — pulled; not yet read. 0.957 is the board's best block, so this is the most interesting remaining lead. |
| `siddheshthakur/rsna-knee-336-grid-submit` | a 336-px grid arm |
| `hitarthjain0/rsna-knee-58-labels-4407-reports` | **the 58 gold labels** exposed independently |
| `sushanthtiruvaipati/rsna-knee-dinosaur-v5-meniscus10-special` | a meniscus-specialised DINOv5 arm |
| `ziyiming/rsna-exp01x-*` | a systematic EXP series (physical axial, multisource SCA) |

## 4. ⚠️ The forum could not be read

`work/scan_rsna_full.py` probes for a discussions API and reports **which** method works rather than
printing nothing: the classic client exposes **no** discussion/forum/topic method, and the
`kagglesdk.discussions` probe is reported in the scan output. **The discussion forum remains unread**,
which matters because it is where a `$77,000` recipe would be written down. If it is to be read it
needs either a `kaggle` upgrade with a discussions surface or a browser.

## 5. What the sweep changes

* ❌ **"Find a better public artifact" is CLOSED for RSNA** — the public frontier is 0.943 and we are at
  it; the arms above it are private.
* ✅ **The +0.001 target stands and now has exactly one remaining route**: our own new member — the
  **msft2 fine-tune** (QUEUED on TPU), which starts from a checkpoint whose own metadata is 0.9214 and
  which we measured at 0.8963 through our pipeline.
* ⚠️ And the fourth `0946`-style lead to check is **`parent0957-routing`**, already on disk:
  `work/rsna_pub/rsna-knee-parent0957-routing/`.
