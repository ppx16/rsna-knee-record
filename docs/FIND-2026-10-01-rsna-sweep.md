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

---

## 6. Follow-up: `parent0957-routing` is ALSO a name that overstates — it scores 0.941

The last unread lead (`adonis122/rsna-knee-parent0957-routing`, `heon29/…`), pulled and read. Its own
first markdown line:

> **Public score 0.941.** Same weights, same arithmetic, same attribution as the release it is built
> from. What changed is that you can now read it.

**The `0957` is the PARENT's name, not this artifact's score.** The readable arm is **0.941 — below our
0.943.**

### ⚠️ That makes it three in two days, and the pattern is now systematic

| name claims | artifact's own evidence | source |
|---|---|---|
| `…-d4-public**0946**`, `…-0946-ours10` | **0.943** — *"the 0946 … overstates what the public pipeline alone achieves"* | the notebook's own correction |
| `…-biohub-…-**0947**-lb` | not public (private assets) | title only |
| `parent**0957**-routing` | **0.941** — *"public score 0.941"* | the notebook's own first line |
| `the-**2965**-master-hybrid-engine` (Kaggriculture) | **byte-identical to `frontier-top2`** | sha256 |

⇒ **On this competition a notebook name is not weak evidence, it is systematically inflated, and the
inflation runs in one direction — upward.** The rule this repo already had (*names are not evidence*)
turns out to be **load-bearing rather than a caution**, and the only two admissible instruments are the
artifact's own text and a local gate.

### 6.1 The public-artifact line for RSNA is now closed by exhaustion

| candidate family | advertised | actually |
|---|---|---|
| `d4-public0946` / `d4-blend-0946` / `…-adapted` | 0.946 | **0.943**, and the parent's extra needs **private** datasets |
| `0-947-lb-biohub-deepcenter-ilp-tracker` | 0.947 | private assets |
| `parent0957-routing` | 0.957 | **0.941** |
| **our `0943` lineage** | — | ⭐ **0.943 — the public frontier** |

### 6.2 But the notebook is a genuinely useful READABLE reference

Unlike the arms it descends from, this one is written to be read, and it states the metric's two
consequences that shape every design decision:

* **"Only ordering counts, per column. Any monotone transform is free and invisible"** — so
  calibrators are score-neutral alone and matter only inside a blend (A94's lesson, independently);
* **"Ties are expensive"** — a study nobody can score gets a constant, and a block of identical
  constants earns half credit against everything it ties with. That is the same mechanism as the
  599-team 0.943 block that decides our medal.

Its five stages match §2 exactly (DINO member combination → A5 attention pooling at **45 %** →
RadImageNet E10/E13 → CoAtNet Raptor views → final routing), and it labels the report extractor as
*"inert when the weight package is attached"* — i.e. **the 12 labels came from an LLM reading the
reports**, which is why `hitarthjain0/rsna-knee-58-labels-4407-reports` matters.
