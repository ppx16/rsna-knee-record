
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
