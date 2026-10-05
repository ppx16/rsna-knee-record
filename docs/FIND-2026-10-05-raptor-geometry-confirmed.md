# FIND 2026-10-05 — the Raptor geometry is CONFIRMED by three dataset descriptions
## (a suspicion of a silent crop mismatch, checked and FALSIFIED)

**Status: negative result (no bug). Recorded because the failure mode it names is invisible, and because
the three descriptions independently pin our Raptor geometry — which is new corroboration we did not have.**

---

## 1. The suspicion

The notebook was searched for the crop constant. The first and most prominent hit is:

```
CROP_MM = 130.0
CACHE_IMG = 336
GROUP = 3
...
```

Meanwhile the public dataset description of `dreaddevelopment/raptor-knee-native384` says, verbatim:

> *"Identical to that model except for the resolution the corpus is stored at: **384 pixels instead of 336**,
> with the same **44 slices**, the same **6 to 94 percent span**, the same **140 mm crop** and the same
> **42 scoring windows**."*

Our `native384-v8` entry is `img 384, slots _SLOTS44, span (0.06, 0.94), k_eval 42`. Three of those four
fields match the description exactly — **but the crop did not appear to.** If our Raptor views render a
130 mm box while the checkpoint's corpus was built at 140 mm, every Raptor view would be silently
mis-framed and **no gate in this repo would notice**: the cache would build, the shapes would match,
the receipt would pass, and the OOF would just read slightly worse than it should.

That is the exact failure class this repo has already paid for twice, so it was checked before anything else.

## 2. The resolution — it is a false alarm, and the reason is structural

**There are two independent preprocessing blocks in the notebook, and they do not share `CROP_MM`.**

| block | constant | value | who uses it |
|---|---|---|---|
| Raptor view renderer | `IMG` / `CROP_MM` / `SPAN_LO,SPAN_HI` | `336` / **`140.0`** / `0.02, 0.98` | **every `ARMS` entry** (maxspan-v5, native384-v8/v10, msft2, cn0, …) |
| DINO / stack stage | `CROP_MM` / `CACHE_IMG` | `130.0` / `336` | the stack's own stage, `adopt_config_globals(cfg)` overrides both from `cfg` |
| `_RAD_E11` / `_RAD_E13` family | `_RAD_E11_CROP_MM` / `_RAD_E13_CROP_MM` | `130.0` / `130.0` | the independent-estimator family only — **never a Raptor view** |
| legacy block | `CROP_MM` | `10_000.0` | "no crop" sentinel, long superseded |

The Raptor block sets its own value before any arm runs:

```
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
IMG = 336
CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.02, 0.98
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12), ("Coronal", 0, 8), ...
```

**And no `ARMS` entry carries a per-arm `crop` key** — verified by reading the list: each entry has
`name / file / arch / res / img / slots / span / k_eval / reverse / w` and nothing else. So all Raptor
views inherit `140.0`, which is what the description says they should.

⇒ **Our Raptor geometry already matches the author's. No change is warranted, and no arm needs re-rendering.**

## 3. What the check bought us anyway: three independent confirmations

The same sweep read the descriptions of the sibling datasets. Together they pin the Raptor families:

| dataset | description says | our entry | verdict |
|---|---|---|---|
| `raptor-knee-native384` | 384 px, 44 slices, span 6–94 %, **140 mm**, 42 windows | `native384-v8`: img 384, `_SLOTS44`, span (0.06,0.94), k_eval 42 | **4/4 match** |
| `raptor-knee-finespacing` | 336 px, **80 slices** (22 sag-FS / 18 sag / 15 cor-FS / 10 cor / 15 ax), span 2–98 %, 140 mm, **read at 78 windows** | `_SLOTS64`, span (0.02,0.98), k_eval 62 | **span+res+crop match; slot count and k_eval do NOT** |
| `raptor-knee-arms` | ConvNeXtV2-Base @336, CoAtNet @384, EfficientNetV2-L @480; *same 2.5D window scheme and the same per-finding attention pooling head*; rank-mean blend **0.915** public, CoAtNet alone **0.914** | no effv2l arm exists | see §4 |

The `finespacing` row is the interesting one: **it is a genuinely denser corpus than anything we render.**
Our `_SLOTS64` uses 64 slots / 62 eval windows; the author's finespacing corpus is 80 slots / 78 windows
at the same span, crop and storage resolution. That is the "v9 finespacing gap" the hand-off named, and
it is **still open** — this check did not close it, it confirmed the shape of it.

⚠️ **Caveat on the slot layout:** the description lists five series with counts 22/18/15/10/15 (= 80).
Our `_SLOTS64` is a *different* five-way split summing to 64. So a finespacing arm is **not** a
constant-flip of an existing arm — it needs a new slot vector, a new cache, and a new `k_eval`.
That is a real build, not a re-render.

## 4. A number worth keeping: `raptor-knee-arms` scores 0.915

The three-backbone rank-mean blends to **0.915** public; the CoAtNet member alone is **0.914**.
Checkpoint `gold_auc` values: `raptor_ft_coatnet384.pt` **0.9025**, `raptor_ft_cnv2b336.pt` **0.8901**,
`raptor_ft_effv2l480.pt` **0.8716**.

**Read this against our board.** Our stack is **0.943**. So this whole family — three backbones, same
window scheme, same head — lands ~0.028 *below* where we already are, and adding the ConvNeXtV2 and
EfficientNetV2 members to each other bought the author only **+0.001**.

⇒ **`raptor-knee-arms` is not the lever.** It is worth recording as *evidence against* the "add more
backbone diversity" intuition: the author tried exactly that, with three architectures and two extra
input resolutions, and the marginal return was +0.001.

⚠️ Still unresolved from the previous turn: `cnv2b336` and `coatnet384` both report **F_dim 1024** while
our v5/v8/v10 load through an **F_dim 768** path. That must be understood before any of these is wired,
or the head will fail silently. It is now *less* urgent, because §4 says the family is unlikely to pay.

## 5. The rule this leaves behind

**`CROP_MM` is not a global in this notebook — it is a per-block global that is reassigned by whichever
preprocessing block ran last.** Any future question of the form "what crop do we use?" must be answered
*per family*, by finding the assignment that dominates the code path in question, never by grepping for
the first or most frequent occurrence. The same holds for `IMG`, `SPAN_LO/SPAN_HI` and `SLOTS`.

Related: `docs/FIND-2026-10-04-stale-log-string.md` (a hardcoded print that lied about the real grouping)
is the same disease in a different organ — **the notebook's *stated* configuration and its *effective*
configuration are not the same object, and only reading the code path settles it.**
