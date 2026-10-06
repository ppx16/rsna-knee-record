# RE-AUDIT 2026-10-06 — the pair is re-confirmed; §D stays closed; and two of my own claims were wrong

Requested by the user ("开始"). Everything below is re-derived from recorded data rather than restated
from the sheets, because the `0.943` block has moved enough that earlier in-block percentiles were
computed in coordinates that no longer exist.

---

## 1. The pair is re-confirmed: `56814081` primary + `56599932` second

The pair's value is the **better of its two arms**, so the question is not which two score highest but
which two maximise `E[max(a, b)]`. With the measured signs — `S` (msft2 view) **> 0** on the rank,
`M` (MCL) **≲ 0**, `R` (routing flattening) **≲ 0** — the candidates are:

| pair | value | when it wins |
|---|---|---|
| (`cfmsft2`, **`cf`**) | `C + max(S, 0)` | **the floor is exactly `C` whenever `S < 0`** |
| (`cfmsft2`, `cfm`) | `C + max(S, M)` | only if `M > 0` privately (measured ≲ 0) |
| (`cfmsft2`, `cflat`) | `C + max(S, R)` | only if `R > S` (measured ≲ 0) |
| (`cfmsft2`, `cfmsft2m`) | `C + S + max(0, M)` | never better than the first when `S < 0` |
| (`cfmsft2`, `cfmsft2flat`) | `C + S + max(0, R)` | same |

⇒ **`(0943cfmsft2, 0943cf)` is the only pair whose value cannot fall below `C`** — the best-measured
claim in the repository — no matter which way `S` goes on the private set. **Unchanged.**

## 2. The block is not just drifting, it is being squeezed

```
at                   teams  rank  margin   lo    hi     n   our in-block %
2026-10-05T07:59Z     5192   518     +1    376   1389  1014    14.02
2026-10-05T16:01Z     5226   551    -29    414   1419  1006    13.63
2026-10-06T06:11Z     5278   645   -118    485   1483   999    16.03
```

Three things at once, and only the first is ordinary drift:

* **the block is SHRINKING** — 1014 → 999 members: teams at `0.943` are leaving it faster than new ones
  arrive, i.e. **improving**;
* **the block's floor is rising fast** — 376 → 485 (+109 in 22 h): the weakest `0.943`s are also moving up;
* **our position inside the block got worse** — 13.63 % → 16.03 %.

⇒ **Our true score sits in the lower part of the `0.943` cohort and is falling through it.** This is the
sharpest statement of the problem the whole campaign faces: a static `0.943` is a *losing* position.

## 3. ⚠️ I checked my own suggestion that the rank instrument reopens §D. It does not.

I had implied that §D ("blend weights / routing / calibration") was closed *at board resolution*, and that
the 2026-10-04 rank discovery therefore reopened it. **Reading the clauses, that is false.** They are
measurements, several with a noise estimate:

| clause | what was actually done |
|---|---|
| **D1** | every pre-registered weight axis swept — **all ≤ 0.941**, and it reports **axis size 0.0046 vs artefact 0.0012** |
| **D2** | per-label routing — **out-of-sample −0.0003** |
| **D3** | five pooling probes + `a5w30` + `arm-weights-alt` — *"all five read 0.941 (flat)"* |
| **D10** | `gap ±1→±2` — OOF paired **−0.0119, CI [−0.0041, −0.0199]** |
| **D11/D12** | `rad_cal_w` off, and all four corners of the 2×2 — monotone **0.941 / 0.940 / 0.939 / 0.938** |
| **D13** | the community's "one weight" claim — **we A/B'd it ourselves: 0.941, flat** |

**⇒ §D is closed by measurement, not by the fallacy that closed D14. The suggestion is withdrawn.**
(Only D14's *stated reason* was that fallacy, and it was already withdrawn separately.)

**D16 independently corroborates yesterday's finding**: gold-58 is unusable for arm-to-arm comparison
(paired CI ±0.021) — the same fact that made `report_labels_v5` score a tautological 1.0000.

## 4. ⚠️ Two of my own claims were wrong, both found in the logs

### 4a. `msft2` fold 0 — the checkpoint IN OUR STACK — ran on **CUDA**, not TPU

```
msft2 fold 0:  [tpu] no XLA (ModuleNotFoundError("No module named 'torch_xla'")) -> dev=cuda
               device cuda | res 384 | k 12/24 | bs 2 | tag msft0
msft2 fold 1:  [tpu] activation checkpointing -> torch_xla.utils.checkpoint
               [tpu] xla_device=xla:0 (single process, no xmp.spawn)
```

The kernel metadata requests TPU, but that image had no `torch_xla`, so **the trainer fell back to CUDA
and said so in its own log**. I told the user "fold 0 已经在 TPU 上跑过" on the strength of the
**metadata flag**. **The flag is not the execution** — this repository's own `A32` lesson, running in the
opposite direction from the case it records ("named `tpu-msft`, pushed as a GPU kernel").

⇒ Consequences: fold 0 consumed **GPU** quota, not TPU; and fold 0 and fold 1 were trained on
**different backends**, so their near-equality (gold-58 **0.9069** vs **0.9088**) is a cross-backend
comparison — evidence the recipe is robust, not a clean same-backend replication.

### 4b. `D17` is contradicted — CoAtNet *does* train on XLA

`D17` closes *"training `coatnet_rmlp_2_rw_384` on torch_xla"*, dying in the additive relative-position
bias (`timm/layers/pos_embed_rel.py:440` → `torch_xla/csrc/helpers.cpp:619`), and concludes *"unless you
patch that timm op, this road is closed"*. **Fold 1 trained exactly that architecture on `xla:0` today
and completed** (SWA over epochs [3,2,1], gold **0.9082**).

⚠️ **The likely cause, stated as a hypothesis and not a result:** the msft2 patch routes activation
checkpointing through `torch_xla.utils.checkpoint` instead of `torch.utils.checkpoint`, and `X7` disabled
`fused_attn`. Either could move the failing op off the path XLA rejects. **This is a hypothesis; it is not
tested.** What *is* established is that D17's "this road is closed" is too broad — it holds for the
configuration that failed, not for the configuration that now runs.

**Nothing currently depends on either finding** (no further folds will be trained), but both are recorded
because a stale closure that says "此路不通" about a road we can now drive is exactly the kind of note
that costs a future session real time.

## 5. What the re-audit changes

**Nothing in the recommendation** — the pair stands, and §D stays closed. What it does is replace two
prose claims with log evidence, one of which was mine and wrong.
