# FINAL SELECTION — current sheet, 2026-10-05

> ## ⛔ BOTH RECOMMENDATIONS BELOW ARE SUPERSEDED — see `docs/RESULT-2026-10-05-two-arms-flat-or-worse.md`
>
> Two arms landed after this sheet was written, and **both came in at or below the baseline**:
> `56848247` (`0943cfmsft2flat`) = **0.942**, `56845973` (`cfmsft2m-men`) = **0.940**. So
> **`R` (routing flattening) is measured unhelpful** — which retires §3's hedge argument *and* the
> §6 amendment that replaced the second slot with the flat arm.
>
> **Final pair, revised:**
>
> | slot | ref | arm | public |
> |---|---|---|---|
> | **primary** | **`56814081`** | `0943cfmsft2` — `C + S` | 0.943 |
> | **second** | **`56599932`** | `0943cf` — `C`, deliberately without `S` | 0.943 |
>
> Reasoning in `RESULT-2026-10-05` §3: with `M ≲ 0` and `R ≲ 0` both **measured**, the pair
> `(cfmsft2, cf)` is the only one of the three candidates that hedges `S` — the one effect whose
> evidence is a public-**rank** gain, so whose private sign is genuinely unknown. `C` is in both arms;
> what the second slot gives up is `S`, and giving it up **is** the hedge.
>
> The body below is kept verbatim so the reversals stay auditable.

**Supersedes `docs/FINAL-SELECTION-2026-10-04.md`, which is stale by one measured result.**
That sheet was written on 2026-10-04 and **never mentions `msft2` at all** — verified by grep, not by
recollection. Everything it says about `0943cflat` being the right *second* slot still holds; its
*primary* is now wrong. The 2026-09-28 `docs/FINAL-SELECTION.md` remains the reasoning of record for
the hedge argument.

Deadline **2026-10-22 23:59 UTC**. **Two** submissions may be selected.

---

## 1. The measured result the old sheet predates

`0943cfmsft2` = **`ref 56814081`**, submitted 2026-10-04 04:28Z, public **0.943**.

| | before 04:13Z | after 12:44Z |
|---|---|---|
| our rank | 708 | **487** |
| `0.943` block | 344..1280 (937) | 357..1309 (953) |
| bronze line (top 10 %) | 506 | 510 |
| **in-block percentile** | **38.89 %** | **13.66 %** |

**This is the only change in the entire repository with a measured improvement on the rank instrument**
(`docs/RESULT-2026-10-04-msft2-arm-rank-move.md`). +221 ranks while the field grew 0.8 %, and the date
confound was excluded (`pearson(rank, submission_time) = −0.13`, dates scattered at both ends).

Every other arm measured since has been **flat**:

| ref | arm | in-block percentile | verdict |
|---|---|---|---|
| `56798408` | `0943cf448` (M448 fallback) | 39.5 % → 38.9 % | no change |
| `56826372` | `0943cfmsft2m` (+MCL) | 13.67 % → 14.31 % | **no improvement; slightly worse** |

## 2. The arms actually in play

Let the private-set effects, measured against the 0.943 baseline, be:

* **`C`** = cn0 member + C81 fracture fix — the best-measured claim we hold (C81: +0.0118 on that
  column, +0.00099 macro, n=4185; five label sources, gold-58 the sole dissenter)
* **`S`** = msft2 CoAtNet SWA as a 6th Raptor view — ⭐ **the only effect with a measured rank gain**
* **`M`** = MCL out of `_RAD_EXCLUDE` — **measured flat** on the rank (row above)
* **`R`** = outer per-label routing fully flattened — the frontier's own written remedy for a map tuned
  on a public split; the axis it calls the likeliest to give back points privately

| ref | arm | = base + | public | evidence carried |
|---|---|---|---|---|
| ⭐ **`56814081`** | **`0943cfmsft2`** | `C + S` | 0.943 | **the measured rank gain** |
| `56826372` | `0943cfmsft2m` | `C + S + M` | 0.943 | rank gain, then a change that measured **flat** |
| ⭐ **`56613303`** | **`0943cflat`** | `C + R` | 0.941 | routing hedge **+** `C`; strict superset of `0943p` |
| `56791368` | `0943cfm` | `C + M` | 0.942 | `M` measured flat |
| `56599932` | `0943cf` | `C` | 0.943 | `C` only |
| `56599159` | `0943p` | `R` | 0.941 | routing hedge only — **dominated by `0943cflat`** |
| `56845973` | `0943cfmsft2m-men` | `C + S + M +` meniscus | **PENDING** | see §5 |

## 3. ⭐ Recommended pair

| slot | ref | arm | public |
|---|---|---|---|
| **primary** | **`56814081`** | `0943cfmsft2` — cn0 + fracture fix + msft2 6th view | 0.943 |
| **second** | **`56613303`** | `0943cflat` — cn0 + fracture fix + flat routing | 0.941 |

### The case check (value of the pair = the better of its two arms)

```
X = (cfmsft2, cfmsft2m)   = base + C + S + max(0, M)
Y = (cfmsft2, cflat)      = base + C + max(S, R)          <- RECOMMENDED
W = (cfm,     cflat)      = base + C + max(M, R)          <- the 2026-10-04 sheet's pair
```

* **`Y ≥ X` always, whenever `M ≤ 0`.** With `M ≤ 0`: `X = C + S` while `Y = C + max(S, R) ≥ C + S`.
  **And `M ≤ 0` is exactly what was measured** (§1, row 2). ⇒ `cfmsft2m`'s only extra content is a
  change that measured flat, so it can never pay for the second slot, while `cflat` can pay `R`.
* **`Y ≥ W` always.** `max(S, R) ≥ max(M, R)` requires only `S ≥ M` where `R < S`; and where `R ≥ S`
  the two are equal. Since `S` measured **> 0** and `M` measured **≤ 0**, `Y` is strictly better
  unless `R ≥ S`.
* `0943p` is **dominated** by `0943cflat` (same `R`, plus `C`) and should never be ticked.

⇒ **Both slots now carry `C`, and between them they hold `S` and `R` — the one measured gain and the
one documented public-fit hedge. Nothing measured-positive is dropped to buy diversification.**

## 4. Browser steps — still a browser action, re-verified

⚠️ **Do not pick by score. The description text on the Submissions page is the only discriminator.**

1. Open the competition's **Submissions** page.
2. **Untick the auto-ticked default** — its description begins `0943 speedy k94 v2 …` (ref `56568520`).
3. Tick the row whose description begins **`0943cfmsft2 - 0943cf + our msft2 CoAtNet SWA`** (ref `56814081`).
4. Tick the row whose description begins **`0943cflat v1 - cn0 member + C81 fracture fix + FLATTENED outer routing`** (ref `56613303`).
5. Do **not** tick `56570144` (`0943f`), `56570151` (`0943s`), `56568520` (`0943 speedy`) — all 0.943 and
   all carrying less measured evidence than the primary — nor `56599159` (`0943p`, dominated).
6. **Re-check before the deadline.** The tick state cannot be read through the API (no `selected` field on
   `ApiSubmission`; the CLI has no `select` verb; legacy `/api/v1` returns 401 to API-key auth — all three
   verified 2026-10-03 on kaggle 2.1.2 **and** the 2.2.2 wheel), so that page is the only place the state exists.

## 5. The one thing that can still change this

**`56845973` (`0943cfmsft2m-men`) is PENDING.** If its meniscus specialist produces a real gain — read as
**in-block percentile better than 14.07 %** against the before-snapshot
(`state/lb_snapshot_2026-10-05T075916Z.json`: rank 518, line 519, **margin +1**, block 376..1389, in-block
**14.02 %**) — then `0943cfmsft2m-men` becomes a candidate and this sheet must be revised again.

⚠️ **The two arms above stay valid regardless**, because the pair's value is the better of the two and
neither of them is `cfmsft2m`.

## 6. ⭐ AMENDMENT same day — the second slot can be strictly improved by one cheap arm

§3 recommends `(cfmsft2, cflat)`. But §1 established that **the rank measures our BEST submission**,
which means `0943cflat` is built on `0943cf` — **an arm that is no longer our best.** Its whole content
is *"the `0943cf` stack plus flattened routing"*, and `0943cf` is two measured results behind.

**So apply the 10-04 sheet's own hedge argument to the arm that is actually our best:**

```
Y  = (cfmsft2, cflat)        = C + max(S, R)
Y' = (cfmsft2, cfmsft2flat)  = C + S + max(0, R)        <- and Y' >= Y for EVERY sign of R
```

With `C` = cn0 + fracture fix, `S` = msft2 view, `R` = routing flattening. In `Y` the routing hedge
**replaces** `S`; in `Y'` it **stacks on** it. That is the identical argument the 10-04 amendment made
when it replaced `0943p` with `0943cflat` — it was simply never re-applied after `cfmsft2` became best.

### Built, and the one-axis claim is PROVEN, not asserted

`kernels/kernel-0943cfmsft2flat` (`scripts/build_msft2_view_arm.py --flat`). **Not pushed.**

Cell-by-cell diff against `kernel-0943cfmsft2` (`work/diff_flat_vs_cfmsft2.py`):

```
cells: cfmsft2=51  cfmsft2flat=51
cells that differ: [2]
--- cell 2: 2 changed line(s) ---
    -PRESET = os.environ.get("RSNA_PRESET", "speedy")
    +PRESET = os.environ.get("RSNA_PRESET", "parent")
VERDICT: every changed line mentions RSNA_PRESET. Claim holds.
```

And the 2×2 that shows it is the *same* axis `cflat` uses:

| arm | preset | `msft2-ours` view | `_RAD_EXCLUDE` |
|---|---|---|---|
| `0943cf` | speedy | 0 | `("Baker's",)` |
| `0943cflat` | **parent** | 0 | `("Baker's",)` |
| `0943cfmsft2` | speedy | 1 | `("Baker's",)` |
| **`0943cfmsft2flat`** | **parent** | **1** | `("Baker's",)` |

### ⚠️ What this does NOT yet have

* **It has never been run.** One kernel execution (~the cost of `cfmsft2`) plus one submission.
* The `parent` preset flattens `coatnet_w` **at runtime**, so the literal `"coatnet_w": {"__default__": 0.65`
  is still present in the source. ⚠️ **Do not "verify" this arm by grepping for the literal** — that is
  exactly the trap `scripts/build_0943_addmember.py` L214-216 records. Verify it the way `preflight_0943.py`
  does: on the preset name, or better, on the runtime log of a real run.
* **The pair in §3 remains valid if this arm is never built.** This amendment adds an option; it does not
  replace the recommendation.

### Amended recommendation, contingent on that run

| slot | ref | arm |
|---|---|---|
| **primary** | `56814081` | `0943cfmsft2` |
| **second** | *new ref, once pushed & scored* | **`0943cfmsft2flat`** — strictly dominates `56613303` |
| *(fallback)* | `56613303` | `0943cflat` — use if the new arm is not run in time |

⚠️ **Still true after this amendment: if `56845973` (meniscus) reads a real gain, revise again.** Neither
of the two slots above is `cfmsft2m`, so the pair survives that too.

## 7. What this sheet does NOT claim

* It does **not** rank `0943cfmsft2` above the others **by its displayed score** — all the 0.943s are one
  number, and §"the board cannot rank these arms" still applies to the SCORE. The claim is about the
  **rank instrument**, which is a different and finer measurement, plus the evidence each arm carries.
* It does **not** claim `0943cflat`'s 0.941 is a defect. It is not: `0943cflat` is a **strict superset of
  `0943p`** (identical routing flattening, plus `C`). The 0.001 gap against a ±0.008 board is not evidence,
  and the 2026-10-04 case check on that point stands unchanged.
* It does **not** upgrade the underlying evidence. Every effect here is ≤ 0.003 on a board whose SE is
  ≈ 0.008; the whole decision is made **inside** the resolution floor.
