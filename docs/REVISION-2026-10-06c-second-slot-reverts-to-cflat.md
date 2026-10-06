# REVISION 2026-10-06 (c) — the final pair's second slot reverts to `0943cflat`, and the organisers' own notice is why

**This supersedes the second-slot recommendation in `docs/RESULT-2026-10-05-two-arms-flat-or-worse.md` §3.**
The primary does not change.

---

## 1. What I had missed: the organisers formally warn that prevalence differs across the three splits

The competition's own Data Description carries a section I had saved but not weighed:

> **## Dataset Distribution Notice**
> *"Although efforts have been made to ensure each abnormality is represented in each dataset, **the
> prevalence of abnormalities is not guaranteed to be the same across the training, public leaderboard, and
> final evaluation datasets.**"*

**This is the organisers telling us that a constant fitted on the public split need not hold on the final
one.** And the constants in question are exactly the ones the frontier fitted there: the published
per-label `coatnet_w` routing map and the `rad_cal_w` calibration. Our primary arm uses both.

⚠️ It also independently confirms the discipline this repository has been applying all along — *"只在意公榜的
分数会对公榜过拟合"* — but it upgrades it from a principle to a **documented property of this competition**.

## 2. The case check, done properly this time

Let `S` be the effect of our own msft2 CoAtNet view, `R` the effect of **flattening** the public-fitted
outer routing, and `C` the cn0 member + C81 fracture fix. The pair's value is the **better of its two
arms**. Three candidates exist among arms we have already submitted:

```
A = (cfmsft2, cf)          = max(P(C+S), P(C))          <- my 2026-10-05 recommendation
B = (cfmsft2, cflat)       = max(P(C+S), P(C+R))        <- the 2026-10-04 sheet's second slot
C = (cfmsft2, cfmsft2flat) = max(P(C+S), P(C+S+R))      <- the arm submitted today
```

| | B vs A | C vs B |
|---|---|---|
| `R ≥ 0` | **B ≥ A** (strict when `R > 0`) | **C > B** |
| `S ≥ 0, R < 0` | **B = A** (both `C+S`) | **B = C** |
| `S < 0, R < 0` | A > B only if `R > S` | B > C if `R > S` |

⇒ **`A` beats `B` only in the joint event that *both* our own member and the frontier's routing map are
worse privately — and then only if `R > S`.** In every other case `B` is equal or better.

⇒ **And on the measurements we actually have (`S > 0` from the rank move, `R ≲ 0` from `cfmsft2flat`
= 0.942 < 0.943), `A` and `B` are EXACTLY EQUAL.** That is the whole reason the distinction was invisible:
every number I had compared them on was a **public** number, and publicly they are the same.

**⇒ Switching to `B` costs nothing on any measurement we hold, and buys protection against the one risk the
organisers explicitly name.**

## 3. Why not `C`, which contains `S` and `R` together

`C = (cfmsft2, cfmsft2flat) = C + S + max(0, R)` keeps `S` in **both** arms. It is therefore a concentrated
bet: if `S` is negative privately, both arms are dragged down and there is no floor. `B` keeps `C` — the
best-measured change in this repository — in both arms and gives up `S` in one of them.

**The diversification is bought by giving up `S` in one arm, not by deleting `C`.** That is the distinction
`FINAL-SELECTION-2026-10-04.md` §3 drew, and it is the reason `cflat` was right then and is right again.

## 4. ⭐ Revised final pair

| slot | ref | arm | public | what it protects |
|---|---|---|---|---|
| **primary** | **`56814081`** | `0943cfmsft2` — `C + S` | 0.943 | the only change in this repository with a measured gain |
| **second** | **`56613303`** | `0943cflat` — `C + R`, i.e. **without** the public-fitted routing | 0.941 | the organisers' named risk: public-fitted constants on a differently-distributed final set |

⚠️ The second slot's `0.941` is **not** a defect and not evidence: `cflat` is a **strict superset of
`0943p`** (identical flattening, plus `C`), and `0.002` against a board whose SE is ≈ `0.008` is nothing.
This is the same point `FINAL-SELECTION-2026-10-04.md` made, and it was correct.

⚠️ **`0943p` (`56599159`) remains dominated by `cflat`** (same flattening, without `C`) and must never be
ticked.

## 5. What would change this again

* If `56872230` (the reader leg) reads a real gain — in-block percentile better than **16.03 %** — it
  becomes a candidate and this sheet is revised again.
* If the meniscus or specialist axes were ever reopened — they are not.
