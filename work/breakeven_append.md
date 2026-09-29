
## 9. The break-even rule for Lever A (and why the AUC question barely matters)

`state/FACTS.md` **A84** settles the measurement question: the local-CV bottleneck is **missing images**,
and every public artifact is a *checkpoint*, not features or a stack — so `coat_private_alpha` and the
individual CoAt members **cannot be priced on OOF locally**. `STATUS.md` reaches the same conclusion and
notes that *"every knob in the 0.943 recipe is gated on the same missing instrument"*. Lever A therefore
costs **one GPU run plus a free submission** to observe.

But the arithmetic says the observation barely constrains the decision. Setting the efficiency cost of an
AUC loss against the efficiency gain of a runtime saving:

```
ΔE = −2.18341 · ΔAUC + ΔRuntime/32400                (we want ΔE < 0)
break-even:  ΔAUC_budget = (ΔRuntime/32400) / 2.18341
```

| runtime saved | efficiency gained | **AUC we can afford to lose** |
|---|---|---|
| 40 min | 0.0741 | **0.0339** |
| 50 min | 0.0926 | **0.0424** |
| 90 min (1 member + shared prep) | 0.1667 | **0.0763** |

⭐ **So dropping a CoAt member is worth doing unless it costs more than ~0.034–0.042 AUC.** For scale:

* our entire AUC is **0.943**, and the **nine-arm spread across four different axes was 0.002** against a
  public-board noise of **±0.008**;
* one of three members inside a family that enters the final blend at `coat_private_alpha = 0.40` is
  therefore very unlikely to be worth 0.034 — that would be larger than every axis this project has ever
  measured, combined.

⚠️ **Two honest caveats.**
1. The 40–50 min per member is itself an **extrapolation** from a 3-study receipt (§7), not a measured
   1322-study number. The *rule* holds for whatever the real saving is; the *decision* inherits that
   uncertainty.
2. Dropping a member is **not a pure runtime knob** — the family is reduced as the rank of its members'
   mean probability, so removing one changes the blend, not just the cost. That is precisely why the AUC
   cost is not zero, and why the break-even budget above is the thing to judge it against rather than
   "does the board move?" — **the board cannot see a 0.034 change either way.**

**⇒ Recommendation order, priced:** `0943s` (free, already measured at 0.943) → trim the ~100 s of fixed
overhead (free, 0.003) → **drop one CoAt member** (1 GPU run + 1 free submission, expected +0.05–0.09
efficiency net of a plausible AUC cost) → shared DICOM preparation (2.1 GB artifact, riskier).
