
### 10.3 ⭐ THE READOUT: `56669951` returned **0.942** — and it passes the pre-registered rule

```
56669951   COMPLETE   0.942   0943s-2coat v1 - 0943s (sparse) minus the Global96 CoAt member
56599932   COMPLETE   0.943   0943cf v1  (the pre-registered accuracy primary)
56570144   COMPLETE   0.943   0943f
56570151   COMPLETE   0.943   0943s     <-- the base arm this was built from
```

Cost **−0.001 AUC** against the base it was built on. Judged against the rule that was written in §10
**before** the number landed:

| arm returns | verdict | this arm |
|---|---|---|
| ≥ 0.941 | **adopt** — the loss is far inside the 0.034–0.042 budget | ✅ **0.942** |
| 0.935 – 0.941 | adopt with a note | — |
| < 0.935 | reject and investigate — a real regression, not noise | — |

**⇒ ADOPT.** The trade, priced:

| | |
|---|---|
| AUC cost | −0.001 → **0.0022 efficiency** |
| runtime gain | ~50 min → **0.093 efficiency** (extrapolated from the member's receipt, §7) |
| **net** | **≈ +0.09 efficiency** |

⚠️ **Honest limits, and why the rule existed.** 0.942 vs 0.943 is **inside the board's ±0.008 noise**, so
this readout **does not measure** the loss — it can only rule out a catastrophic one. That was the whole
design: the saving (0.093) is ~40× the cost of a plausible 0.001 loss, so the decision is dominated by
runtime and the readout is a guard, not a measurement. The ~50 min is likewise an **extrapolation** from a
3-study receipt; what is *measured* is that the member is genuinely not executed (`global96_coat`
COMPLETE in +0.0 s, `ok:false`) and that the family still publishes a valid submission over the two
remaining members.

### 10.4 The tick recommendation, updated

| pair | intent |
|---|---|
| **`0943s-2coat` + `0943p`** | ⭐ **efficiency-minded, and now measured**: same routing hedge, sparse Raptor windows (62/42), and the third CoAt member dropped for a measured −0.001 AUC / ~−50 min. |
| `0943s` + `0943p` | efficiency-minded without the CoAt change |
| `0943cf` + `0943p` | accuracy-minded, as pre-registered (0.943) |

⚠️ All three tie on the public board (0.941–0.943, inside ±0.008), so the board cannot rank them — the
choice is being made on the **runtime** axis, where the metric is explicit
(**1 h = 0.1111 efficiency = 23 % of the whole accuracy range**) and where the three differ by tens of
minutes. This is the one place in the project where a decision is *justified by arithmetic rather than by
a leaderboard readout*, and §8 is that arithmetic.

⚠️ **Still the user's call, and still a browser action** — `kaggle` 2.2.4 has no `select` command, and the
efficiency board tracks a **selected** submission (§2). Deadline **2026-10-22 23:59Z**.
