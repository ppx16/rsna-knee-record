# FIND 2026-10-04 — the medal is decided INSIDE a ~794-team pile-up, which changes how small gains should be judged

Source: `dariushafshar/kaggle-competition-leaderboard-intelligence` — a third-party scrape of this
competition's public leaderboard, 48 observations from 2026-08-17 to 2026-10-02. Pulled and read this
session; the raw panel is in `work/newds/`.

⚠️ **Scope first.** The panel's `our_rank` / `our_score` fields are the **collecting account's**
(`dariushafshar`, whose own last scrape was **0.948 / rank 114**), not ours — the dictionary defines
`entered` as "whether the collecting account had entered the competition". **Nothing below is our
score.** The useful content is the *medal line* and the *tie structure*.

## The trend, as scraped

| scraped (UTC) | public rows | bronze_rank | bronze_score | **teams tied at that score** | tie fraction |
|---|---|---|---|---|---|
| 2026-09-22 21:02 | 4193 | 419 | 0.942 | 259 | 6.2 % |
| 2026-09-26 23:03 | 4402 | 440 | 0.943 | 415 | 9.4 % |
| 2026-09-29 06:26 | 4552 | 455 | 0.943 | 545 | 12.0 % |
| **2026-10-02 05:52** | **4829** | **482** | **0.943** | **794** | **16.4 %** |

`medal_line_basis`, verbatim: `floor(public leaderboard rows * percentage), with minimum ranks gold
10, silver 50, bronze 100; Meta Kaggle CanQualifyTiers=true`. So the bronze line is the top **10 %** of
entered teams, and **the score at that line is 0.943 — the score our stack produces.**

## Why this matters more than the "±0.008 noise" framing alone

Public scores are reported to **three decimals**, so "0.943" is a **0.001-wide bin**. **794 teams sit
in a single 0.001-wide bin, and bronze cuts through it at rank 482** — roughly 312 of those tied teams
will not medal.

⇒ **The private ranking is decided inside an extremely dense band.** Our rank is
`1 + #{teams whose private score exceeds ours}`, so a genuine private gain of δ removes every team in
`[ours, ours+δ)` from that count — and that interval is **densely populated**. A gain far too small for
the public board to resolve can still move us past a large number of teams.

⚠️ **This corrects a claim I made earlier and should not be left standing.** I wrote that the EffB0
leg's +0.0049 was "below the instrument, so it could never be safely preferred over the arm it
duplicates". **That is right about *measurement* and wrong about *ranking*.** Inside a 794-team band,
a small change with a stable sign has real expected rank value even though the board cannot confirm it.

**This strengthens the rationale for the MCL arm already submitted** (`56791368`): +0.000904 macro,
**negative in 0/35** source×assembly combinations, direction pre-registered by an independent 58-study
audit. "Small but sign-stable" is exactly the bet this band rewards — and under the old
"below resolution ⇒ pointless" framing it would have been dismissed.

## ⚠️ The honest limits — the band is a PROXY, not the verdict

1. **The band is public; the medal is private.** Private scores are computed on the other ~70 % of the
   test set, so a 794-team public tie will **scatter** privately. The band tells us the *density to
   expect*, not the ordering.
2. **The medal line is the collector's derivation**, not a quoted Kaggle tie-break rule. I could not
   verify Kaggle's own tie-break text: `kaggle.com` is unreachable from this sandbox (DNS → non-public
   IP), and neither the API nor the panel carries the rules text.
3. **`bronze_rank` grows with the field**, so the line is not fixed; at 4829 rows it is 482, and more
   teams will enter before 2026-10-22.
4. **The tie fraction is rising fast** (6.2 % → 16.4 % in ten days). If that continues, the band gets
   deeper and the value of small real gains goes **up**, but so does the value of the tie-break's
   opacity — we cannot influence it (no `select` API; the tick is a browser action).

## What this changes, concretely
* **Do not dismiss a small, sign-stable improvement as "below the noise".** That reasoning was applied
  to EffB0 and it was applied too broadly. Judge such a change on **expected sign and stability**, not
  on whether the public board can see it.
* **Nothing about the final-selection recommendation changes.** `0943cf` is still the primary. The
  private medal is decided in a dense band; the arms are 0.941–0.943 publicly and the recommendation
  already hedges the one axis documented as public-fitted.
* ⚠️ It does **not** revive EffB0: that line is closed because the leg **would not run** (the rebuild
  produced `0 slot-series`, so its own gate rejects regardless of budget) and because its headline
  rationale was **falsified** — not merely because its effect was small.
