# RSNA Knee — the efficiency track is a RUNTIME competition, and the tick choice sets both tracks

Written 2026-09-29. Supersedes the open question in `docs/EFFICIENCY-PLAN.md` Step 4.

## 1. Our efficiency standing, measured

`scripts/eff_rank.py` (now encoding-fixed — see §5):

```
efficiency board : 4419 teams
US               : eff#3388 / 4419  (worse than 76.7% of teams)  score 0.943
DateSubmitted    : Sat Sep 26 05:24:23 2026   (56570144 (0943f) -- our then-best score, NOT our latest)
teams at our score: 308, best efficiency rank among them: eff#14
eff#1            : Scott Willis (score 0.958)
```

**308 teams share our 0.943 and the best of them is eff#14 while we are eff#3388.** Same accuracy, a
gap of 3374 places. `Efficiency = AUC / (Benchmark − max AUC) + RuntimeSeconds / 32400`, `Benchmark = 0.5`
measured, denominator negative ⇒ more AUC is better, less runtime is better, and **1 hour = 0.0510
AUC-equivalent**. With the AUC term effectively pinned by our ensemble, **the entire gap is runtime.**

## 2. ⭐ The efficiency board tracks a SELECTED submission — so the ticks set both tracks

The board's `DateSubmitted` is `Sat Sep 26 05:24:23` and matches `56570144` (`0943f`). Since then we
have submitted `56599159`, `56599162`, `56599163`, `56599932`, `56613303` (Sep 27) and the date did not
move. **So it is not "latest".** It matches one of the two auto-ticked submissions (A123 recorded the
auto-tick as `56568520` + `56570144`), and picking *one* of a pair is what a "your selected submission"
rule looks like. ⇒ **The browser ticks are the single control over the efficiency runtime as well as
the main-LB pair**, which makes §3 a joint decision rather than an accuracy-only one.

⚠️ Not yet pinned down: *which* of the two selected submissions it takes (first? best score? latest of
the two?). That is cheap to resolve — re-run `eff_rank.py` after any tick change and watch
`DateSubmitted`. Until then, the robust move is to tick two arms that are **both** fast.

## 3. Measured runtime of our nine arms, and which side of the routing hedge each is on

`scripts/arm_timing.py` reads each arm's own downloaded log and extracts its `[  123.4s]` stamps and its
`[config]` line. Full table (3 **visible** studies):

| arm | total s | preset | outer `coatnet_w` | public AUC |
|---|---|---|---|---|
| **`0943`** | **258.2** | speedy | **published per-label** | **0.943** |
| `0943c_v3` | 270.7 | speedy | published per-label | 0.942 |
| `0943p` | 280.4 | parent | **flat 0.65** | 0.941 |
| `0943h` | 288.7 | halfway | published, scaled ×~0.93 | 0.942 |
| `0943c` | 295.1 | speedy | published per-label | 0.942 |
| `0943s` | 303.3 | sparse | published per-label | 0.943 |
| `0943f` | 332.0 | speedy | published per-label | **0.943** |
| `0943cf` | 337.0 | speedy | published per-label | **0.943** |
| `0943o` | 374.1 | outer060 | published per-label | 0.942 |
| `0943r` | 378.0 | speedy | published per-label | 0.942 |

Spread: **119.8 s (31.7 %)**. Variance floor from existing data: `0943c` (295.1) vs `0943c_v3` (270.7)
are the same arm at two versions — **24.4 s (8.3 %)**, an upper bound on machine/version noise.

### The verified finding

⭐ **`0943` and `0943cf` are on the SAME side of the routing hedge.** Both log
`preset=speedy` with `outer CoAtNet weight per finding: {'ACL': 0.75, 'Medial Meniscus': 0.8,
'Lateral Meniscus': 1.0, 'Lateral OA': 0.75, 'Fracture': 0.75}` — the published per-label routing.
Only **`0943p`** (`preset=parent`, `flat 0.65`) is the other side, and `0943h` is a *dose* between them.

⇒ The current recommendation `0943cf` + `0943p` hedges routing **as intended**. But so would
**`0943` + `0943p`**, at identical public AUC on the primary (**0.943**), with the primary being
**78.8 s faster** on the visible run — 3.2× the 24.4 s noise floor.

### ⚠️ Why this is NOT yet a recommendation to switch

1. **The tick pair `0943cf` + `0943p` was chosen for a private-AUC reason, not a public one.** `0943cf`
   is `0943` **plus the two changes this repo actually has local measurements for** — the rescued `cn0`
   as a fifth Raptor view and the fracture fix (§200). `0943` lacks both. The public board cannot see
   the difference (nine-arm spread 0.002 against ±0.008 noise), but the local measurement can.
2. **So the real trade is ~79 s of runtime against an expected private AUC gain**, and those two are
   *commensurable*: 79 s = `79/32400` = **0.0024** efficiency, while a `+0.001` private AUC gain is
   `0.001/0.459` = **0.0022**. **It is close to a wash on the visible-run numbers** — which is a reason
   to measure, not to guess.
3. ⚠️ **The 1322-study composition is still unmeasured, and it can flip the sign.** Every number above
   is a **3-study** run where ~100 s is fixed overhead (EDA ~29 s, 20 fingerprint re-verifications
   ~40 s, publish ~30 s — `EFFICIENCY-PLAN.md`). `0943cf`'s extra view adds **per-study** cost (A143
   measured Raptor 28.9 → 33.2 s, +15 %, on 3 studies). If the 79 s gap is mostly per-study it scales
   with 1322/3 and the runtime argument becomes overwhelming; if it is mostly machine noise it vanishes.
   **`EFFICIENCY-PLAN.md` Step 1 is what settles this and it is still not done** — no script in the repo
   references `RSNA_COMP_ROOT`.

## 4. What to do about it

1. ⭐ **The one human action is unchanged and still blocking both tracks:** in the browser, untick
   `RSNA knee 0943 - Version 2`, tick **(`0943cf` + `0943p`)** as pre-registered — or, if the runtime
   argument is accepted, **( `0943` + `0943p` )**, which holds the same routing hedge at 78.8 s less
   primary runtime for equal public AUC. **Either way it is a browser action; `kaggle` 2.2.4 has no
   `select` command.** Deadline **2026-10-22 23:59Z**.
2. **Run `EFFICIENCY-PLAN.md` Step 1** — the per-study slope at N = 3, 10, 30 by pointing the pipeline
   at a synthetic test root built from training studies. This is the only measurement that converts §3
   from "suggestive" into arithmetic, and it also says which stage to attack. It is cheap (a few
   minutes of inference) and it is the highest-value RSNA experiment available.
3. **Cut the ~100 s of fixed overhead** regardless (skip EDA in the submission run, stop re-verifying 20
   fingerprints that cannot change between runs, trim publish). Free, zero AUC risk, and it is 0.0031
   efficiency — worth ~30 % of the entire visible-run runtime.

## 5. Two tool bugs fixed here (both were silently eating output)

* `scripts/eff_rank.py` died with `UnicodeEncodeError: 'gbk' codec can't encode character '\u21d2'`
  **after** printing the standings and **before** printing its closing recommendation — i.e. it hid the
  only actionable line. Fixed by reconfiguring stdout/stderr to UTF-8. **Third instance of this same
  class of bug in this workspace** (the others: the ARC starter's `slim_framework.py` and
  `build_notebook.py`).
* `state/logs_0943*.json` — **5 of 10 are not parseable JSON**, because they were captured from a CLI
  call whose stderr was merged into stdout, so they begin with a kaggle version warning. `json.load`
  fails at line 1 column 1 and those arms vanish from any comparison. `arm_timing.py` now locates the
  `[{"stream_name"` payload instead of assuming the file starts with it.
