
## 10. Lever A built, run, and verified — and what a submission can and cannot tell us

`scripts/build_2coat_arm.py` builds `kernels/kernel-0943s-2coat` from `kernel-0943s` (the **sparse,
0.943** arm) by raising inside the Global96 child's existing `try`, which reuses the block's own
fail-soft path. Pushed as **`ppxl16/rsna-knee-0943s-2coat`**; it ran and **COMPLETED**.

**The skip is exactly free**, from the arm's own `[phase]` records:

```
residual_coat   START 142.6  COMPLETE 197.9   ok: true     (+55.3)
global96_coat   START 197.9  COMPLETE 197.9   ok: false    (+ 0.0)   <- not run
d4              START 197.9  COMPLETE 239.8   ok: true     (+41.8)
[coat-arm] Global96 CoAt child FAILED; continuing without it (flagged): RuntimeError
[coat-arm] family reduction: rank_of_member_probability_mean
[coat-arm] CoAt family members: resgated_top3, d4_swa3
COMPLETE: /kaggle/working/submission.csv, 3 studies, SHA256=8f0c5e7b2b561538dae55439a4d2e9173141096da3defd9bc0a4bac351eef37a
```

with `DINOv2 fail-closed gate PASS: 20/20 fingerprints and members`. So: the member is genuinely not
executed, the family reduces over the **two** remaining members by `rank_of_member_probability_mean`
(the 0.943 method, applied to two), and the whole submission graph still publishes. Summing the arm's
stages gives **236.2 s against `0943cf`'s 290.1 s on the same 3 visible studies — a ~54 s saving, which is
essentially the whole `global96_coat` phase (52.5 s)**.

⚠️ **That 54 s is the FIXED part of the member, not the part that matters.** On a 3-study run the member
is dominated by its `pip install` of a private `timm-1.0.22` wheel and three checkpoint loads. The saving
that transfers to the hidden rerun is its **per-study** cost, which §7 puts at ~4.5 s/study across two
shards ⇒ **~50 min at 1322 studies ⇒ 0.093 efficiency.**

### What the submission readout can and cannot decide

⚠️ The public board's noise is **±0.008** and the expected AUC cost of dropping the *third* member is
**~0.001–0.005**. So a single readout **cannot measure the loss** — it can only catch a **catastrophic
regression**. That is still the right thing to spend a free submission on, because the decision rule from
§9 is asymmetric:

| arm returns | verdict |
|---|---|
| ≥ 0.941 | adopt — the loss is far inside the 0.034–0.042 budget, and ~50 min is banked |
| 0.935 – 0.941 | adopt with a note; the loss is at or below the board's resolution |
| < 0.935 | **reject and investigate** — that is a real regression, not noise |

This is the honest framing: the submission is a **guard, not a measurement**. The measurement that would
actually price the member — an honest OOF — does not exist and cannot be built from public artifacts,
because **A84** established they are all checkpoints rather than features (§9).

### 10.1 SUBMITTED

```
ref 56669951   kernel ppxl16/rsna-knee-0943s-2coat v1
message: 0943s-2coat v1 - 0943s (sparse) minus the Global96 CoAt member; ~50 min less runtime on the
         hidden rerun, break-even AUC cost 0.034 (docs/EFFICIENCY-SLOPE.md)
```

⚠️ **The score takes hours to appear** — this is a code submission that is rerun over the hidden 1322
studies, and the arm's projected runtime is ~3 h (against `0943cf`'s ~3.5–4 h). Do not poll it tightly;
read it back with `scripts/watch_scores.py` / `scripts/eff_rank.py` when it lands.

⚠️ **A score alone does not move the efficiency rank.** §2 showed the efficiency board carries a
`DateSubmitted` that tracks a **selected** submission, so this arm only counts for the efficiency track
once it is one of the two ticks. The tick decision is therefore still the user's, and it now has a third
candidate:

| pair | intent |
|---|---|
| `0943cf` + `0943p` | accuracy-minded, as pre-registered |
| `0943s` + `0943p` | efficiency-minded, same routing hedge, Raptor windows back to 62/42 |
| **`0943s-2coat` + `0943p`** | ⭐ the most aggressive: same hedge, sparse Raptor, and the third CoAt member dropped — pending the `56669951` guard readout |

### 10.2 A process lesson worth keeping

`api.kernels_output` pulls **everything** in `/kaggle/working`, including the three pip `--target` env
directories the CoAt children install into. That is **501 files and counting**, so the download is slow
and it died on the documented intermittent `SSLEOFError` **before reaching `submission.csv`**, which made
a successful run look like one that had produced no submission. `scripts/kernel_output_urls.py` — already
in the repo — lists the output files with **signed per-file URLs** and takes a `--pattern`, so the fix is
to fetch `submission.csv` alone (830 bytes) instead of the whole tree. Use that, not `kernels_output`, for
any arm we intend to submit.
