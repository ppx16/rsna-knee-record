# FINAL SELECTION — current sheet, 2026-10-04

**This supersedes nothing; it is a *current* reading of `docs/FINAL-SELECTION.md` (2026-09-28), which
remains the reasoning of record.** That document was written before the last readouts landed and then
read against its own pre-registered table. This sheet only updates the candidate list, because three
more arms exist now.

Deadline **2026-10-22 23:59 UTC**. Two submissions may be selected.

---

## ⛔ The selection is a BROWSER action — verified three ways, do not look for a CLI route
* `kaggle` CLI (2.1.2 and the 2.2.2 wheel both inspected) has **no `select` command**. Its
  `competitions` verbs are list/files/download/submit/submissions/leaderboard/episodes/replay/logs/
  pages/topics.
* `ApiSubmission` exposes **no `selected` field** — only `ref`, `public_score`, `private_score`,
  `status`, `description`, `file_name`, `date`, … **so the current tick state cannot even be read
  programmatically.**
* The legacy `www.kaggle.com/api/v1/...` endpoints return **401** to API-key Basic auth.
* ⚠️ **The auto-ticked default is NOT the recommended pair** (Kaggle auto-selects `0943` + `0943f`,
  A123). It must be changed by hand.

## The arms that are actually in play

| ref | arm | public | what it changes | evidence for that change |
|---|---|---|---|---|
| **`56599932`** | `0943cf` | **0.943** | our `cn0` CoAtNet as a 5th Raptor view **plus** the C81 fracture fix | ⭐ **two independently measured changes**; `cn0` is the only member this repo has ever measured **above** the frontier's own DINOv2 seed members (0.8769 vs 0.8676–0.8697, A130) |
| `56791368` | `0943cfm` | **PENDING** | `0943cf` **+ `MCL` added to `_RAD_EXCLUDE`** | +0.000904 macro on n=3802/210pos, **35/35** source×assembly sign-stable, direction pre-registered by the 58-study audit (which named MCL worst of 12 and was ignored) |
| `56570144` | `0943f` | 0.943 | `_RAD_EXCLUDE` drops `Fracture` | C81: +0.0118 on that column, +0.00099 macro, n=4185 |
| `56570151` | `0943s` | 0.943 | Raptor windows 94 → 62/42 | measured neutral; **an efficiency lever, not a score lever** |
| `56599159` | `0943p` | 0.941 | outer per-label routing **fully flattened** | the frontier's *own* written remedy for a map tuned on "a public split that is a fraction of the test set" — the axis it calls "the likeliest place to give back points privately" |
| `56568524` | `0943h` | 0.942 | same axis, softened halfway | same direction, weaker dose |

## Recommended pair

**Primary `0943cf` = `56599932`.** Unambiguous: joint-highest, and the only arm carrying two locally
measured changes.

**Second slot — one of two, and this is the only real judgement call:**

* **`0943p` = `56599159`** (0.941) — the pre-registered recommendation. It sits on the **opposite side
  of the one knob whose provenance is documented as public-fitted**, so the pair *hedges* that axis
  instead of doubling it.
* **`0943h` = `56568524`** (0.942) — the **pre-registered alternative**, same hedge direction one notch
  weaker. `FINAL-SELECTION.md` itself notes this is "now the more defensible alternative" because
  `0943p` returned the lowest score of the nine.

⚠️ **The 0.002 between them is NOT evidence** — see below.

**If `56791368` (MCL) lands ≥ 0.942**, it becomes a legitimate third candidate for the primary slot on
the same footing as `0943cf` (two measured changes each). If it lands **≤ 0.940**, treat that as a real
regression signal and drop it — the board can catch that, even though it cannot rank 0.941–0.945.

**If `0943cfm2eff` lands with a stage file that DIFFERS from `0943cfm`'s**, the EffB0 leg is live and it
is the strongest candidate measured so far (+0.00403 out-of-fold over four independent folds,
CI [+0.00285, +0.00491], 4/4 positive). ⚠️ **Check this by diffing `_pipeline_stage.csv` against
`56791368`'s — if they are byte-identical the leg did not apply and the kernel is a DUPLICATE that must
not be selected.** That is exactly how the v1 run was caught.

## ⚠️ Why the scores are not the basis for this — the board proved it cannot rank these arms

Nine arms spanning four axes came back across a total spread of **0.941 → 0.943 = 0.002**, against a
public-LB **SE ≈ 0.008**. The 2026-10-03 full-leaderboard pull (`ryanholbrook/rsna-knee-abnormalities-efficiency-lb`,
4745 teams) shows **558 teams sitting at exactly 0.943** and **none of the 240 at ≥0.944 publishing
code**. So the selection is made on **what each arm changes and what evidence that change has**, never
on the displayed score. The one thing the board *can* do is catch a regression: anything **≤ 0.940**
should be dropped even though the nominal gap is inside the noise.

## Steps

1. Open the competition's **Submissions** page.
2. **Untick `RSNA knee 0943 - Version 2`** (the auto-ticked default).
3. Tick **`56599932`** (primary) and **one** of `56599159` / `56568524`.
4. Re-check after any new arm lands: the tick state is not readable from the API, so this is the only
   place it exists.
