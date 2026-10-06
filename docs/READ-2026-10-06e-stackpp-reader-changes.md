# READ 2026-10-06 (e) — Stack++'s reader changes, extracted from its code and judged

Purpose: the TTA variant is **pre-registered but not built** (`docs/VERIFY-2026-10-06-reader-leg-is-clean.md`).
Its prereg makes two claims that decide whether it is admissible at all. Both are now checkable against the
author's actual code rather than against its prose — read from
`work/newpub_1006/rsna-knee-dinosaur-v5`, which is where the changes live.

---

## 1. ✅ The admissibility claim holds: the code contains NO leaderboard reference

The prereg's argument was that TTA and rank aggregation are *variance reduction, not tuning against the
public split*, which is what makes them admissible while a msft2 weight sweep (deliberately not done) is not.

```
leaderboard / public_lb / tuned references in Stack++ reader code: 0
```

⇒ **The change set never consults the leaderboard.** The claim stands.

## 2. What actually changes — five items, not the two the prereg named

| # | change | code evidence |
|---|---|---|
| 1 | folds × TTA aggregated in **rank** space, not by probability mean | `_rank_members(members)`; `members` = `len(models) * len(TTA_MODES)` |
| 2 | slice-grid TTA modes | `TT A_MODES` from `RSNA_READER_TTA`; the caller sets `"mid,edge"` when ≤ 3 checkpoints, else `"mid"` |
| 3 | **overlapping** windows | `TTA_OVERLAP = True` → `window_starts` returns `range(n_slice - group + 1)` instead of non-overlapping starts. A large increase in window count. |
| 4 | per-label window pooling | `PUBLIC_FRONTIER_TARGET_POOL = {'Fracture':'max','Contusion':'max','Medial Meniscus':'max','Lateral Meniscus':'max','ACL':'top2','MCL':'top2',"Baker's":'max'}` — then `TTA_TARGET_POOL = {**that, 'Synovitis': 'original_mean'}` |
| 5 | ⭐ **per-label BLEND weight** | the final print is `"Synovitis_w=0.08 other_w=0.30"` — the reader's contribution to Synovitis is cut to **0.08** while everything else stays at **0.30** |

⚠️ **Item 5 is a mechanism I had not recorded from any earlier reading**, and it is the most interesting one:
it is a *per-label* blend weight, motivated by the same observation `goodpjw2008`'s panel 3 makes — that
report-derived labels teach **Synovitis** wrongly because most reports never mention it.

Item 4 also matters: **`Fracture`, `Contusion`, `Medial Meniscus`, `Lateral Meniscus` and `Baker's` are pooled
by `max` over windows, and `ACL`/`MCL` by `top2`.** Our stack does have max/top-2 pooling for focal findings,
so this is not new capability — but it is the first place we have seen the *complete* per-label mode table.

## 3. ⚠️ And the weak point, stated plainly

Items 4 and 5 **are** choices, and their justification is the **58 gold studies** — whose paired bootstrap
CI is **±0.021** (`FACTS` D16). A per-label weight difference of `0.30 → 0.08` fitted on 58 studies is
**inside that interval's shadow**.

⇒ **So Stack++ is admissible but weakly supported**: the aggregation changes (1–3) are parameter-free, while
the per-label choices (4–5) rest on the same 58-study instrument this repository has repeatedly found too
noisy for arm-to-arm comparison.

⚠️ **And Stack++ still reports no leaderboard score** — by its own words it is *"an improvement experiment,
not a claimed leaderboard score until it is submitted"*.

## 4. What this changes

**Nothing is fired.** The gate is unchanged: the TTA variant only runs if the **base** reader leg
(`56872230`) moves the in-block percentile by ≥ 3 points. But two things are now settled:

* the prereg's admissibility claim is **verified against code**, not assumed;
* the exact delta is known — five items, of which 1–3 are free and 4–5 carry a 58-study caveat.

⚠️ **If it is fired, item 5 should be the FIRST thing varied**, because it is the only item that changes a
fitted quantity, and the honest prior for a 58-study-fitted per-label weight is weak.
