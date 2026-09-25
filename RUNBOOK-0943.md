# RUNBOOK — the four 0.943-line arms

**State:** all four arms are built and pass preflight (`PREFLIGHT OK`, 0 problems / 0 warnings —
`state/preflight_4arms.txt`). They are blocked on one thing only: the weekly GPU quota
(`refreshAt 2026-09-26T00:00:00Z`). A GPU kernel **cannot even be pushed** while quota is exhausted
(FACTS A50), so arms must be pushed the moment quota returns.

**Why a written runbook:** a mistake here costs a whole ~8 h session. Everything below is
copy-pasteable, and the order matters — in particular the acceptance test in step 3 comes BEFORE the
submission, because a run can complete "successfully" and still be 0.003 worse for a reason the log
reveals.

| arm | preset | windows | `_RAD_EXCLUDE` | role |
|---|---|---|---|---|
| `kernel-0943` | `speedy` | 94 | published | **the 0.943 run itself — highest expected** |
| `kernel-0943f` | `speedy` | 94 | `("Baker's",)` | our one-token fracture fork (+0.001 internally) |
| `kernel-0943o` | `outer060` | 94 | published | flat-0.60 outer; a public analysis recommends a flat-0.60 version as the **second final pick** (the per-label outer map was tuned on the public split) |
| `kernel-0943s` | `sparse` | 62/42 | published | **the quota lever** — the arms' native window count; ~90 min vs an 8 h budget, at some cost in score (94 windows is one of the 0.942→0.943 deltas) |

Rebuild any of them at any time with `python scripts/build_0943_arms.py` (one variable per arm, with
exact-match assertions; the builder itself is versioned in `scripts/`).

---

## 0. FIRST: record the quota **before** the run

The single open question that decides the whole week: **is quota charged per wall-clock hour or per
GPU-hour?** The runs use **2×T4** (`device_count() == 2`), so an 8 h run may cost 8 h or **16 h** — and
the weekly budget is 30 h, i.e. ~3 arms versus ~1.5.

```
quota before = ____________          (read the Kaggle web UI; CLI 2.1.2 has no `kaggle quota`)
```

Do **not** queue a second arm until this is measured. (§158.5)

## 1. Re-run the preflight (sources can change while we wait)

```powershell
python scripts/preflight_0943.py          # must end with: PREFLIGHT OK
```

## 2. Push arm 1 — the 0.943 run itself

```powershell
kaggle kernels push -p kernel-0943 -t 34200
kaggle kernels status ppxl16/rsna-knee-0943
```
`-t 34200` (9.5 h) because the notebook carries `TIME_BUDGET = 8.0 * 3600` internally and runs each
model branch as a subprocess with the *remaining* budget as its timeout. Too small a `-t` turns a
would-be submission into a `TimeoutError`.

## 3. On COMPLETE — **acceptance test before submitting**

The pipeline **degrades silently**: with `model_sources`/`kernel_sources` missing it still runs to
completion, logs `dropped -- costs one vote, not the run` twenty times, and scores **0.937** instead
of 0.940. So check the log, not the vibe.

```powershell
kaggle kernels logs ppxl16/rsna-knee-0943 | Out-File -Encoding utf8 state\logs_0943.json
python scripts/klog.py state\logs_0943.json > state\logs_0943.txt
Select-String -Path state\logs_0943.txt -Pattern 'fingerprint matches within' | Measure-Object   # want >= 20
Select-String -Path state\logs_0943.txt -Pattern 'dropped'                                     # want ZERO
```

**Only if that passes**, submit:

```powershell
kaggle competitions submit rsna-knee-abnormality-detection -k ppxl16/rsna-knee-0943 -m "0943 speedy 94win"
```
It is a **code competition**: submit the *kernel* (`-k`), and Kaggle reruns it on the hidden test.
The public board's visible test is **3 studies** — do not over-read a small difference there.

## 4. Read the score back, and record the quota again

```powershell
python scripts/lb_ledger.py
quota after  = ____________      -> cost of one arm = after - before
```

This answers §0 experimentally. It also decides step 5: if one arm cost ~8 h, we can fit two more
this week; if it cost ~16 h, we can fit at most one.

## 5. Arm 2 — choose by what step 4 said

* **If quota is plentiful** (`sparse` is cheap): run `kernel-0943s` (**sparse**) — it isolates what the
  94-window sampling contributed, and it is the cheap one.
* **If quota is tight**: pick exactly one of
  * `kernel-0943o` (**outer060**) — the public analysis's recommendation for the **second final pick**,
    because the per-label outer map was tuned on the public split and `Lateral Meniscus = 1.00`
    discards three of that column's four stages; or
  * `kernel-0943f` (**fracture fork**) — our own +0.001, strong internal evidence but never measured on
    our submission.
  The choice is a judgement call between *robustness against public-LB overfitting* (`outer060`) and
  *a measured internal gain* (`0943f`); both are submission arms, so the leaderboard will settle it.

```powershell
kaggle kernels push -p <arm dir> -t 34200
# then repeat steps 3 and 4 for that arm
```

## 6. Endgame — the final two picks (deadline 2026-10-22)

Current picks: `56166132` (0.941) and `56175178` (0.940). After the new readouts, choose the best two
from the full ledger, **including at least one flat-0.60-class version** if the per-label-map version
turns out to be overfitted to the public split.

## Notes / gotchas learned the hard way

* **`rc=0` is not evidence of training.** A missing `--labels` makes a trainer print
  `No training labels found` and exit **0**. Judge by log content, not exit codes.
* **`Repository not found` on a private GitHub repo = wrong credential**, not a missing repo.
* **A CRLF `.sh` shipped to Linux fails** with a stray-carriage-return error; `.gitattributes` pins LF.
* **Never** use PowerShell `Set-Content -Encoding utf8` on files read by tools — it writes a **BOM**
  (this bit us three times; a BOM makes the Kaggle CLI's `json.load` fail with
  `Expecting value: line 1 column 1`).
