# RUNBOOK — the five 0.943-line arms

**State:** all five arms are built and pass preflight (`PREFLIGHT OK`, 0 problems / 0 warnings —
`state/preflight_5arms.txt`). They are blocked on one thing only: the weekly GPU quota
(`refreshAt 2026-09-26T00:00:00Z`). A GPU kernel **cannot even be pushed** while quota is exhausted
(FACTS A50), so arms must be pushed the moment quota returns.

**Why a written runbook:** a mistake here costs a whole ~8 h session. Everything below is
copy-pasteable, and the order matters — in particular the acceptance test in step 3 comes BEFORE the
submission, because "COMPLETE" alone does not tell you *which* code path ran (and, per A73, a bad
mount does not degrade quietly here — it dies — so read the log before you trust the run).

| arm | preset | windows | `_RAD_EXCLUDE` | role |
|---|---|---|---|---|
| `kernel-0943` | `speedy` | 94 | published | **the 0.943 run itself — highest expected** |
| `kernel-0943f` | `speedy` | 94 | `("Baker's",)` | our one-token fracture fork (+0.001 internally) |
| `kernel-0943o` | `outer060` | 94 | published | flat-0.60 outer; a public analysis recommends a flat-0.60 version as the **second final pick** (the per-label outer map was tuned on the public split) |
| `kernel-0943s` | `sparse` | 62/42 | published | **the quota lever** — the arms' native window count; ~90 min vs an 8 h budget, at some cost in score (94 windows is one of the 0.942→0.943 deltas) |
| `kernel-0943h` | `halfway` | 94 | published | ⭐ per-label routing **softened toward its default** — our own C57 evidence says softening cost 0.001 where flattening cost 0.002 (see §5) |

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

⚠️ **Set `PYTHONUTF8=1` first — the push fails without it** (FACTS A77). The notebook's ipywidgets
metadata contains a U+200B zero-width space, and on this zh-CN Windows the CLI opens the `.ipynb`
with the locale codec (gbk) and dies with `'gbk' codec can't decode byte 0x8b in position 3768`.
**That message looks nothing like a quota problem, and the first reading of it was "quota did not
refresh" — it is not. Unlike most of this toolchain's failures, this one is visible in the exit code:
`kernels push` returns **rc=1** here.

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:KAGGLE_API_TOKEN = [Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN','User')
kaggle kernels push -p kernels/kernel-0943 -t 34200   # must print: Kernel version N successfully pushed
kaggle kernels status ppxl16/rsna-knee-0943      # expect KernelWorkerStatus.RUNNING
```
`-t 34200` (9.5 h) because the notebook carries `TIME_BUDGET = 8.0 * 3600` internally and runs each
model branch as a subprocess with the *remaining* budget as its timeout. Too small a `-t` turns a
would-be submission into a `TimeoutError`.

Then hand off to `powershell -File scripts\watch_0943.ps1`, which polls every 6 min (~10 calls/hour,
deliberately far below the ~100/hour that got the read paths 403-banned in A26), fetches the log when
the run reaches a terminal status, and prints the acceptance verdict.

## 3. On COMPLETE — **acceptance test before submitting**

⚠️ **Corrected 2026-09-25 (A73). The 0943 arms are fail-CLOSED, not fail-soft.** The old text here
said a missing source "degrades silently ... logs `dropped` twenty times, and scores 0.937". That
describes the **`kernel-ens`/v2src** line, where a member that fails *at inference* is caught and
"costs one vote, not the run" (`kernel-ens` line 1520). The five arms in this runbook are built on
`maverickss26/...restructured-version-3`, which has **no soft path at all** (`soft-hits = 0` across
all five 094x dirs):

- any member failure ⇒ `raise WeightsError('DINOv2 member failure; no fallback permitted')`
- `_DINOV2_MATCHED_MEMBERS != 20` ⇒ raise (and re-asserted post-A5 at two more sites)
- `find_weights() is None` ⇒ `raise WeightsError('required public checkpoint manifest not found')`
- `main()` is called bare — no enclosing `try/except`

Consequence: a missing or unreadable source means **no `submission.csv` at all**, not a 0.937
consolation. The preflight's source-COMPLETE check is therefore load-bearing, and a failed run is
visible within minutes — do not sit on it for 8 h.

```powershell
kaggle kernels logs ppxl16/rsna-knee-0943 | Out-File -Encoding utf8 state\logs_0943.json
python scripts/klog.py state\logs_0943.json > state\logs_0943.txt
Select-String -Path state\logs_0943.txt -Pattern 'fingerprint matches within' | Measure-Object   # want >= 20
Select-String -Path state\logs_0943.txt -Pattern 'DINOv2 fail-closed gate PASS'                 # want 1
Select-String -Path state\logs_0943.txt -Pattern 'no fallback permitted|WeightsError|Traceback' # want ZERO
Select-String -Path state\logs_0943.txt -Pattern 'COMPLETE: /kaggle/working/submission.csv'      # want 1
```

⚠️ **`dropped` is NOT a clean greps target** — the notebook prints the sentence *"a dropped study is not
a wrong label — it is no label"* as explanation, so a naive `dropped` count is 1 on a **perfectly
healthy** run. Judge by the four markers above (and by the `COMPLETE: .../submission.csv` line, which is
the one that actually says the artifact exists), not by the word.

⚠️ **The decoder can lie; check the byte count first** (FACTS A79). If `state\logs_0943.txt` comes out
absurdly small — a few bytes — the log was not converted, and every count above reads 0, which is
indistinguishable from "the pipeline never ran". A healthy run's text is ~16 KB. The fixed `klog.py`
skips the CLI's leading version warning; if a future `kaggle` version changes that warning's shape,
re-check the byte count before believing a FAIL.

**Only if that passes**, submit:

```powershell
# `-f` (the output file name the kernel produced) and `-v` (the version) are BOTH REQUIRED for a
# code competition.  Plain `-k` alone is rejected client-side with
#   "Code competition submissions require both the output file name and the version number".
kaggle competitions submit rsna-knee-abnormality-detection `
    -k ppxl16/rsna-knee-0943 -f submission.csv -v 2 `
    -m "0943 speedy k94 - 20/20 DINOv2 fingerprints, outer 0.65"
```

⚠️ **If this returns `400 Bad Request`, do not guess** — the CLI discards the body, and a bare 400 has
several very different causes (daily cap, FACTS A8; internet enabled, FACTS A78). Ask the server:

```powershell
python scripts\probe_submit_code.py --kernel ppxl16/rsna-knee-0943 --version 2
```
It prints the raw JSON error. A8's cap reads as a bare 400; A78's internet refusal reads as
`FAILED_PRECONDITION: Your Notebook cannot use internet access in this competition`.

It is a **code competition**: submit the *kernel* (`-k`), and Kaggle reruns it on the hidden test.
The public board's visible test is **3 studies** — do not over-read a small difference there.
Note the visible-side run is only ~5 min because there are 3 studies; the 8 h internal budget is
sized for the hidden 1322, so **a short run is expected and is not evidence of an early exit** —
the acceptance markers above are what tell you it ran.

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
  * `kernel-0943h` (**halfway**) — ⭐ **the best-supported hedge.** It keeps the 0.65 default and pulls
    every per-label override halfway toward it. Our own LB evidence (C57, on this same object — the
    outer CoAtNet routing — in the older v2src line) is that **flattening cost 0.002 while softening
    cost 0.001**; or
  * `kernel-0943o` (**outer060**) — reverts the *default* 0.65→0.60 and keeps every override. A
    different axis from `halfway`, and the public analysis's recommendation for the second final pick
    (the per-label outer map was tuned on the public split, and `Lateral Meniscus = 1.00` discards
    three of that column's four stages); or
  * `kernel-0943f` (**fracture fork**) — our own +0.001, strong internal evidence but never measured on
    our submission.
  ⚠️ `halfway` and `outer060` are **not interchangeable hedges** (one softens the overrides, the other
  lowers the default) — see HANDOFF 163.3. This is a judgement call between *robustness against
  public-LB overfitting* (`halfway`/`outer060`) and *a measured internal gain* (`0943f`); all three are
  submission arms, so the leaderboard will settle it.

```powershell
kaggle kernels push -p kernels/<arm dir> -t 34200
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
