# FIND 2026-10-05 — the GPU quota was mis-read by ~20× for two days, and the error was in the DENOMINATOR

**Status: my error, found because the user said "Gpu额度还有很多，你应该是没刷新". They were right.
Nothing had been refreshed wrong — the arithmetic had been wrong all along.**

---

## 1. What was claimed, and what is true

For this whole session I treated GPU as nearly exhausted and used that to reject experiments.
The figure I quoted was **`19664 s / 21600 s`**, which I read as **≈ 32 minutes remaining**.

Live reading, `D:\kaggle\venv-kaggle\Scripts\kaggle.exe quota`, 2026-10-05:

```
resource  used   remaining  total   refreshAt
GPU       5.46h  24.54h     30.00h  2026-10-10T00:00:00
TPU       3.79h  16.21h     20.00h  2026-10-10T00:00:00
```

**24.54 h remaining, not 32 minutes.**

## 2. The precise failure, which is checkable and is NOT a stale cache

```
19664 s / 3600 = 5.46 h   ==  GPU used, live      ✓
21600 s / 3600 = 6.00 h   !=  GPU total (30 h)    ✗
```

**The numerator was right the whole time. Only the denominator was wrong.**
`21600 s` is **6 h**, which is the **per-session** ceiling, not the weekly allowance
(Kaggle's weekly GPU allowance is **30.00 h = 108000 s**).

So the "remaining" was computed as `21600 − 19664 = 1936 s ≈ 32 min` when it should have been
`108000 − 19664 = 88336 s ≈ 24.54 h`. **Same numerator, wrong total, off by ~20×.**

⚠️ This is why "refresh the page" would not have fixed it. **A stale number and a wrong unit look
identical from the outside.** Only recomputing from the live source exposed it.

## 3. Where it leaked into decisions

| where | what it says | effect |
|---|---|---|
| session notes | `GPU 19664 s / 21600 s ⇒ ~32 min left (enough for one commit test)` | used to reject spending GPU |
| `docs/PREREG-2026-10-04-convnext-member.md` §预算 | `GPU 17004 s / 21600 s (6 h) ⇒ ≈1.28 h left` | framed the ConvNeXt member as affordable **only as a fold-0 probe**; the full 5-fold (~13 h) was inside budget all along |

The prereg line is corrected in place, with the wrong number kept visible.
`state/quota_log.tsv` had also been stale since **2026-09-26** — the log existed but nothing was
writing to it, so the only surviving number was whichever one a previous turn had typed into prose.

## 4. The rule this leaves behind

**Never record an accelerator budget as a bare seconds count or a single percentage. Always read it
from `kaggle quota` at the moment of decision, and always print `remaining / total` together** —
the pairing is what makes a unit error visible. A number with a wrong unit is indistinguishable
from a correct number, so the check has to be structural, not attentional.

Two mechanical notes, both already known and both re-confirmed here:

* **`kaggle quota` must be run through `D:\kaggle\venv-kaggle\Scripts\kaggle.exe`**, not the default
  `kaggle` on PATH. The PATH one failed **6/6** attempts with `SSLEOFError` on the OAuth
  `IntrospectToken` call while the venv one succeeded first try — the CLI's OAuth path and the
  Python API's key path are not the same authentication path (`scripts/audit_state.py` §54 says this;
  I had to rediscover it).
* `SSLEOFError` on `api.kaggle.com` remains **transport, not a verdict**. But note the asymmetry:
  a retry wrapper that swallows exceptions can also **manufacture** a false result — see
  `work/check_running.py`, which printed a confident "ACTIVE: NONE" three separate times before it
  was written correctly.

Tooling: `work/quota.py` reads the live quota and appends it to `state/quota_log.tsv`.
