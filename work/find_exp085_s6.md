
---

## 6. ⚠️ Verification: three claims re-checked with real exit codes

Prompted by the observation that PowerShell shows red when it should not — and that every command in
this session was piped through `Select-Object` / `Select-String`, so **the exit code was never checked
once**.

**`Select-Object -First N` — and any early-terminating pipeline stage — KILLS the upstream process, so
PowerShell reports a non-zero exit code.** A truncation is therefore *indistinguishable* from a real
failure. Three load-bearing commands were re-run with output redirected to files:

| command | exit | stderr | verdict |
|---|---|---|---|
| `work/inspect_ckpt.py work/newckpt/fold0.pt` | **0** | empty | ✅ and the output is **bit-identical across two runs** |
| `scripts/lb_ceiling.py` | **0** | empty | ✅ rank 578 / 4692, 0.943, board max 0.961 — as reported |
| `gate_pair.py dt_d27_624 vs d27_528`, 12 games | **0** | 10,480 B | ✅ result **2-10-0 → `d27_528` wins**, and the stderr is *entirely* the OpenSpiel game list that `kaggle_environments` prints on import, twice, with **zero real error lines** |

⇒ **The §1–§5 conclusions stand**, and the two displayed red exit codes were the pipe, not the commands.

⚠️ **But the practice was wrong and is now a rule:**
1. **Redirect to a file and read the file.** Do not truncate a pipeline whose exit code matters.
2. **Print `$LASTEXITCODE` explicitly**, in the same message as the conclusion drawn from it.
3. **Read stderr even when stdout looks right** — and know that this repo's most common stderr is
   benign: `kaggle_environments` prints ~10 KB of OpenSpiel game names on import.
4. **Prefer a reproducibility check** (run twice, diff) over eyeballing.
5. This is the same bug class as the failed import, the dropped `ACTION6` coordinate and the refused
   `kernels_push`: **a silently-swallowed failure reported as success**, one level up. Given how often
   that class has produced a wrong finding here, an unchecked exit code is not a style issue.
