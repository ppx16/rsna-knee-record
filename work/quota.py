"""Read the LIVE Kaggle accelerator quota and append it to state/quota_log.tsv.

⚠️ TWO ENVIRONMENTS, NOT INTERCHANGEABLE (scripts/audit_state.py §54 has the full story):
  * `python work/*.py`            -> D:\\python        (numpy/sklearn, but `kaggle quota` fails here)
  * `kaggle quota` + credentials  -> D:\\kaggle\\venv-kaggle\\Scripts\\kaggle.exe
Running `kaggle quota` under the wrong interpreter does not error loudly -- and under the right
one it still dies intermittently with SSLEOFError during OAuth, which is transport, not a verdict.

⚠️ AND THE UNITS MATTER, WHICH IS THE WHOLE REASON THIS FILE EXISTS.
Kaggle's weekly GPU allowance is **30.00 h**. A figure of "21600 s" is 6 h and is the **per-session
limit**, not the weekly quota. Quoting the session limit as the weekly quota made GPU look exhausted
(32 min left) when 24.54 h were actually available, and that wrong number was used to reject
experiments. Always print `remaining / total` together, never a bare seconds count.
"""

# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded
# `print("... \u26a0 ...")` raises UnicodeEncodeError at that line and kills the script MID-REPORT, while
# everything printed earlier is already on screen -- so the run looks complete. Self-contained on purpose:
# it imports `sys` itself and depends on nothing else in this tree.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream: leave it alone, prints still degrade
    pass
# ----------------------------------------------------------------------------------------------------
import datetime as dt
import subprocess
import sys

KAGGLE_EXE = r"D:\kaggle\venv-kaggle\Scripts\kaggle.exe"
LOG = "state/quota_log.tsv"


def read_quota(tries=5):
    for i in range(tries):
        r = subprocess.run([KAGGLE_EXE, "quota"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode == 0 and "SSLError" not in out and "Traceback" not in out:
            return out
        print(f"  attempt {i+1}: transport error, retrying", file=sys.stderr)
    return None


out = read_quota()
if out is None:
    print("COULD NOT READ QUOTA (all attempts failed) -- this is NOT 'no quota left'")
    sys.exit(1)

print(out)
rows = [ln.split() for ln in out.splitlines() if ln.strip().startswith(("GPU", "TPU"))]
iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
gpu = next((r for r in rows if r[0] == "GPU"), None)
tpu = next((r for r in rows if r[0] == "TPU"), None)
note = f"live  GPU rem {gpu[2]}/{gpu[3]}  TPU rem {tpu[2]}/{tpu[3]}" if gpu and tpu else "live read"

with open(LOG, "a", encoding="utf-8") as fh:
    fh.write(f"{iso}\t{out.strip().replace(chr(10), ' | ')}\t{note}\n")
print("appended to", LOG)
