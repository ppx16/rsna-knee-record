"""Fetch ONLY the kernel log (the full-output download drags unrelated working-dir files and dies).

Then verify from the RUN'S OWN log that:
  (a) the msft2 view was actually mounted and run, and
  (b) the outer routing was actually FLATTENED at runtime -- the thing the source cannot show.
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
import inspect
import pathlib
import shutil
import time

from kaggle.api.kaggle_api_extended import KaggleApi

REF = "ppxl16/rsna-knee-0943cfmsft2flat"
SLUG = "rsna-knee-0943cfmsft2flat"
OUT = pathlib.Path("work/klog_flat")

api = KaggleApi()
api.authenticate()
print(inspect.signature(api.kernels_output))

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)

ok = False
# ⚠️ the parameter is `file_pattern`, NOT `file_name` -- passing the wrong kwarg raises TypeError,
# which my first version swallowed into a retry loop and reported as a *transport* problem.
for name in (f"{SLUG}.log", "*.log", None):
    for attempt in range(1, 4):
        try:
            res = api.kernels_output(REF, path=str(OUT), file_pattern=name, force=True, quiet=True)
            print("   ->", res if not isinstance(res, tuple) else (len(res[0]), res[1][:80]))
            ok = True
            break
        except Exception as e:  # noqa: BLE001
            print("  fetch(%s) attempt %d failed: %s: %s"
                  % (name, attempt, type(e).__name__, str(e)[:110]))
            time.sleep(5)
    if ok:
        print("fetched with file_pattern=%r" % name)
        break

files = sorted(p for p in OUT.rglob("*") if p.is_file())
print("files: %d" % len(files))
for p in files:
    print("   %9d  %s" % (p.stat().st_size, p.relative_to(OUT)))
