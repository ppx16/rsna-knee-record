"""Render a Kaggle kernel log (JSON-lines of {stream_name,time,data}) into plain readable text."""

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
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   "work/klog_flat/rsna-knee-0943cfmsft2flat.log")
for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.lstrip(",").strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except Exception:
        print(line)
        continue
    d = rec.get("data")
    if d is None:
        continue
    sys.stdout.write(d if d.endswith("\n") else d + "\n")
