"""Submit the cfmsft2flat kernel version, with a BEFORE snapshot taken first.

⚠️ Why the Python API and not the CLI: measured 2026-09-11 with identical arguments in the same
minute, `kaggle competitions submit -c ... -k ... -v ...` returned
    400 Bad Request .../CreateCodeSubmission
with NO response body (so the cause was invisible), while KaggleApi.competition_submit_code with
the same arguments succeeded (ref 56166132). scripts/director.py L348-360 records this. The CLI
path is not usable for code competitions on this version.

⚠️ `file_name` must be the name of the file the KERNEL wrote (`submission.csv`), not a local path.
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
import json
import pathlib

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
REF = "ppxl16/rsna-knee-0943cfmsft2flat"
VERSION = 1
MESSAGE = ("0943cfmsft2flat - cfmsft2 with the outer per-label routing FLATTENED (preset=parent): "
           "differs from 0943cfmsft2 by exactly one line. Strictly improves the final pair, since "
           "the pair's value is the better of the two and cfmsft2 stays in it")

snaps = sorted(pathlib.Path("state").glob("lb_snapshot_*.json"))
before = json.loads(snaps[-1].read_text(encoding="utf-8")) if snaps else None
print("=== BEFORE snapshot (most recent on disk) ===")
print(json.dumps(before, indent=2) if before else "  (none)")
if before:
    print("  in-block percentile to beat: %.3f%%  (lower is better)" % before["in_block_pct"])
print()

api = KaggleApi()
api.authenticate()
print("submitting %s v%d ..." % (REF, VERSION))
resp = api.competition_submit_code(file_name="submission.csv", message=MESSAGE,
                                   competition=COMP, kernel=REF, kernel_version=VERSION, quiet=True)
print("response:", resp)
ref = getattr(resp, "ref", None) or (resp.get("ref") if isinstance(resp, dict) else None)
print("SUBMISSION REF:", ref)

rec = {
    "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "arm": "0943cfmsft2flat",
    "kernel": REF,
    "version": VERSION,
    "ref": ref,
    "before_snapshot": before,
    "message": MESSAGE,
}
pathlib.Path("state").mkdir(exist_ok=True)
out = "state/submit_%s.json" % (ref or "unknown")
pathlib.Path(out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
print("recorded ->", out)
