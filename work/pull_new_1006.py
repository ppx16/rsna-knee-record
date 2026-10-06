"""Pull the notebook that claims LB 0.944, and the other new arrivals worth reading."""

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
import pathlib

from kaggle.api.kaggle_api_extended import KaggleApi

OUT = pathlib.Path("work/newpub_1006")
OUT.mkdir(parents=True, exist_ok=True)

REFS = [
    "goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944",
    "kiritohayate95/rsna-knee-train-dinov2-part-1",
    "chenchinglong/rsna-kad-bench-0942",
    "matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed",
]

api = KaggleApi()
api.authenticate()
for ref in REFS:
    slug = ref.split("/", 1)[1]
    dst = OUT / slug
    dst.mkdir(parents=True, exist_ok=True)
    try:
        api.kernels_pull(ref, path=str(dst), metadata=True, quiet=True)
        print("pulled %s" % ref)
    except Exception as e:  # noqa: BLE001
        print("PULL FAILED %s: %s: %s" % (ref, type(e).__name__, str(e)[:140]))
        continue
    for p in sorted(dst.rglob("*")):
        if p.is_file():
            print("   %9d  %s" % (p.stat().st_size, p.relative_to(dst)))
