"""Resolve a kernel directory wherever the layout happens to put it.

WHY THIS EXISTS
---------------
The 2026-09-26 reorganisation moved every kernel into `kernels/<name>/`.  Kernel paths appear in three
places, and only one of them is code:

  1. literals in scripts -- fixed in place by `_apply_reorg_paths.py`;
  2. names assembled at runtime -- `dst = ROOT / name` -- which cannot be fixed by a literal edit;
  3. **stored data**: `state/cv_targets.json` carries twenty-five `"dir": "kernel-exp/..."` entries.

Rewriting the stored data would leave older configs broken, and it would have to be redone at the next
move.  So the resolution lives here and the callers go through it: `kdir()` returns the grouped
location when it exists and falls back to the old flat one, which means a stale config, an older
checkout, or a path typed by hand all keep working.

    from kpaths import kdir
    dst = kdir(name)              # bare name or relative path, either layout
    dst.mkdir(parents=True, exist_ok=True)

For an output directory that does not exist yet, `kdir()` returns `kernels/<name>` -- the new
location -- which is what a builder wants.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "kernels"


def kdir(name):
    """Kernel directory for `name`, preferring the grouped layout and falling back to flat."""
    p = Path(str(name))
    for cand in (KERNELS / p, ROOT / p):
        if cand.is_dir():
            return cand
    return KERNELS / p
