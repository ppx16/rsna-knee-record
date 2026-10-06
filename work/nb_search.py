"""Search an .ipynb's CELLS (not the raw single-line JSON) for a pattern and print the
surrounding text, so we can tell whether the code has a ConvNeXt member path and what it
does when the dataset is absent.

This decides whether our 0.943 fork is silently missing a member the frontier's stack has.
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
import json
import re
import sys
import pathlib

pat = re.compile(sys.argv[1], re.I)
for name in sys.argv[2:]:
    p = pathlib.Path(name)
    nb = json.loads(p.read_text(encoding="utf-8"))
    print("=" * 78)
    print(f"FILE {p.name}   cells={len(nb.get('cells', []))}")
    print("=" * 78)
    hits = 0
    for i, c in enumerate(nb.get("cells", [])):
        src = c.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        for m in pat.finditer(src):
            hits += 1
            a = max(0, m.start() - 320)
            b = min(len(src), m.end() + 320)
            print(f"\n--- cell {i} ({c.get('cell_type')}) match at {m.start()} ---")
            print(src[a:b].replace("\\n", "\n"))
            if hits >= 12:
                break
        if hits >= 12:
            break
    if hits == 0:
        print("  no match in any cell")
