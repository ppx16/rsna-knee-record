# -*- coding: utf-8 -*-
"""Dump a notebook's cells as plain text, so a 150 KB .ipynb can be READ.

The research pulls in downloads/research/ are JSON notebooks; reading them raw burns
context on escaping. This prints markdown only (the prose, where the reasoning is),
code only, or everything.

    python scripts/nb_text.py downloads/research/why941/*.ipynb md
    python scripts/nb_text.py path/to.ipynb code | head -200
"""
import json
import pathlib
import sys


def main() -> int:
    # ⚠️ 4th instance of this bug (klog.py, now.py, then here). Windows consoles default to a
    # legacy code page (GBK) and notebooks are FULL of emoji, so `print(src)` dies with
    # UnicodeEncodeError AFTER doing all the work. RULE: any script that prints non-ASCII must
    # reconfigure stdout first, with errors="replace" so reporting can never fail the command.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    p = pathlib.Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "md"
    want = {"md": "markdown", "code": "code"}.get(mode)
    nb = json.loads(p.read_text(encoding="utf-8"))
    total = 0
    for i, c in enumerate(nb.get("cells", [])):
        if want and c.get("cell_type") != want:
            continue
        src = c.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        if not src.strip():
            continue
        total += len(src)
        print("\n===== [%d] %s =====" % (i, c.get("cell_type")))
        print(src)
    print("\n----- %d cells, %d chars shown -----" % (len(nb.get("cells", [])), total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
