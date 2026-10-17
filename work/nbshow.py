"""Print the cells of a notebook without fighting PowerShell quoting.

Usage:
    python work/nbshow.py <notebook>              # index of every cell
    python work/nbshow.py <notebook> 0 1          # full text of cells 0 and 1
    python work/nbshow.py <notebook> --grep 0.95  # every 0.95x mention with context
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def cells_of(path: Path) -> list[tuple[str, str]]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"no such notebook: {path}")
        return 2
    cs = cells_of(path)

    if len(argv) == 2:
        print(f"{path.name}: {len(cs)} cells")
        for i, (t, s) in enumerate(cs):
            head = s.strip().splitlines()[0][:100] if s.strip() else ""
            print(f"  {i:3d} {t:8s} len={len(s):6d}  {head}")
        return 0

    if argv[2] == "--grep":
        pat = argv[3]
        for i, (t, s) in enumerate(cs):
            for m in re.finditer(r".{0,240}" + re.escape(pat) + r".{0,240}", s, re.S):
                print(f"--- cell {i} ({t}) ---")
                print("   " + " ".join(m.group(0).split())[:500])
        return 0

    for a in argv[2:]:
        i = int(a)
        t, s = cs[i]
        print(f"########## cell {i} ({t}, {len(s)} chars) ##########")
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
