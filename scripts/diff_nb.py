"""Diff two notebooks cell by cell.

Built for the 2026-10-10 check on rabari9999/rsna-knee-sota-apex-grandmaster-0-954:
its file is only 340 bytes larger than the triple-reader it claims to improve on, so the
question is whether the PIPELINE changed at all or only the title inflated.

Usage: python work/diff_nb.py <notebook-a> <notebook-b> [--full]
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path


def cells(p: Path) -> list[tuple[str, str]]:
    nb = json.loads(p.read_text(encoding="utf-8"))
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    a, b = Path(argv[1]), Path(argv[2])
    ca, cb = cells(a), cells(b)
    print(f"A {a.name}: {len(ca)} cells")
    print(f"B {b.name}: {len(cb)} cells")
    if len(ca) != len(cb):
        print("!! different cell counts")

    changed = []
    for i in range(max(len(ca), len(cb))):
        ta, sa = ca[i] if i < len(ca) else ("<none>", "")
        tb, sb = cb[i] if i < len(cb) else ("<none>", "")
        same = sa == sb
        print(f"  cell {i:2d} {ta:8s} {len(sa):7d} vs {len(sb):7d}  "
              f"{'SAME' if same else 'DIFF(+%d)' % (len(sb) - len(sa))}")
        if not same:
            changed.append(i)

    if not changed:
        print("\n==> notebooks are cell-for-cell IDENTICAL in content")
        return 0

    print(f"\n==> {len(changed)} changed cell(s): {changed}")
    for i in changed:
        ta, sa = ca[i] if i < len(ca) else ("<none>", "")
        tb, sb = cb[i] if i < len(cb) else ("<none>", "")
        print("\n" + "=" * 90)
        print(f"### cell {i} ({ta}) diff")
        d = list(difflib.unified_diff(sa.splitlines(keepends=True), sb.splitlines(keepends=True),
                                      fromfile="A", tofile="B", n=1))
        cap = len(d) if "--full" in argv else min(len(d), 80)
        sys.stdout.write("".join(d[:cap]))
        if cap < len(d):
            print(f"\n... ({len(d) - cap} more diff lines; pass --full to see them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
