"""Grep the *source* of a notebook cell, not the raw JSON.

`notebooks/*.ipynb` and `kernel-*/*.ipynb` store each cell's source as a single
JSON string on one physical line, so a plain grep match shows the whole cell
truncated at some arbitrary column. This prints the matching cells with real
line numbers so the result is readable and quotable.

    python scripts/nb_grep.py kernel-train/rsna-knee-train-v1.ipynb plan_cache
    python scripts/nb_grep.py kernel-train/rsna-knee-train-v1.ipynb plan_cache -C 12
"""
from __future__ import annotations


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
import argparse
import json
import re
import sys
from pathlib import Path


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("pattern")
    ap.add_argument("-C", "--context", type=int, default=6,
                    help="lines of context around each hit (default 6)")
    ap.add_argument("-l", "--cells-only", action="store_true",
                    help="only list which cells match, do not print lines")
    ap.add_argument("--cell", type=int, default=None,
                    help="dump one code cell in full by index (ignores pattern)")
    ap.add_argument("--list", action="store_true",
                    help="list every code cell with index, size and first line")
    ap.add_argument("--lines", default="",
                    help="with --cell, print only this 1-based inclusive range, e.g. 715:800")
    args = ap.parse_args()

    path = Path(args.notebook)
    if not path.is_file():
        print(f"no such notebook: {path}", file=sys.stderr)
        return 1

    nb = json.loads(path.read_text(encoding="utf-8"))

    if args.list:
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            lines = [l for l in cell_source(cell).split("\n") if l.strip()]
            head = lines[0][:84] if lines else "(empty)"
            print(f"cell[{i:3d}] {len(lines):4d} lines | {head}")
        return 0

    if args.cell is not None:
        cells = nb.get("cells", [])
        if not (0 <= args.cell < len(cells)):
            print(f"cell index {args.cell} out of range (0..{len(cells) - 1})",
                  file=sys.stderr)
            return 1
        lines = cell_source(cells[args.cell]).split("\n")
        lo, hi = 1, len(lines)
        if args.lines:
            a_, _, b_ = args.lines.partition(":")
            lo = int(a_) if a_ else 1
            hi = int(b_) if b_ else len(lines)
        print(f"===== cell[{args.cell}] lines {lo}..{hi} of {len(lines)} =====")
        for k in range(lo - 1, min(hi, len(lines))):
            print(f"{k + 1:4d}| {lines[k]}")
        return 0

    rx = re.compile(args.pattern)
    total = 0

    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        lines = cell_source(cell).split("\n")
        hits = [n for n, ln in enumerate(lines) if rx.search(ln)]
        if not hits:
            continue
        total += len(hits)
        print(f"\n===== cell[{i}]  {len(hits)} hit(s) =====")
        if args.cells_only:
            continue
        shown: set[int] = set()
        for n in hits:
            lo = max(0, n - args.context)
            hi = min(len(lines), n + args.context + 1)
            for k in range(lo, hi):
                if k in shown:
                    continue
                shown.add(k)
                mark = ">>" if k == n else "  "
                print(f"{mark} {k + 1:4d}| {lines[k]}")
            print("   ---")

    print(f"\n[{total} matching line(s) in {path.name}]", file=sys.stderr)
    return 0 if total else 2


if __name__ == "__main__":
    raise SystemExit(main())
