"""Build `kernels/kernel-0943s-2coat` — the 0.943 arm with one CoAt member dropped.

THE TRADE, PRICED (docs/EFFICIENCY-SLOPE.md §7-§9)
    The CoAt family is **~2.1-2.5 h** of our ~3.5-4 h hidden rerun, and the efficiency metric prices the
    two terms comparably:

        E(a, R) = -2.18341 * a + R/32400            (lower is better)

    so **1 hour of runtime = 0.1111 efficiency = 23 % of the entire accuracy range from AUC 0.74 to
    0.958**. Dropping one of the three members saves ~40-50 min = 0.074-0.093 efficiency, which is worth
    an AUC loss of

        (0.074..0.093) / 2.18341 = 0.034 .. 0.042

    For scale: our whole AUC is 0.943, the nine-arm spread across four different axes was 0.002, and the
    public board's noise is +-0.008. And Global96 is not an arbitrary member -- it is the THIRD one, added
    in exactly the change that took the arm from **0.942 to 0.943**, so its measured price is ~0.001.
    Dropping it is therefore net positive on the efficiency track unless it is worth 0.034, i.e. larger
    than every axis this project has ever measured, combined.

WHY THIS IS A TWO-LINE PATCH AND NOT A REWRITE
    Cell 45 already supports a missing member: each child runs inside `try/except Exception` that logs a
    `coat_*_child_failed` event, prints "continuing without it (flagged)", and sets `_coat_<m>_ok = False`.
    The family reduction then guards each member with `if _coat_<m>_ok:` and simply does not append it, and
    it only bails out entirely `if not _members`. So RAISING INSIDE THE EXISTING `try` reuses the whole
    fail-soft path: no re-indentation, no new control flow, and the phase records still show
    `global96_coat` completing with `ok: false`, which makes the run self-documenting.

    The raise is unconditional on purpose. This notebook exists only to measure this one change, and an
    unconditional raise has no name lookups and no env plumbing to get wrong.

BASE ARM: `kernel-0943s`, not `0943cf`
    `0943s` is `0943` with one word changed (`PRESET` default "speedy" -> "sparse"), and it scored **0.943
    -- tied for our best** -- while cutting the Raptor stage's per-study window count ~34 %. Building on it
    means this arm differs from a MEASURED 0.943 arm by exactly one thing (the dropped member), so the AUC
    readout is attributable. It also means the arm carries both runtime levers we have measured: the
    sparse Raptor and the two-member CoAt.

Usage:
    python scripts/build_2coat_arm.py
    python scripts/build_2coat_arm.py --push
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "kernels" / "kernel-0943s"
DST = ROOT / "kernels" / "kernel-0943s-2coat"
NB_NAME = "rsna-knee-restructured-version-3.ipynb"
CELL = 45


def cell_source(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def set_source(cell, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    nb = json.loads((SRC / NB_NAME).read_text(encoding="utf-8"))
    cells = nb["cells"]
    src = cell_source(cells[CELL])

    anchor = "    rsna_phase('global96_coat', 'START')\n    try:\n"
    if src.count(anchor) != 1:
        # Print the neighbourhood rather than just failing -- an anchor miss should be one edit to fix.
        i = src.find("global96_coat")
        print("ANCHOR MISS. Context around 'global96_coat':")
        print(repr(src[max(0, i - 120): i + 260]))
        raise SystemExit(f"expected exactly 1 match, found {src.count(anchor)}")

    patch = (
        "    rsna_phase('global96_coat', 'START')\n"
        "    try:\n"
        "        # ---- efficiency arm: SKIP the Global96 CoAt member --------------------------------\n"
        "        # docs/EFFICIENCY-SLOPE.md §9. The CoAt family is ~2.1-2.5 h of the 1322-study rerun.\n"
        "        # Global96 is the THIRD member -- added in the very change that took this arm from\n"
        "        # 0.942 to 0.943 -- so its measured price is ~0.001 AUC, against a break-even budget of\n"
        "        # 0.034-0.042 for dropping a member (1 h = 0.1111 efficiency = 23% of the whole\n"
        "        # accuracy range). Raising here reuses this block's own fail-soft path: the except below\n"
        "        # logs coat_global96_child_failed, sets _coat_g96_ok = False, and the family reduction\n"
        "        # omits the member. Nothing else in the pipeline changes.\n"
        "        raise RuntimeError(\n"
        "            'efficiency arm 0943s-2coat: CoAt member global96 intentionally skipped; see '\n"
        "            'docs/EFFICIENCY-SLOPE.md section 9')\n"
    )
    # ⚠️ The anchor ends at `try:` and the ORIGINAL next line is already
    # `g96_art = find('coatnet_global96_top3_manifest.json', ...)`. The first version of this patch
    # re-emitted that line too, producing a duplicate (harmless after a `raise`, but it signals an
    # imprecise patch and the compile gate cannot see it because duplicated statements are valid Python).
    # Assert the line now appears exactly twice in the cell: once in the patch-shadowed original, once
    # further down where the real work would have happened.
    if src.count("g96_art = find('coatnet_global96_top3_manifest.json'") != 1:
        raise SystemExit("unexpected g96_art occurrences before patch")
    src = src.replace(anchor, patch, 1)
    after = src.count("g96_art = find('coatnet_global96_top3_manifest.json'")
    if after != 1:
        raise SystemExit("patch produced %d g96_art lines; expected 1" % after)
    set_source(cells[CELL], src)

    bad = 0
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        try:
            compile(cell_source(c), f"cell{i}", "exec")
        except SyntaxError as e:
            bad += 1
            print("SYNTAX ERROR cell[%d]: %s (line %s)" % (i, e.msg, e.lineno))
    if bad:
        raise SystemExit("%d cell(s) do not compile; nothing written" % bad)

    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = "ppxl16/rsna-knee-0943s-2coat"
    meta["title"] = "RSNA knee 0943s 2coat"
    meta["code_file"] = NB_NAME
    meta["is_private"] = True

    DST.mkdir(parents=True, exist_ok=True)
    (DST / NB_NAME).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote %s (base 0943s/sparse, global96 skipped, %d code cells, all compile)"
          % (DST.relative_to(ROOT), sum(1 for c in cells if c.get("cell_type") == "code")))

    if a.push:
        from kaggle import api
        api.kernels_push(str(DST))
        print("PUSHED ppxl16/rsna-knee-0943s-2coat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
