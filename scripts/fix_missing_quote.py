"""Add the ONE missing closing quote that makes F58's .pth body parse.

FOUND 2026-10-11, and it is the real reason the high-tier numpy fix never worked:

    line 280|     "print('[arc_numpy_fix] pruned',bad,'root',r,flush=True)
    line 281| ", encoding="utf-8")

The string fragment on line 280 is missing its terminating `"`. Python therefore reads the literal
as continuing into line 281, and the whole cell dies with
`SyntaxError: unterminated string literal`.

WHY NOBODY SAW IT: F59 proved that v5 (the version carrying this fix) never EXECUTED at all -- it
died earlier, in nbconvert rendering (`missing attachment: tufa_labs.png`). So this code had never
been parsed by Python. F71's attachment fix moved the failure into execution, and execution is what
finally surfaced the parse error.

⇒ This is a one-character edit. It changes no logic, no path, no weight -- it makes the existing
   code parse. Everything F67/F69 established about WHERE the .pth must go still applies, so this is
   applied on top of the minimal F69 correction (`patch_pth_min.py`), not instead of it.

Usage: python work/fix_missing_quote.py <in.ipynb> <out.ipynb>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BROKEN = '''    "print('[arc_numpy_fix] pruned',bad,'root',r,flush=True)\n'''
FIXED = '''    "print('[arc_numpy_fix] pruned',bad,'root',r,flush=True)"\n'''


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])
    nb = json.loads(src.read_text(encoding="utf-8"))

    hits = 0
    edited = None
    for i, c in enumerate(nb["cells"]):
        s = "".join(c.get("source", []))
        if BROKEN in s:
            hits += s.count(BROKEN)
            edited = i
            c["source"] = s.replace(BROKEN, FIXED).splitlines(keepends=True)
    print(f"broken fragments found and fixed: {hits} (cell {edited})")
    if hits != 1:
        print(f"REFUSING: expected exactly 1, got {hits}")
        return 1

    # Syntax-check every code cell, but only the EDITED cell is a hard gate.
    # `cell 45: 'await' outside function` is a FALSE POSITIVE -- IPython/papermill execute cells
    # with a top-level await, which `compile(..., 'exec')` cannot model. Refusing on it (as the
    # first version of this script did) meant the real fix was never written to disk at all.
    bad, warned = [], []
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        s = "".join(c.get("source", []))
        if s.lstrip().startswith(("%", "!")):
            continue
        try:
            compile(s, f"cell{i}", "exec")
        except SyntaxError as e:
            (bad if i == edited else warned).append((i, e.msg, e.lineno))
    print("EDITED cell still failing:", bad if bad else "NONE")
    if warned:
        print("other cells that do not compile (informational, NOT a gate):", warned)
    if bad:
        print("REFUSING: the cell we edited still does not parse")
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, ensure_ascii=True), encoding="utf-8")
    raw = dst.read_bytes()
    print(f"wrote {dst}  {len(raw)} bytes")
    print("  ascii-only:", all(b < 128 for b in raw))
    try:
        raw.decode("gbk")
        print("  GBK-decodable: YES")
    except UnicodeDecodeError as exc:
        print("  GBK-decodable: NO", exc)
        return 1
    print("  round-trip cells:", len(json.loads(raw.decode("utf-8"))["cells"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
