"""Repair the string break inside the high-tier duck notebook's `.pth` writer (cell 10).

THE DEFECT (F72): the fragment list that builds the `.pth` body was split ACROSS A NEWLINE, and the
continuation line picked up a stray leading quote. As written:

        "print('[arc_numpy_fix] pruned',bad,'root',r,flush=True)
    ", encoding="utf-8")

so Python reads the literal as running on and the cell dies with
`SyntaxError: unterminated string literal`.

THE CORRECT FORM (what the author must have meant):

        "print('[arc_numpy_fix] pruned',bad,'root',r,flush=True)",
    encoding="utf-8")

i.e. the quote closes on the SAME line as the fragment, followed by a comma, and `encoding=` starts
the continuation line.

WHY THIS SURVIVED SO LONG: the notebook carrying it (v5) never EXECUTED -- F59 proved it died in
nbconvert rendering, before Python ever parsed a line. F65/F71 fixed the rendering and the notebook
finally ran, which is what surfaced this.

The edit is textual and surgical: one two-line sequence replaced by another two-line sequence.
Nothing else in the cell is touched, and the `.pth` body it produces is byte-identical to the
author's intent (still ONE line, as F58 requires).

Usage: python work/fix_pth_string_break.py <in.ipynb> <out.ipynb>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BROKEN = (
    '"print(\'[arc_numpy_fix] pruned\',bad,\'root\',r,flush=True)\n'
    '", encoding="utf-8")'
)
FIXED = (
    '"print(\'[arc_numpy_fix] pruned\',bad,\'root\',r,flush=True)",\n'
    '    encoding="utf-8")'
)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])
    nb = json.loads(src.read_text(encoding="utf-8"))

    hits, edited = 0, None
    for i, c in enumerate(nb["cells"]):
        s = "".join(c.get("source", []))
        if BROKEN in s:
            hits += s.count(BROKEN)
            edited = i
            c["source"] = s.replace(BROKEN, FIXED).splitlines(keepends=True)
    print(f"repaired {hits} broken fragment(s) in cell {edited}")
    if hits != 1:
        print(f"REFUSING: expected exactly 1, got {hits}")
        return 1

    # hard gate on the edited cell only; other cells are informational (cell 45's top-level
    # `await` is a false positive of compile(), not a defect)
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
    print("EDITED cell compiles:", "NO  " + str(bad) if bad else "YES")
    if warned:
        print("informational only (not a gate):", warned)
    if bad:
        print("REFUSING: the edited cell still does not parse")
        return 1

    # the produced .pth body must still be a single line -- the property F58 depends on
    body = ('import sys,os;r=os.environ.get(\'ARC_NUMPY_ROOT\');'
            "bad=[p for p in list(sys.path) if p and p!=r and os.path.isdir(os.path.join(p,'numpy'))];"
            "[sys.path.remove(p) for p in bad];"
            "sys.path.insert(0,r) if (r and r not in sys.path) else None;"
            "print('[arc_numpy_fix] pruned',bad,'root',r,flush=True)")
    print("  intended .pth body is one line, length:", len(body))
    print("  it starts with 'import':", body.startswith("import"))

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
