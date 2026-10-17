"""Fix the "missing attachment" rendering failure that kills duck-harness notebooks.

FACTS: F59 diagnosed that our high-tier notebook (duck-18 v5) did NOT die in boot but in
nbconvert RENDERING:

    nbconvert.filters.markdown_mistune.InvalidNotebook: missing attachment: tufa_labs.png

and that the defect is IN THE AUTHOR'S NOTEBOOK (both the original and our build have zero
embedded attachments while a markdown cell references one). So the fix belongs on our side, and
it is mechanical: either embed a real image attachment, or drop the dangling reference.

Why this matters beyond tidiness: F53 called the high-tier line blocked by "one boot bug"; F57/F58
moved that failure out of boot; F59 shows what is now left is *rendering*. A cell that cannot be
rendered aborts the run, so this single dangling image is plausibly the last thing standing between
us and the whole NVFP4/W4A16 duck line.

Two modes:
  --embed   put a real 1x1 PNG attachment into every referencing cell (keeps the visual placeholder
            slot; nothing else changes)
  --strip   delete only the image line (smallest possible edit)

Both are single-variable edits: no code cell, no metadata, no other markdown is touched.

Usage:
  python work/fix_missing_attachments.py <notebook> --embed     [--out <notebook>]
  python work/fix_missing_attachments.py <notebook> --strip     [--out <notebook>]
  python work/fix_missing_attachments.py <notebook> --check     # report only
"""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

# A 1x1 transparent PNG. Small, valid, and renders wherever the original image would have.
PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

REF = re.compile(r"!\[[^\]]*\]\(attachment:([^)]+)\)")


def scan(nb: dict) -> list[tuple[int, list[str]]]:
    """(cell index, referenced attachment names) for every markdown cell that dangles."""
    out = []
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") != "markdown":
            continue
        src = "".join(c.get("source", []))
        names = REF.findall(src)
        if not names:
            continue
        have = set((c.get("attachments") or {}).keys())
        missing = [n for n in names if n not in have]
        if missing:
            out.append((i, missing))
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 3 or not any(a in argv for a in ("--embed", "--strip", "--check")):
        print(__doc__)
        return 2
    src_path = Path(argv[1])
    nb = json.loads(src_path.read_text(encoding="utf-8"))

    dangling = scan(nb)
    print(f"{src_path.name}: {len(nb.get('cells', []))} cells")
    if not dangling:
        print("  no dangling attachment references -- nothing to fix")
        return 0
    for i, names in dangling:
        print(f"  cell {i}: references {names} with no embedded attachment")

    mode = "embed" if "--embed" in argv else ("strip" if "--strip" in argv else "check")
    if mode == "check":
        print("\n  (check only; pass --embed or --strip to fix)")
        return 1

    for i, names in dangling:
        c = nb["cells"][i]
        src = "".join(c.get("source", []))
        if mode == "embed":
            att = c.setdefault("attachments", {})
            for n in names:
                att[n] = {"image/png": base64.b64encode(PNG_1x1).decode("ascii")}
        else:
            src = "\n".join(l for l in src.splitlines()
                            if not REF.search(l))
            c["source"] = src.splitlines(keepends=True)
    still = scan(nb)
    if still:
        print(f"REFUSING: {len(still)} dangling reference(s) remain after the edit")
        return 1

    out = src_path
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])
    out.write_text(json.dumps(nb, ensure_ascii=True), encoding="utf-8")
    raw = out.read_bytes()
    print(f"\n  wrote {out}  mode={mode}  {len(raw)} bytes")
    print(f"  ascii-only: {all(b < 128 for b in raw)}")
    try:
        raw.decode("gbk")
        print("  GBK-decodable: YES")
    except UnicodeDecodeError as exc:
        print(f"  GBK-decodable: NO -- {exc}")
        return 1
    back = json.loads(raw.decode("utf-8"))
    print(f"  round-trip: {len(back['cells'])} cells, dangling now = {len(scan(back))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
