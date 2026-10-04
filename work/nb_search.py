"""Search an .ipynb's CELLS (not the raw single-line JSON) for a pattern and print the
surrounding text, so we can tell whether the code has a ConvNeXt member path and what it
does when the dataset is absent.

This decides whether our 0.943 fork is silently missing a member the frontier's stack has.
"""
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
