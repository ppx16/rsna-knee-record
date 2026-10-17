# -*- coding: utf-8 -*-
"""Digest a pulled notebook: score claims, headings, and whether it is original."""
import json
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
nb = json.loads(path.read_text(encoding="utf-8"))
cells = nb.get("cells", [])
print(f"{path.name}: {len(cells)} cells")

pat = re.compile(r"0\.9[0-9]{2}|score|leaderboard|\bLB\b", re.I)
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    if not src.strip():
        continue
    flat = re.sub(r"\s+", " ", src).strip()
    if c.get("cell_type") == "markdown":
        print(f"--- md {i}: {flat[:500]}")
    elif pat.search(src):
        hits = sorted(set(m.group(0) for m in pat.finditer(src)))
        print(f"--- code {i} hits={hits}")
        for line in src.splitlines():
            if pat.search(line):
                print(f"      {line.strip()[:160]}")
