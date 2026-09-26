"""Compare our ensemble fork against the upstream public notebook.

WHY (2026-09-14)
    `mattiaangeli/bend-the-knee-to-the-dinosaurs` is the public notebook our `kernel-ens` was
    forked from.  It has 215 votes and its last commit ran TODAY, so it may have moved since we
    forked (2026-09-10).  Adopting an upstream improvement is the cheapest possible score gain,
    and it is exactly how we got to 0.941 in the first place.

    Inputs are already known to be identical (same 12 dataset_sources, same 2 kernel_sources,
    same dinov2 model source, same T4 shape) -- verified by reading both kernel-metadata.json.

    So the only question left is CODE.  This prints a per-cell diff summary, newest-looking
    logic first, plus a diff of every numeric-ish config line.

Usage:
    python scripts/nb_diff_upstream.py --ours kernel-ens/rsna-knee-ensemble-v1.ipynb \
        --theirs downloads/ref/bend/bend-the-knee-to-the-dinosaurs.ipynb
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys


def cells(path):
    nb = json.load(open(path, encoding="utf-8"))
    out = []
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = c.get("source")
        if isinstance(src, list):
            src = "".join(src)
        out.append(src or "")
    return out


def norm(line):
    return re.sub(r"\s+", " ", line).strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="kernels/kernel-ens/rsna-knee-ensemble-v1.ipynb")
    ap.add_argument("--theirs", default="downloads/ref/bend/bend-the-knee-to-the-dinosaurs.ipynb")
    ap.add_argument("--max-cells", type=int, default=12)
    a = ap.parse_args(argv)

    A, B = cells(a.ours), cells(a.theirs)
    print("ours   %d code cells" % len(A))
    print("theirs %d code cells" % len(B))

    pairs = []
    for i in range(max(len(A), len(B))):
        x = A[i].splitlines() if i < len(A) else []
        y = B[i].splitlines() if i < len(B) else []
        sm = difflib.SequenceMatcher(None, [norm(l) for l in x], [norm(l) for l in y])
        added = [norm(y[j2]) for tag, _i1, _i2, j1, j2 in sm.get_opcodes()
                 if tag in ("insert", "replace") for j2 in range(j1, j2)]
        removed = [norm(x[i1]) for tag, i1, i2, _j1, _j2 in sm.get_opcodes()
                   if tag in ("delete", "replace") for i1 in range(i1, i2)]
        pairs.append((len(added) + len(removed), i, added, removed))

    pairs.sort(reverse=True)
    print("\n== cells with the largest source difference ==")
    for n, i, added, removed in pairs[: a.max_cells]:
        if n == 0:
            continue
        print("\n--- cell[%d]: +%d / -%d lines" % (i, len(added), len(removed)))
        for l in added[:6]:
            print("   ONLY-UPSTREAM | %s" % l[:150])
        for l in removed[:6]:
            print("   ONLY-OURS     | %s" % l[:150])

    # numeric config lines anywhere
    def nums(texts):
        d = {}
        for t in texts:
            for line in t.splitlines():
                s = norm(line)
                m = re.match(r"^([A-Za-z_][A-Za-z0-9_\[\]'\"]*)\s*[:=]\s*(-?[0-9.]+)\s*,?$", s)
                if m:
                    d.setdefault(m.group(1), set()).add(m.group(2))
        return d

    na, nb = nums(A), nums(B)
    print("\n== numeric config keys that differ ==")
    for k in sorted(set(na) | set(nb)):
        va, vb = na.get(k), nb.get(k)
        if va != vb:
            print("  %-28s ours=%-22s upstream=%s" % (k, sorted(va or []), sorted(vb or [])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
