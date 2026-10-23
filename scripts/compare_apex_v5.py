"""Compare the Apex 0.950 kernel against its V5 variant.

Prints the V5 metadata, the headline markdown, and a unified diff of the
fusion cell so we can see what (if anything) moved the score.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p: Path) -> list[str]:
    nb = json.loads(p.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


def main() -> None:
    apex = ROOT / "scratch" / "apex" / "apex-nartaa-rank-blend-0-950-lb.ipynb"
    v5 = ROOT / "scratch" / "apex_v5" / "rsna-knee-apex-grandmaster-stack-v5.ipynb"

    print("=== V5 metadata ===")
    print((ROOT / "scratch" / "apex_v5" / "kernel-metadata.json").read_text(encoding="utf-8"))

    a = load(apex)
    v = load(v5)
    print(f"cells: apex={len(a)} v5={len(v)}")

    print("\n=== V5 headline markdown (cell 0) ===")
    print(v[0][:2200])

    print("\n=== cell-by-cell: identical? ===")
    for i in range(max(len(a), len(v))):
        sa = a[i] if i < len(a) else "<missing>"
        sb = v[i] if i < len(v) else "<missing>"
        same = "SAME" if sa == sb else f"DIFF (apex {len(sa)} vs v5 {len(sb)})"
        print(f"  cell {i}: {same}")

    print("\n=== diff of the LAST code cell (fusion) ===")
    fv = v[-1].splitlines(keepends=True)
    fa = a[-1].splitlines(keepends=True)
    d = list(difflib.unified_diff(fa, fv, fromfile="apex", tofile="v5", n=2))
    print("".join(d[:200]) if d else "(identical)")


if __name__ == "__main__":
    main()
