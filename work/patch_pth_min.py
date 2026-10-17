"""Minimal, syntax-safe variant of the F69 correction.

F71 established the failure mode of the FIRST attempt: `patch_pth_target.py` inserted comment lines
and blank lines next to the `.pth` string-concatenation, which must stay contiguous because F58
requires the `.pth` body to be ONE line. The result was
`SyntaxError: unterminated string literal`, raised DURING execution -- which at least proved F65's
attachment fix worked (the notebook finally executed).

This patcher therefore does the smallest possible thing:
  * baseline is `kernel-arc-duck18-va` (the version whose syntax was never broken);
  * ONE token is replaced:  `Path(_np).parent.parent`  ->  `Path(__import__('sysconfig').get_paths()['purelib'])`
  * no new lines, no imports, no comments, nothing near the string literals.

Why the change is right at all (F67 + F69):
  * F67 measured that `/usr/local/lib/python3.13/dist-packages` is writable and that a `.pth` there
    really executes;
  * F69 measured that `.pth` execution order == the order site dirs were added, so a `.pth` landing
    in the runtime's own site dir executes too early and silently fixes nothing;
  * `Path(_np).parent.parent` follows wherever numpy happened to resolve -- that is the defect.

Usage: python work/patch_pth_min.py <in.ipynb> <out.ipynb>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OLD = "Path(_np).parent.parent"
NEW = "Path(__import__('sysconfig').get_paths()['purelib'])"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])
    nb = json.loads(src.read_text(encoding="utf-8"))

    hits = 0
    for c in nb["cells"]:
        s = "".join(c.get("source", []))
        n = s.count(OLD)
        if n:
            hits += n
            c["source"] = s.replace(OLD, NEW).splitlines(keepends=True)
    print(f"occurrences replaced: {hits}")
    # TWO occurrences, and both must change: `ARC_NUMPY_ROOT` is the directory the .pth inserts at
    # position 0, and `_pth` is where the .pth is written. Both carry the same semantics -- "the
    # site dir that must win" -- so both must be dist-packages, not "wherever numpy resolved".
    if hits != 2:
        print(f"REFUSING: expected exactly 2 occurrences, got {hits}")
        return 1

    joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    for must in ("_pth.write_text(", "zz_arc_numpy_fix.pth", "print('[arc_numpy_fix] pruned"):
        if must not in joined:
            print(f"REFUSING: the .pth body is damaged (missing {must!r})")
            return 1
    if OLD in joined:
        print("REFUSING: the old expression is still present")
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
    back = json.loads(raw.decode("utf-8"))
    print("  round-trip cells:", len(back["cells"]))
    print("  line-count delta per cell is the check to run next (diff_nb)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
