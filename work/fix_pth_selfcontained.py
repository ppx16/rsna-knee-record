"""Make the high-tier `.pth` self-contained: it must NOT depend on an env var.

THE MEASURED DEFECT (F74). The run of `duck18-vd` finally produced real readings:

    NUMPY_ROOT_FOR_SUBPROCESS /usr/local/lib/python3.12/dist-packages      <- correct
    NUMPY_PTH_WRITTEN         .../dist-packages/zz_arc_numpy_fix.pth True <- correct place
    [arc_numpy_fix] pruned ['/usr/local/lib/python3.12/dist-packages'] root None

`root None` is the bug. A `.pth` file is executed by `site` at INTERPRETER STARTUP, before the
child process inherits anything useful, so `os.environ.get('ARC_NUMPY_ROOT')` comes back None. The
body is then:

    r = None
    bad = [p for p in sys.path if p and p != r and os.path.isdir(os.path.join(p,'numpy'))]
    [sys.path.remove(p) for p in bad]                      # removes EVERY numpy-bearing path
    [sys.path.insert(0, r) if (r and r not in sys.path) else None]   # puts NOTHING back

i.e. on the one run where the `.pth` finally executed, it **deleted the working numpy and inserted
nothing**. That is why the kernel still failed -- and it is a strictly worse outcome than not
running the fix at all.

THE FIX: bake the root into the `.pth` text instead of reading it from the environment. `sysconfig`
is always importable at startup, so the body becomes self-contained and cannot degrade to None.

This is a token-level rewrite of the `.pth` body string; nothing else in the cell is touched.

Usage: python work/fix_pth_selfcontained.py <in.ipynb> <out.ipynb>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OLD = "import sys,os;r=os.environ.get('ARC_NUMPY_ROOT');"
NEW = "import sys,os,sysconfig;r=sysconfig.get_paths()['purelib'];"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])
    nb = json.loads(src.read_text(encoding="utf-8"))

    hits, edited = 0, None
    for i, c in enumerate(nb["cells"]):
        s = "".join(c.get("source", []))
        if OLD in s:
            hits += s.count(OLD)
            edited = i
            c["source"] = s.replace(OLD, NEW).splitlines(keepends=True)
    print(f"rewrote {hits} .pth body header(s) in cell {edited}")
    if hits != 1:
        print(f"REFUSING: expected exactly 1, got {hits}")
        return 1

    # the .pth body must remain ONE line starting with `import` (F58's requirement)
    joined = "".join(nb["cells"][edited]["source"])
    assert "r=sysconfig.get_paths()['purelib']" in joined, "rewrite did not land"
    assert "os.environ.get('ARC_NUMPY_ROOT')" not in joined, "old env lookup still present"

    bad = []
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        s = "".join(c.get("source", []))
        if s.lstrip().startswith(("%", "!")):
            continue
        try:
            compile(s, f"cell{i}", "exec")
        except SyntaxError as e:
            if i == edited:
                bad.append((i, e.msg, e.lineno))
    print("EDITED cell compiles:", "NO " + str(bad) if bad else "YES")
    if bad:
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, ensure_ascii=True), encoding="utf-8")

    # PROVE IT BY RE-READING THE DISK -- "the tool said it wrote" is not evidence (F73)
    back = json.loads(dst.read_text(encoding="utf-8"))
    txt = "".join(back["cells"][edited]["source"])
    ok = ("r=sysconfig.get_paths()['purelib']" in txt
          and "os.environ.get('ARC_NUMPY_ROOT')" not in txt)
    raw = dst.read_bytes()
    print(f"wrote {dst}  {len(raw)} bytes  re-read verified: {ok}")
    print("  ascii-only:", all(b < 128 for b in raw))
    try:
        raw.decode("gbk")
        print("  GBK-decodable: YES")
    except UnicodeDecodeError as exc:
        print("  GBK-decodable: NO", exc)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
