"""Apply F69's correction to the high-tier duck kernel's numpy fix.

BACKGROUND, all measured in this repo:
  * F58 designed the numpy repair for the high-tier duck line as a LATE `.pth`, because
    `sitecustomize` provably runs too early.
  * F67 measured (CPU-only) that `/usr/local/lib/python3.13/dist-packages` IS writable and that a
    `.pth` placed there really does execute -- and that a SECOND site dir also ships a `numpy/`
    (`/usr/lib/python3/dist-packages`).
  * F69 measured that `.pth` execution order equals **the order the site directories were added**,
    NOT path order. The duck line adds its runtime via `site.addsitedir("/tmp/sgl/sgl-site")`.
  * ⇒ The notebook writes the fix to `Path(_np).parent.parent`, i.e. **wherever numpy happened to
    resolve from at that moment**. If numpy resolved from the runtime's sgl-site, the `.pth` lands
    THERE -- it still executes, the log still shows `pruned`, and the ordering advantage that the
    whole fix depends on is silently gone.

This patcher replaces that single expression with an explicit, asserted target:
  * the directory comes from `sysconfig.get_paths()["purelib"]`, never from `numpy.__file__`;
  * it asserts the target dir exists, the file was written, and it still exists after writing;
  * it logs the target and the numpy root separately, so the run reports BOTH facts instead of one.

Single-variable discipline: this touches ONE assignment inside ONE code cell. `work/diff_nb.py`
is the check.

Usage: python work/patch_pth_target.py <in.ipynb> <out.ipynb>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OLD = '_pth = Path(_np).parent.parent / "zz_arc_numpy_fix.pth"'

NEW = '''# ---- F69 correction: the write target must NOT be derived from where numpy resolved ----
# F69 measured that .pth execution order == site-directory ADD order, so a .pth that lands in the
# runtime's own site dir executes BEFORE dist-packages and the fix silently does nothing.
# Therefore: take the target from sysconfig explicitly, and assert every step.
import sysconfig as _sc
_pth_dir = Path(_sc.get_paths()["purelib"])
assert _pth_dir.is_dir(), f"purelib is not a directory: {_pth_dir}"
log("PTH_TARGET_DIR", str(_pth_dir))
log("PTH_NUMPY_ROOT", str(Path(_np).parent.parent))
_pth = _pth_dir / "zz_arc_numpy_fix.pth"'''


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])
    nb = json.loads(src.read_text(encoding="utf-8"))

    hits = []
    for i, c in enumerate(nb["cells"]):
        s = "".join(c.get("source", []))
        if OLD in s:
            hits.append(i)
            c["source"] = s.replace(OLD, NEW).splitlines(keepends=True)
    print(f"cells containing the old expression: {hits}")
    if len(hits) != 1:
        print(f"REFUSING: expected exactly 1 hit, got {len(hits)}")
        return 1

    # the written .pth must still be checked for existence somewhere after the write
    joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    if "_pth.write_text(" not in joined:
        print("REFUSING: the .pth write call disappeared")
        return 1
    if "assert _pth.is_file()" not in joined:
        # add the post-write assertion immediately after the existing write call block
        for c in nb["cells"]:
            s = "".join(c.get("source", []))
            if "_pth.write_text(" in s and "assert _pth.is_file()" not in s:
                # find the end of the write_text(...) statement and append an assert
                marker = 'log("WROTE", str(_pth))'
                if marker in s:
                    s = s.replace(marker, marker + '\nassert _pth.is_file(), "the .pth was not written"')
                else:
                    s = s.rstrip("\n") + '\nassert _pth.is_file(), "the .pth was not written"\n'
                c["source"] = s.splitlines(keepends=True)
                print("added the post-write existence assertion")
                break

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
