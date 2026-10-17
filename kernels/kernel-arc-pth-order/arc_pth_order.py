"""CPU-ONLY test of F68's ordering claim. Consumes NO GPU slot.

F68 hypothesised: if the duck line's `.pth` lands in the RUNTIME's site dir
(`/tmp/sgl/sgl-site`, added via `site.addsitedir`) instead of dist-packages, then it still executes
-- but it executes BEFORE dist-packages' `.pth`, so the whole "late .pth wins" property that F58
relies on is defeated, and the run can look healthy while numpy stays broken.

That claim is testable with no datasets and no GPU: fabricate two site directories, give each a
`.pth` that appends a marker, put them into `sys.path` in the two orders that matter, and see which
marker comes first.

This kernel answers ONLY that. Standard library only.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

MARK_RT = "MARK_RUNTIME_SITE_PTH"
MARK_DP = "MARK_DIST_PACKAGES_PTH"


def make_site(root: str, name: str, mark: str, also_numpy: bool) -> str:
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    if also_numpy:
        # a decoy numpy package, so "which site dir provides numpy" is also observable
        nd = os.path.join(d, "numpy")
        os.makedirs(nd, exist_ok=True)
        with open(os.path.join(nd, "__init__.py"), "w", encoding="utf-8") as fh:
            fh.write("SRC = %r\n" % d)
    with open(os.path.join(d, "zz_probe.pth"), "w", encoding="utf-8") as fh:
        # one line, starting with import, exactly the .pth rule F58 recorded
        fh.write(f'import sys; print("{mark}")\n')
    return d


def run_child(script: str) -> str:
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    return (r.stdout or "") + (r.stderr or "")


def main() -> None:
    root = tempfile.mkdtemp(prefix="pthorder_")
    try:
        rt = make_site(root, "runtime_site", MARK_RT, also_numpy=True)   # decoy: HAS numpy/
        dp = make_site(root, "dist_packages_like", MARK_DP, also_numpy=False)
        print("runtime site :", rt)
        print("dist-packages-like:", dp)
        print()

        # ORDER A: runtime first, dist-packages second  (what the duck line risks)
        a = run_child(
            "import site, sys\n"
            f"site.addsitedir({rt!r})\n"
            f"site.addsitedir({dp!r})\n"
            "print('PATH_A', sys.path[:6])\n"
        )
        # ORDER B: dist-packages first, runtime second  (what F58 wants)
        b = run_child(
            "import site, sys\n"
            f"site.addsitedir({dp!r})\n"
            f"site.addsitedir({rt!r})\n"
            "print('PATH_B', sys.path[:6])\n"
        )

        for label, out in (("A runtime-then-dist", a), ("B dist-then-runtime", b)):
            print("=" * 78)
            print("ORDER", label)
            print("=" * 78)
            for l in out.splitlines():
                if l.startswith("MARK_") or l.startswith("PATH_"):
                    print("   |", l[:260])
            first = next((l for l in out.splitlines()
                          if l.startswith("MARK_")), "<none>")
            print(f"   first marker executed: {first}")

        print()
        print("=" * 78)
        print("VERDICT KEY (F68)")
        print("=" * 78)
        print("  In the order that mirrors the duck line (runtime added first), if the RUNTIME marker")
        print("  prints BEFORE the dist-packages marker, then F68 is CONFIRMED: a .pth that lands in")
        print("  the runtime's site dir executes too early to fix anything, and the log can still")
        print("  look healthy.  In that case the fix must write to dist-packages EXPLICITLY (never")
        print("  derived from wherever numpy happened to resolve) and must assert that the")
        print("  dist-packages copy exists.")
        print("  If the two markers print in the opposite order, F68 is REFUTED and the ordering")
        print("  concern can be dropped.")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
