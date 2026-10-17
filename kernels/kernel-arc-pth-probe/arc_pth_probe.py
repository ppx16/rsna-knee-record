"""CPU-ONLY probe of F58's two unverified premises. Consumes NO GPU slot.

F58 §3 designed the numpy fix for the high-tier duck line as

    /usr/local/lib/python3.13/dist-packages/zz_arc_numpy_fix.pth

and listed two things it could NOT verify:
  (1) whether `/tmp/sgl/sgl-site` actually contains `numpy/`  -- needs the 4 GB runtime, skipped here
  (2) whether `/usr/local/lib/python3.13/dist-packages` is WRITABLE

Premise (2) is testable with no datasets and no GPU, and it is the premise that would make the fix
silently do nothing: if the directory is read-only, the `.pth` is never written and the run fails
exactly as before, with no new evidence about why.

This kernel answers, and only answers:
  * what the site directories are on this image;
  * whether dist-packages is writable;
  * whether a `.pth` placed there is actually EXECUTED by a fresh interpreter, and at which point
    in site processing (the whole point of F58: dist-packages' `.pth` runs AFTER the runtime's);
  * what the resulting `sys.path` order is, and which entries contain a `numpy/` directory.

It is deliberately dependency-free so it can run on the default CPU machine.
"""
from __future__ import annotations

import os
import subprocess
import sys
import sysconfig

print("=" * 80)
print("STEP 0 -- interpreter and site layout")
print("=" * 80)
print("python     :", sys.version.replace("\n", " "))
print("executable :", sys.executable)
print("purelib    :", sysconfig.get_paths().get("purelib"))
print("platlib    :", sysconfig.get_paths().get("platlib"))
try:
    import site
    print("site dirs  :", site.getsitepackages())
    print("usersite   :", site.getusersitepackages())
except Exception as exc:  # noqa: BLE001
    print("site import failed:", exc)

TARGET = "/usr/local/lib/python3.13/dist-packages"
print()
print("=" * 80)
print(f"STEP 1 -- is {TARGET} writable?  (F58 premise #2)")
print("=" * 80)
print("exists     :", os.path.isdir(TARGET))
print("os.access W:", os.access(TARGET, os.W_OK))
try:
    probe = os.path.join(TARGET, "zz_arc_probe_marker.txt")
    with open(probe, "w", encoding="utf-8") as fh:
        fh.write("written\n")
    print("actually wrote a file: YES ->", probe)
    os.remove(probe)
except Exception as exc:  # noqa: BLE001
    print("actually wrote a file: NO ->", type(exc).__name__, str(exc)[:200])

print()
print("=" * 80)
print("STEP 2 -- does a .pth placed there actually EXECUTE in a fresh interpreter?")
print("=" * 80)
pth = os.path.join(TARGET, "zz_arc_probe.pth")
MARK = "ARC_PROBE_PTH_RAN"
# A .pth line starting with "import " is executed at startup. Keep it one line (the .pth rule).
line = (f'import sys; print("{MARK} at index", '
        f'[i for i, p in enumerate(sys.path) if p == {TARGET!r}])\n')
try:
    with open(pth, "w", encoding="utf-8") as fh:
        fh.write(line)
    print("wrote", pth)
    r = subprocess.run([sys.executable, "-c",
                        "import sys; print('CHILD_PATH', sys.path)"],
                       capture_output=True, text=True, timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    print("fresh-interpreter output:")
    for l in out.splitlines():
        print("   |", l[:400])
    print()
    print(f"  .pth EXECUTED: {'YES' if MARK in out else 'NO'}")
except Exception as exc:  # noqa: BLE001
    print("could not write/execute the .pth:", type(exc).__name__, str(exc)[:300])
finally:
    try:
        os.remove(pth)
        print("  (removed the probe .pth again)")
    except OSError:
        pass

print()
print("=" * 80)
print("STEP 3 -- which sys.path entries actually contain a numpy/ directory?")
print("=" * 80)
for p in sys.path:
    if not p:
        continue
    hit = os.path.isdir(os.path.join(p, "numpy"))
    if hit:
        print(f"  HAS numpy/ : {p}")

print()
print("=" * 80)
print("VERDICT KEY (for the duck line / F58)")
print("=" * 80)
print("  dist-packages writable + .pth executes -> F58's fix has a landing place; the remaining")
print("      unknown is only whether `/tmp/sgl/sgl-site` really contains `numpy/`.")
print("  dist-packages NOT writable            -> F58's fix can never be installed this way;")
print("      the numpy fix must be moved to a location that IS writable AND still processed after")
print("      the runtime's site dir (this probe's STEP 3 output shows the candidate entries).")
