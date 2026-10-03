"""Change the EffB0-MIL leg's default weight inside kernel-0943cfm2eff.

WHY
    The honest dose curve (`work/eff_dose.py`, `docs/FIND-2026-10-04-effb0-honest-test.md`) shows
    **w=0.10 is under-dosing**: on the leg's own four out-of-fold folds,

        w=0.10 -> mean +0.00361, positive 4/4
        w=0.15 -> mean +0.00493, positive 4/4
        w=0.20 -> mean +0.00594, positive 3/4   (fold 0 turns negative)

    w=0.15 is the largest weight at which **all four folds agree in sign**, and it is well below the
    in-sample optimum, so it is not chasing the peak. The public board cannot see the difference
    (SE ~= 0.008), so the choice is made on the local measurement.

⚠️  DO NOT run this while the kernel is RUNNING or has a PENDING submission -- pushing a new version
    destroys the running version's output. This script only EDITS; it never pushes. Push separately,
    after the current version reaches a terminal status.

Usage:
    python scripts/set_effb0_weight.py --check      # report only
    python scripts/set_effb0_weight.py --weight 0.15
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "kernels" / "kernel-0943cfm2eff" / "rsna-knee-restructured-version-3.ipynb"
PAT = re.compile(r'("effb0mil_w"\s*:\s*float\(os\.environ\.get\(\s*"RSNA_EFFB0MIL_W"\s*,\s*")([0-9.]+)(")')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="0.15")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if not NB.exists():
        print("!! notebook missing: %s" % NB)
        return 2
    nb = json.loads(NB.read_text(encoding="utf-8", errors="replace"))
    hits, current, cells = 0, [], None
    for i, c in enumerate(nb["cells"]):
        txt = "".join(c.get("source", []))
        m = PAT.search(txt)
        if m:
            hits += 1
            current.append(m.group(2))
            cells = i if cells is None else cells
            if not a.check and m.group(2) != a.weight:
                c["source"] = [PAT.sub(lambda mm: mm.group(1) + a.weight + mm.group(3), txt)]

    print("default-weight site found %d time(s) (cell %s); current default = %s"
          % (hits, cells, current))
    if hits != 1:
        print("!! expected exactly ONE site -- refusing to guess")
        return 3
    if a.check:
        print("--check: nothing written.")
        return 0
    if current[0] == a.weight:
        print("already at %s; nothing to do." % a.weight)
        return 0

    whole = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert ('"%s"' % a.weight) in whole, "new weight did not land"
    assert len(nb["cells"]) == len(nb["cells"]), "noop"
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote default weight %s -> %s" % (current[0], a.weight))
    print("!! now push it MANUALLY (never while RUNNING / with a PENDING submission):")
    print('   $env:PYTHONUTF8=1; kaggle kernels push -p kernels/kernel-0943cfm2eff -t 34200')
    return 0


if __name__ == "__main__":
    sys.exit(main())
