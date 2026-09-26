# -*- coding: utf-8 -*-
"""One-shot patch: make every RUN fusion weight print itself at run time.

Why: `[config]` used to print only `preset` and `coat_private_alpha`. The `a5_w` bracket
therefore ran with no way to prove from its own log that the value had taken effect - so a
run that silently failed to apply would be indistinguishable from one whose axis is
genuinely flat, and a LIVE axis would get killed by mistake (HANDOFF 32.3).

Run once against kernel-fresh/rsna-knee-ens-v2src.ipynb. Idempotent: re-running is a no-op.
"""
import hashlib
import json
import pathlib
import sys
from kpaths import kdir          # kernels/ layout, see kpaths.py

P = kdir("kernel-fresh") / "rsna-knee-ens-v2src.ipynb"

OLD = (
    'print("[config] outer CoAtNet weight per finding:",\n'
    '      {k: v for k, v in RUN["coatnet_w"].items() if k != "__default__"} or "flat 0.60",\n'
    '      flush=True)'
)
# NOTE: the marker deliberately has NO trailing newline. This print is the LAST statement in
# the config cell, so the cell source ends at `flush=True)` with no final "\n" - anchoring on
# a trailing newline matched zero times and cost a round to find.
NEW = OLD + (
    "\n"
    "# Every other fusion weight, in one line. Without this an experiment that silently\n"
    "# failed to take effect is indistinguishable from one whose axis is genuinely flat,\n"
    "# and that would kill a live axis by mistake. a5_w, the rad_* weights,\n"
    "# coat_private_alpha and the Raptor view weights are all visible here now.\n"
    'print("[config] fusion weights:",\n'
    '      {k: v for k, v in RUN.items() if k != "coatnet_w"}, flush=True)'
)
MARK = "[config] fusion weights"


def main() -> int:
    raw = P.read_text(encoding="utf-8")
    if MARK in raw:
        print("already patched - nothing to do")
        return 0
    before = hashlib.sha256(P.read_bytes()).hexdigest()
    nb = json.loads(raw)

    hits = []
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        s = "".join(c.get("source", []))
        n = s.count(OLD)
        if n:
            hits.append((i, c, s, n))
    total = sum(n for _, _, _, n in hits)
    if total != 1:
        print("ABORT: the config print block matched %d times, expected 1" % total)
        return 1
    idx, cell, s, _ = hits[0]
    s = s.replace(OLD, NEW)
    cell["source"] = s.splitlines(keepends=True)

    src = "".join(cell["source"])
    compile(src, "<cell%d>" % idx, "exec")          # the cell must still parse
    if MARK not in src:
        print("ABORT: the new print is not in the edited cell")
        return 1

    tmp = P.with_suffix(".ipynb.tmp")
    tmp.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(P)

    readback = json.loads(P.read_text(encoding="utf-8"))
    rb = "".join(readback["cells"][idx]["source"])
    if MARK not in rb or "outer CoAtNet weight per finding" not in rb:
        print("ABORT: read-back does not contain BOTH prints")
        return 1
    after = hashlib.sha256(P.read_bytes()).hexdigest()
    print("patched cell[%d]" % idx)
    print("  source sha256 %s -> %s" % (before[:16], after[:16]))
    print("  both the per-finding line and the new full-weights line are present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
