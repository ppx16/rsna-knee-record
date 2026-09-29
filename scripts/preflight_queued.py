"""Verify that every QUEUED kernel on disk carries the change it is supposed to carry.

WHY
    HANDOFF 58 established "prove the configuration landed before spending the submission".
    Until now each pre-flight was hand-rolled, and the first hand-rolled version of THIS check
    produced a false alarm: it asserted that `rad_cal_w` appears exactly once in the radcal00
    notebook, but the string legitimately appears TWICE -- once as the config value
    `"rad_cal_w": 0.00,` and once as its consumer `_RAD_CAL_W = RUN['rad_cal_w']`.
    Per HANDOFF 55.5, when my script and the authoritative reading disagree, the script is
    wrong.  So the expectations live in a table here, with the semantics spelled out, instead
    of being re-invented in a shell one-liner each round.

    This also catches the failure mode the project fears most (HANDOFF 16.1): a kernel that
    silently carries NO change, which then reads as "that knob does nothing".

Usage:
    python scripts/preflight_queued.py                 # every kernel in state/cv_targets.json
    python scripts/preflight_queued.py --verbose
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
from kpaths import kdir          # kernels/ layout, see kpaths.py
TARGETS = ROOT / "state" / "cv_targets.json"

# kernel dir -> list of (label, source, pattern, expectation)
#   expectation:  ("eq", n) exact count | ("ge", n) at least n | ("absent",) must not occur
EXPECT = {
    "kernel-cv-gap-fold0": [
        ("+/-2 window spacing", "py", "vol[c - 2], vol[c], vol[c + 2]", ("eq", 1)),
        ("no +/-1 remnant", "py", "vol[c - 1]", ("absent",)),
        ("no stale clamp", "py", "self._D - 2", ("absent",)),
    ],
    "kernel-cv-r288-fold0": [
        ("res overridden", "py", "'--res', '288'", ("eq", 1)),
    ],
    "kernel-cv-seed-fold0": [
        ("seed overridden", "py", "'--seed', '123'", ("eq", 1)),
    ],
    "kernel-cv-norm-fold0": [
        ("norm set", "py", "'--norm', 'imagenet'", ("eq", 1)),
    ],
    "kernel-cv-r336-fold0": [
        ("res overridden to native", "py", "'--res', '336'", ("eq", 1)),
    ],
    "kernel-cv-k24-fold0": [
        # NOTE: the base argv still carries '--k', '12' -- argparse keeps the LAST occurrence, so
        # asserting the old value is ABSENT would be wrong.  Assert the appended one instead, and
        # that the base line is still intact (so the edit landed on the argv, not somewhere else).
        ("windows doubled", "py", "'--k', '24'", ("eq", 1)),
        ("base value still present, overridden", "py", "'--k', '12'", ("eq", 1)),
    ],
    "kernel-exp/v2src-radcal00": [
        # NOTE the count is 1 for the CONFIG VALUE and 1 for its consumer, so assert on the
        # value, not on the identifier.
        ("calibration weight zeroed", "nb", '"rad_cal_w": 0.00', ("eq", 1)),
        ("old weight gone", "nb", '"rad_cal_w": 0.40', ("absent",)),
    ],
    "kernel-exp/v2src-radcal00-parent": [
        ("routing pinned to parent", "nb", 'PRESET = "parent"', ("eq", 1)),
        ("calibration weight zeroed", "nb", '"rad_cal_w": 0.00', ("eq", 1)),
        ("old weight gone", "nb", '"rad_cal_w": 0.40', ("absent",)),
        # the preset must still be able to be probed at all -- if this line vanished the edit
        # above would have replaced something else.
        ("preset env hook intact", "nb", 'os.environ.get("RSNA_PRESET"', ("eq", 0)),
    ],
    "kernel-exp/v2src-seedalt": [
        # reproducibility probe: the seed must have moved and nothing else should have.
        ("seed moved to 2027", "nb", "SEED = 2027", ("eq", 1)),
        ("old seed gone", "nb", "SEED = 2026", ("absent",)),
        ("recipe untouched", "nb", '"rad_cal_w": 0.40', ("eq", 1)),
        ("routing untouched", "nb", 'os.environ.get("RSNA_PRESET", "probe22")', ("eq", 1)),
    ],
    "kernel-tpu-3dsmoke": [
        # The run died at ~75 s on `TypeError: not all arguments converted during string
        # formatting`: 5 placeholders, 6 args, because the literal `1` in `input (1,%d,%d,%d)` was
        # ALSO passed as an argument.  Assert the fixed tuple is present and the broken one is gone.
        ("format-args fixed", "py", "nparam / 1e6, a.depth, a.hw, a.hw", ("eq", 1)),
        ("6-arg tuple gone", "py", "nparam / 1e6, 1, a.depth", ("absent",)),
        # ⚠️ The metadata assertion exists because a kernel NAMED tpu-msft was pushed with
        # enable_tpu FALSE and ran on a T4 for 10.7 h, silently spending GPU quota.  Source text
        # alone cannot see that; the flags and the name had disagreed with nothing checking.
        ("runs on TPU", "meta", '"enable_tpu": true', ("eq", 1)),
        ("does not claim GPU", "meta", '"enable_gpu": false', ("eq", 1)),
    ],
}


def source_text(d: pathlib.Path, kind: str):
    if kind == "py":
        # Script kernels use either name; try both rather than reporting "source not found".
        for name in ("knee-mri-training-the-twelve-finding-model.py", "kernel_main.py"):
            f = d / name
            if f.exists():
                return f.read_text(encoding="utf-8")
        return None
    if kind == "meta":
        f = d / "kernel-metadata.json"
        return f.read_text(encoding="utf-8") if f.exists() else None
    hits = glob.glob(str(d / "*.ipynb"))
    if not hits:
        return None
    nb = json.load(open(hits[0], encoding="utf-8"))
    return "".join("".join(c.get("source") or []) for c in nb.get("cells", [])
                   if c.get("cell_type") == "code")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    targets = json.load(open(TARGETS, encoding="utf-8"))
    dirs = []
    for t in sorted(targets, key=lambda x: x.get("priority", 99)):
        dirs.append((t.get("dir"), t.get("ref"), t.get("hold")))

    bad, unchecked = [], []
    for d, ref, hold in dirs:
        if not d:
            continue
        p = kdir(d)
        if not p.is_dir():
            print("%-30s MISSING DIRECTORY" % d)
            bad.append(d)
            continue
        checks = EXPECT.get(d)
        if not checks:
            unchecked.append(d)
            continue
        problems, notes = [], []
        for label, kind, pat, exp in checks:
            txt = source_text(p, kind)
            if txt is None:
                problems.append("%s: source not found" % label)
                continue
            n = txt.count(pat)
            if exp[0] == "eq":
                good = n == exp[1]
                notes.append("%s ok" % label if good else "%s: %d != %d" % (label, n, exp[1]))
            elif exp[0] == "ge":
                good = n >= exp[1]
                notes.append("%s ok" % label if good else "%s: %d < %d" % (label, n, exp[1]))
            else:
                good = n == 0
                notes.append("%s ok" % label if good else "%s: found %d" % (label, n))
            if not good:
                problems.append(notes[-1])
        status = "OK  " if not problems else "FAIL"
        if hold:
            status += " (held)"
        print("%-30s %s | %s" % (d, status, "; ".join(notes) if a.verbose or problems else ""))
        if problems:
            bad.append(d)

    if unchecked:
        print("\nNo assertion defined (NOT a pass -- these are simply unverified here):")
        for d in unchecked:
            print("   %s" % d)

    print("\nproblems: %d" % len(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
