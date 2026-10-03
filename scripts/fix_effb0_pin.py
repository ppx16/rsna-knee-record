"""v3 fix for kernel-0943cfm2eff: the frozen-corpus pin's always-False slot term.

THE BUG (diagnosed from the v2 log, then confirmed in the source)
    cell 50, line 61:   _EFF_SLOT_NAMES = ["SAG_FS", "COR_FS", "AX_FS", "SAG_T1"]   <- stale 4th
    cell 50, line 276:  and _EFF_SLOT_LABELS[:3] == _EFF_SLOT_NAMES[:3]

    `_EFF_SLOT_LABELS` is built from THIS notebook's SLOTS -> ['SAG_FLUID_FS', 'COR_FLUID_FS',
    'AX_FLUID_FS', ...].  `_EFF_SLOT_NAMES` is the CACHE's vocabulary -> ['SAG_FS', 'COR_FS',
    'AX_FS', ...].  Two different vocabularies: the comparison is False on EVERY run.

    So the leg died after assets, library, vocabulary resolution and the metadata cross-check had
    ALL succeeded -- killed by its own defensive assert.  And because the raise printed only the
    actual side, the log could not say which of the seven terms had failed.

THE FIX
    1. Drop the category-wrong term. The meaningful invariant (each resolved slot's plane and
       fat-suppression matching the library's SLOT_PLANE/SLOT_FS) is ALREADY enforced per slot
       further up, so the pin does not need a spelling comparison at all.
    2. Report per-term, with expected AND actual, so the next failure is diagnosable in one read.
    3. Un-stale `_EFF_SLOT_NAMES` so it can never mislead again.

Usage:  python scripts/fix_effb0_pin.py [--check]
"""
from __future__ import annotations

import json
import pathlib
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "kernels" / "kernel-0943cfm2eff" / "rsna-knee-restructured-version-3.ipynb"

OLD_NAMES = '_EFF_SLOT_NAMES = ["SAG_FS", "COR_FS", "AX_FS", "SAG_T1"]'
NEW_NAMES = ('# The CACHE\'s vocabulary, in cache order -- NOT this notebook\'s spellings (those are\n'
             '# SAG_FLUID_FS / COR_FLUID_FS / AX_FLUID_FS / SAG_FLUID_NOFS). Kept only as the\n'
             '# documented ground truth; nothing compares it against our labels, because that\n'
             '# comparison is False by construction and once killed the leg (see _EFF_PIN below).\n'
             '_EFF_SLOT_NAMES = ["SAG_FS", "COR_FS", "AX_FS", "SAG_NOFS"]')

OLD_PIN = '''        _EFF_PIN = (RULES == RULES_LEGACY and CACHE_SLICES == 16 and IMG == 224
                    and abs(CROP_MM - 130.0) < 1e-9 and tuple(SLICE_BAND) == (0.12, 0.88)
                    and len(set(_EFF_SLOT_IDX)) == _EFF_N_SLOT
                    and _EFF_SLOT_LABELS[:3] == _EFF_SLOT_NAMES[:3])
        if not _EFF_PIN:
            raise RuntimeError(f"frozen-corpus pin failed: {RULES} {CACHE_SLICES} {IMG} "
                               f"{CROP_MM} {SLICE_BAND} {_EFF_SLOT_LABELS} {_EFF_SLOT_IDX}")
        print(f"[effb0] pinned: rules={RULES} slices={CACHE_SLICES} img={IMG} "
              f"crop={CROP_MM:g}mm band={SLICE_BAND}", flush=True)'''

NEW_PIN = '''        # v3 FIX.  This pin used to end with
        #     `_EFF_SLOT_LABELS[:3] == _EFF_SLOT_NAMES[:3]`
        # which compares THIS notebook's spellings ('SAG_FLUID_FS', ...) against the CACHE's
        # ('SAG_FS', ...).  Two different vocabularies => False on every run, so the leg was
        # killed here AFTER assets, library, slot resolution and the metadata cross-check had all
        # succeeded, and the raise printed only the actual side so the log could not say which
        # term failed.  The meaningful invariant -- each resolved slot's plane and fat-suppression
        # matching the library's SLOT_PLANE/SLOT_FS -- is already enforced per slot above, so
        # there is no spelling comparison to make here.  Terms are now reported individually.
        _EFF_PIN_TERMS = [
            ("rules", RULES, RULES_LEGACY, RULES == RULES_LEGACY),
            ("cache_slices", CACHE_SLICES, 16, CACHE_SLICES == 16),
            ("img", IMG, 224, IMG == 224),
            ("crop_mm", CROP_MM, 130.0, abs(CROP_MM - 130.0) < 1e-9),
            ("slice_band", tuple(SLICE_BAND), (0.12, 0.88), tuple(SLICE_BAND) == (0.12, 0.88)),
            ("slot_indices_injective", list(_EFF_SLOT_IDX),
             f"{_EFF_N_SLOT} distinct", len(set(_EFF_SLOT_IDX)) == _EFF_N_SLOT),
        ]
        _EFF_PIN_BAD = [(n, a, e) for n, a, e, ok in _EFF_PIN_TERMS if not ok]
        if _EFF_PIN_BAD:
            raise RuntimeError(
                "frozen-corpus pin failed on "
                + "; ".join(f"{n}: got {a!r}, expected {e!r}" for n, a, e in _EFF_PIN_BAD))
        print(f"[effb0] pinned: rules={RULES} slices={CACHE_SLICES} img={IMG} "
              f"crop={CROP_MM:g}mm band={SLICE_BAND} slots={_EFF_SLOT_LABELS}", flush=True)'''


def main() -> int:
    check = "--check" in sys.argv
    nb = json.loads(NB.read_text(encoding="utf-8", errors="replace"))
    cell = None
    for i, c in enumerate(nb["cells"]):
        if "_EFF_SLOT_NAMES" in "".join(c.get("source", [])):
            cell = i
            break
    if cell is None:
        print("!! cell containing _EFF_SLOT_NAMES not found")
        return 2

    src = "".join(nb["cells"][cell]["source"])
    n_names = src.count(OLD_NAMES)
    n_pin = src.count(OLD_PIN)
    print("cell %d: _EFF_SLOT_NAMES match=%d  pin-block match=%d" % (cell, n_names, n_pin))
    if n_names != 1 or n_pin != 1:
        print("!! expected exactly 1 each -- refusing to guess")
        return 3
    if check:
        print("--check: nothing written.")
        return 0

    new = src.replace(OLD_NAMES, NEW_NAMES, 1).replace(OLD_PIN, NEW_PIN, 1)
    assert "SAG_T1\"]" not in new.split("_EFF_META_SLOTS")[0], "stale SAG_T1 survived in _EFF_SLOT_NAMES"
    # NB: _EFF_PIN_TERMS legitimately appears more than once (definition + comprehension), so
    # counting it is not a usable check; assert on the distinctive failure-report string instead.
    assert "frozen-corpus pin failed on " in new, "per-term pin did not land"
    assert "_EFF_PIN_BAD" in new, "per-term failure list did not land"
    assert "_EFF_SLOT_LABELS[:3] == _EFF_SLOT_NAMES[:3]" not in new, "the always-False term survived"

    nb["cells"][cell]["source"] = [new]
    n_cells = len(nb["cells"])
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    # re-read and compile every code cell
    nb2 = json.loads(NB.read_text(encoding="utf-8", errors="replace"))
    assert len(nb2["cells"]) == n_cells, "cell count changed on write"
    ok = 0
    for c in nb2["cells"]:
        if c.get("cell_type") != "code":
            continue
        try:
            compile("".join(c.get("source", [])), "<cell>", "exec")
            ok += 1
        except SyntaxError as e:
            print("!! SyntaxError in a code cell: %s" % e)
            return 4
    print("wrote cell %d; %d/%d code cells compile" % (cell, ok, ok))
    print("!! push it MANUALLY (never while RUNNING / with a PENDING submission):")
    print('   $env:PYTHONUTF8=1; kaggle kernels push -p kernels/kernel-0943cfm2eff -t 34200')
    return 0


if __name__ == "__main__":
    sys.exit(main())
