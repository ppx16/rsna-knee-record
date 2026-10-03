"""v4 hardening for kernel-0943cfm2eff: bring the test-cache build INSIDE the leg's budget guard.

THE DEFECT (read out of the code, documented before v3 finished -- docs/PREDICT-2026-10-04-*.md)
    cell 50, in order:

        500  _gate_t0 = time.time()
        521  _eff_check_budget("convention gate", since=_gate_t0)   <- covers ONLY the gate
        555  _got = _eff_rebuild(_EFF_TEST_IDS, ...)                <- THE TEST-CACHE BUILD
        581  _EFF_TS, _EFF_TM = _effb0mil_build_test_cache()
        588  for _f in range(4):
        589      _eff_check_budget(f"fold {_f}")                     <- first look at the LEG budget

    The single most expensive step therefore runs with NO budget check in front of it. The leg's
    whole purpose for existing is to abandon ITSELF so that a slow leg cannot reach the kernel's
    fail-CLOSED `rsna_deadline` (which would abort the kernel and produce NO submission at all) --
    and the step most likely to overrun is the one step that guard does not cover.

    It is also why v3 is expected to abandon at fold 0: `_EFF_BUDGET_S = 540`, while C82 records the
    corpus build at 0.2-0.7 h per 4407 studies, i.e. ~216-756 s for ~1322 test studies before the
    gate and before four folds of inference.

WHAT THIS CHANGES
    1. A budget check IMMEDIATELY before the test-cache build -- so an overrun is caught before the
       expensive step is paid for, and can never reach `rsna_deadline`.
    2. The default leg budget 540 -> 1800 s, sized to the measured estimate. The pipeline runs
       ~3.5-4 h against TIME_BUDGET = 8 h and the kernel is pushed with -t 34200, so ~4 h of margin
       makes 30 min safe. `_EFF_RESERVE_S = 240` still guards the kernel deadline.

⚠️ `kernel-0943cfm2eff` v3 is RUNNING -- this edits the local notebook only and never pushes.
   Push only after v3 reaches a terminal status.

Usage:  python scripts/harden_effb0_budget.py [--check]
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

OLD_BUDGET = 'float(os.environ.get("RSNA_EFFB0MIL_BUDGET_S", "540"))'
NEW_BUDGET = 'float(os.environ.get("RSNA_EFFB0MIL_BUDGET_S", "1800"))'

OLD_BUILD = "        _EFF_TS, _EFF_TM = _effb0mil_build_test_cache()"
NEW_BUILD = '''        # v4: THE EXPENSIVE STEP MUST BE BEHIND THE GUARD.  The test-cache build is the single
        # costliest thing this leg does (~216-756 s by the C82 corpus-build rate, for ~1322 test
        # studies) and until now nothing checked the budget in front of it -- the first look at the
        # cumulative leg budget came at fold 0, i.e. after it had already been paid for.  That
        # defeats the leg's whole reason for self-bounding: a leg that overruns here could reach
        # the kernel's fail-CLOSED `rsna_deadline`, which aborts the kernel and yields NO
        # submission, instead of the valid kernel-0943cfm fallback.
        _eff_check_budget("test cache build (pre-flight)")
        _eff_build_t0 = _eff_time.time()
        _EFF_TS, _EFF_TM = _effb0mil_build_test_cache()
        print(f"[effb0] test cache built in {_eff_time.time() - _eff_build_t0:.1f}s", flush=True)'''


def main() -> int:
    check = "--check" in sys.argv
    nb = json.loads(NB.read_text(encoding="utf-8", errors="replace"))

    # NOTE: the two edits live in DIFFERENT cells. The 540 default is in the RUN CONFIG (cell 2),
    # while cell 50 reads it back via RUN.get and holds the build call. Targeting one cell for both
    # is what the first version of this script did, and its assertion caught the no-op.
    cfg_cell = guard_cell = None
    for i, c in enumerate(nb["cells"]):
        t = "".join(c.get("source", []))
        if OLD_BUDGET in t:
            cfg_cell = i
        if OLD_BUILD in t and "_EFF_BUDGET_S" in t:
            guard_cell = i
    print("budget default in cell %s ; build call in cell %s" % (cfg_cell, guard_cell))
    if cfg_cell is None or guard_cell is None:
        print("!! could not locate both edit sites -- refusing to guess")
        return 2

    cfg_src = "".join(nb["cells"][cfg_cell]["source"])
    guard_src = "".join(nb["cells"][guard_cell]["source"])
    n_b = cfg_src.count(OLD_BUDGET)
    n_g = guard_src.count(OLD_BUILD)
    print("match counts: budget default=%d  build call=%d" % (n_b, n_g))
    if n_b != 1 or n_g != 1:
        print("!! expected exactly 1 each -- refusing to guess")
        return 3
    if check:
        print("--check: nothing written.")
        return 0

    new_cfg = cfg_src.replace(OLD_BUDGET, NEW_BUDGET, 1)
    new_guard = guard_src.replace(OLD_BUILD, NEW_BUILD, 1)
    assert NEW_BUDGET in new_cfg, "budget default did not change"
    assert "test cache build (pre-flight)" in new_guard, "pre-flight guard did not land"
    assert "_eff_build_t0" in new_guard, "build timing did not land"
    assert new_guard.count("_effb0mil_build_test_cache()") == 2, "build call count changed unexpectedly"

    nb["cells"][cfg_cell]["source"] = [new_cfg]
    nb["cells"][guard_cell]["source"] = [new_guard]
    n_cells = len(nb["cells"])
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    nb2 = json.loads(NB.read_text(encoding="utf-8", errors="replace"))
    assert len(nb2["cells"]) == n_cells, "cell count changed on write"
    ok = bad = 0
    for c in nb2["cells"]:
        if c.get("cell_type") != "code":
            continue
        try:
            compile("".join(c.get("source", [])), "<cell>", "exec")
            ok += 1
        except SyntaxError as e:
            print("!! SyntaxError: %s" % e)
            return 4
    print("wrote cells %d (config) and %d (guard); %d/%d code cells compile"
          % (cfg_cell, guard_cell, ok, ok))
    print("budget default 540 -> 1800 ; pre-flight guard added before the test-cache build")
    print("!! do NOT push while v3 is RUNNING -- push after it reaches a terminal status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
