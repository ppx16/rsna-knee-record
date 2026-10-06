# -*- coding: utf-8 -*-
"""Catch `NameError`-class bugs locally, before burning a Kaggle round-trip.

WHY THIS EXISTS
    Two kernels in one day died on names that were never defined:

      * `_SLOTS44` / `_SLOTS64` in `kernel-diag-slots/diag_slots.py` -- they exist ONLY
        inside the module DOCSTRING, written as a code example.  `grep '_SLOTS44'` finds
        them and reads as proof they are defined.  It is not.  (py_compile passes too:
        it only checks syntax, never name resolution.)
      * the same file's earlier `or [0]` fallback, which turned a failed lookup into a
        printed "finding" rather than an error.

    Kaggle gives no way to test a kernel without a queue round-trip, so the only cheap
    place to catch this is here.

WHAT IT CHECKS
    Every symbol that a scope resolves as GLOBAL must be bound at module level, be a
    builtin, or be in the allowlist.  That is exactly the `NameError` class: a name read
    but never assigned anywhere in the module.  Closure variables (`is_free`) and locals
    are correctly ignored, so nested helpers do not false-positive.

    It does NOT check attribute names, types, or anything runtime.  It is a cheap net,
    not a substitute for running.

    python scripts/check_names.py kernel-diag-slots/diag_slots.py
    python scripts/check_names.py kernel-tpu-cv/*.py kernel-cv-r288-fold1/*.py
"""
from __future__ import annotations


# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded
# `print("... \u26a0 ...")` raises UnicodeEncodeError at that line and kills the script MID-REPORT, while
# everything printed earlier is already on screen -- so the run looks complete. Self-contained on purpose:
# it imports `sys` itself and depends on nothing else in this tree.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream: leave it alone, prints still degrade
    pass
# ----------------------------------------------------------------------------------------------------
import builtins
import pathlib
import symtable
import sys

# Names the interpreter itself provides at module level; never "undefined".
ALLOW = {"__name__", "__file__", "__doc__", "__package__", "__spec__",
         "__loader__", "__builtins__", "__debug__"}


def module_bindings(st: symtable.SymbolTable) -> set[str]:
    """Everything bound at module level: assignments, imports, defs, classes."""
    out = set()
    for s in st.get_symbols():
        if s.is_assigned() or s.is_imported() or s.is_namespace():
            out.add(s.get_name())
    return out


def walk(st: symtable.SymbolTable, bound: set[str], path: str, problems: list):
    for s in st.get_symbols():
        name = s.get_name()
        if name in ALLOW:
            continue
        # Only globals can NameError at runtime; free vars are closures.
        if s.is_global() and name not in bound and not hasattr(builtins, name):
            problems.append((path, st.get_name(), name, st.get_lineno()))
    for child in st.get_children():
        walk(child, bound, "%s.%s" % (path, child.get_name()), problems)


def main() -> int:
    files = [pathlib.Path(a) for a in sys.argv[1:]]
    if not files:
        print(__doc__)
        return 2
    bad = 0
    for p in files:
        try:
            src = p.read_text(encoding="utf-8")
        except Exception as e:
            print("SKIP  %s (%s)" % (p, e))
            continue
        try:
            st = symtable.symtable(src, str(p), "exec")
        except SyntaxError as e:
            print("SYNTAX %s: %s" % (p, e))
            bad += 1
            continue
        bound = module_bindings(st)
        problems: list = []
        walk(st, bound, "<module>", problems)
        if problems:
            bad += 1
            print("FAIL  %s" % p)
            for _f, scope, name, line in problems:
                print("        %s: %s is used as a global but never bound (near line %s)"
                      % (scope, name, line))
        else:
            print("ok    %s (%d module-level names)" % (p, len(bound)))
    print("\n%d file(s) with undefined-global problems" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
