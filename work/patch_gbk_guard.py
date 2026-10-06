"""Insert the GBK console guard into every script whose `print` emits a char cp936 cannot encode.

WHY. This console is cp936/GBK. A `print("... ⚠️ ...")` raises UnicodeEncodeError AT THAT LINE, killing the
script mid-report -- and because everything printed BEFORE it is already on screen, the run LOOKS complete.
That is how `collect_results.py` silently reported only 1 of its 3 measurements.

The fix belongs in the files, not in an environment variable: an env var does not travel to another machine
or to Kaggle, and this repository is meant to be re-runnable from a bare checkout.

The guard goes immediately AFTER the module docstring, so it runs before any other statement, and it imports
`sys` itself so it never depends on the target script's imports.
"""
from __future__ import annotations

import argparse
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
import ast
import pathlib
import sys

BLOCK = '''
# --- GBK CONSOLE GUARD (see work/patch_gbk_guard.py) -------------------------------------------------
# This console is cp936, which cannot encode the U+26A0 warning sign used below. An unguarded
# `print("... \\u26a0 ...")` raises UnicodeEncodeError at that line and kills the script MID-REPORT, while
# everything printed earlier is already on screen -- so the run looks complete. Self-contained on purpose:
# it imports `sys` itself and depends on nothing else in this tree.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  -- non-reconfigurable stream: leave it alone, prints still degrade
    pass
# ----------------------------------------------------------------------------------------------------
'''

MARK = "GBK CONSOLE GUARD"


def at_risk(path: pathlib.Path) -> tuple[bool, list[str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    bad: set[str] = set()
    for line in text.splitlines():
        if "print(" not in line:
            continue
        for ch in line:
            try:
                ch.encode("cp936")
            except Exception:  # noqa: BLE001
                bad.add(ch)
    return bool(bad), sorted(bad)


def insert_after_docstring(text: str) -> str:
    """Return `text` with BLOCK placed after the docstring AND after any `from __future__` imports.

    The first version inserted right after the docstring only, and that BROKE `scripts/build_msft_kernel.py`:
    a `from __future__ import annotations` was pushed below the block, which is a SyntaxError
    ("from __future__ imports must occur at the beginning of the file").

    The safety net did not catch it: `ast.parse` does NOT raise on a misplaced future import, so the patcher
    happily wrote a file that cannot be compiled. Verified after the fact with `python -m py_compile`, which
    does raise. Never trust `ast.parse` as a compile check for this class of error.
    """
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    body = tree.body

    idx = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        idx = 1
    while idx < len(body) and isinstance(body[idx], ast.ImportFrom) and body[idx].module == "__future__":
        idx += 1

    insert_at = (body[idx].lineno - 1) if idx < len(body) else len(lines)
    head = "".join(lines[:insert_at])
    tail = "".join(lines[insert_at:])
    if head and not head.endswith("\n"):
        head += "\n"
    return head + BLOCK + tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--all",
        action="store_true",
        help="guard EVERY .py in work/ and scripts/, not only those whose SOURCE prints a bad character",
    )
    a = ap.parse_args()

    targets = []
    for d in ("work", "scripts"):
        root = pathlib.Path(d)
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*.py")):
            risky, bad = at_risk(p)
            if risky or a.all:
                targets.append((p, bad))

    # ⚠️ WHY `--all` EXISTS. The literal scan is STRUCTURALLY unable to find the worst cases: when a script
    # prints data that ARRIVES AT RUNTIME -- a downloaded leaderboard, a remote kernel title -- the offending
    # character is not in the source at all. That killed three runs today: collect_results.py on a literal
    # warning sign, then lb_snapshot.py on a team name containing U+0E3F, then the public-surface mining
    # command on a kernel title containing U+2B50.
    #
    # A user-level PYTHONIOENCODING is NOT a fix either, which was measured the hard way: it reaches only
    # processes started after it is set, and a long-running parent hands its STALE environment to every child
    # it spawns. So the guard has to live in the file.
    print("files to consider: %d%s" % (len(targets), "   (--all)" if a.all else ""))
    patched, skipped, failed = [], [], []
    for p, bad in targets:
        text = p.read_text(encoding="utf-8")

        # Strip any block a previous run of THIS patcher inserted, so a bad insertion is repairable and the
        # operation is idempotent. A hand-written guard from another script is recognised by `reconfigure`
        # and left alone.
        if MARK in text:
            text = text.replace(BLOCK, "", 1)
            if MARK in text:  # a second copy: strip repeatedly
                text = text.replace(BLOCK, "", 1)
        elif "reconfigure" in text:
            skipped.append(p.name)
            continue

        try:
            new = insert_after_docstring(text)
            # ⚠️ `compile` (NOT `ast.parse`) -- ast.parse does not reject a misplaced future import.
            compile(new, str(p), "exec")
        except Exception as exc:  # noqa: BLE001
            failed.append((p.name, repr(exc)))
            continue
        p.write_text(new, encoding="utf-8")
        patched.append(p.name)

    print("\nPATCHED  (%d): %s" % (len(patched), ", ".join(patched) or "-"))
    print("ALREADY  (%d): %s" % (len(skipped), ", ".join(skipped) or "-"))
    if failed:
        print("FAILED   (%d):" % len(failed))
        for n, e in failed:
            print("   ", n, e)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
