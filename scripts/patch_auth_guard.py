"""Give every WATCHER a retry around `api.authenticate()`, so a transient outage cannot kill a long poll.

WHY. `api.kaggle.com` intermittently answers OAuth with `SSLEOFError: UNEXPECTED_EOF_WHILE_READING`. Measured
twice today. Every script here authenticates ONCE at module scope, before its polling loop, so a failure on
that one line is instant death -- and for a watcher, death means the result it was waiting hours for is
never read.

⚠️ A WATCHER WITH TWO AUTHENTICATION POINTS AND ONE GUARD HAS NO GUARD. `collect_results.py` proved this the
hard way: its import was wrapped in a retry and the import retry SUCCEEDED, and then the bare
`api.authenticate()` a few lines below died of the same error. Searching a file for one retry loop and
concluding "guarded" is exactly the declaration-instead-of-artefact mistake this repository keeps repeating,
so this patcher rewrites the CALL SITE and nothing else.

⚠️ It does NOT touch scripts that are already running: a running process has already passed this line
successfully, and restarting it would discard hours of polling. This only makes FUTURE runs survivable.

Usage:
    python work/patch_auth_guard.py --check
    python work/patch_auth_guard.py --apply
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
import argparse
import pathlib
import re
import sys

# Only the long-running watchers and the diagnostics that a handover actually depends on. Deliberately an
# allowlist: rewriting an arbitrary script's auth site is not worth the risk of getting one wrong.
TARGETS = [
    "watch_rdr_submission.py",
    "watch_kernel.py",
    "watch_scores.py",
    "watch_msft2_folds.py",
    "watch_reader_arm.py",
    "lb_snapshot.py",
    "check_running.py",
    "check_subs.py",
    "read_finished_runs.py",
    "round_state.py",
    "state_read.py",
]

GUARD = '''# ⚠️ RETRY THE AUTH CALL ITSELF, not just the import. This line authenticates ONCE at module scope, before
# any polling loop, so a transient api.kaggle.com SSLEOFError here is instant death -- and for a watcher that
# means the result it waited hours for is never read. See work/patch_auth_guard.py.
# Self-contained on purpose: it imports what it needs, so it does not depend on the host file's imports.
import sys as _auth_sys
import time as _auth_time

_auth_last = None
for _auth_try in range(6):
    try:
        api.authenticate()
        break
    except Exception as _auth_e:  # noqa: BLE001
        _auth_last = _auth_e
        _auth_sys.stderr.write("api.authenticate() attempt %d failed: %s\\n"
                               % (_auth_try + 1, type(_auth_e).__name__))
        _auth_sys.stderr.flush()
        _auth_time.sleep(8)
else:
    _auth_sys.stderr.write("COULD NOT AUTHENTICATE after 6 attempts (%r) -- TRANSPORT failure, "
                           "not a result.\\n" % (_auth_last,))
    raise SystemExit(2)
'''

MARK = "RETRY THE AUTH CALL ITSELF"
CALL = re.compile(r"^(?P<indent>[ \t]*)api\.authenticate\(\)[ \t]*$", re.M)


def needs_guard(text: str) -> int:
    n = 0
    for m in CALL.finditer(text):
        tail = text[m.end():m.end() + 900]
        if MARK in tail:
            continue
        n += 1
    return n


def patch(text: str) -> tuple[str, int]:
    out = []
    pos = 0
    applied = 0
    for m in CALL.finditer(text):
        tail = text[m.end():m.end() + 900]
        if MARK in tail:
            continue
        indent = m.group("indent")
        block = "\n".join((indent + ln) if ln else "" for ln in GUARD.rstrip("\n").split("\n"))
        out.append(text[pos:m.start()])
        out.append(block)
        pos = m.end()
        applied += 1
    out.append(text[pos:])
    return "".join(out), applied


def has_sys_and_time(text: str) -> list[str]:
    missing = []
    if not re.search(r"^import sys$|^import sys\b", text, re.M) and "import sys" not in text:
        missing.append("sys")
    if not re.search(r"^import time$|^import time\b", text, re.M) and "import time" not in text:
        missing.append("time")
    return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    # Locate each target wherever it actually lives. The first run printed a wall of MISSING because it
    # probed both work/ and scripts/ for every name; the watchers are in work/ and watch_kernel.py /
    # watch_scores.py are in scripts/.
    found = {}
    for d in ("work", "scripts"):
        root = pathlib.Path(d)
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*.py")):
            if p.name in TARGETS:
                found.setdefault(p.name, p)

    for name in TARGETS:
        p = found.get(name)
        if p is None:
            print("MISSING  %s" % name)
            continue
        text = p.read_text(encoding="utf-8")
        n = needs_guard(text)
        if n == 0:
            print("already  %s" % p)
            continue
        new, applied = patch(text)
        try:
            compile(new, str(p), "exec")
        except Exception as exc:  # noqa: BLE001
            print("FAILED   %s -- %r" % (p, exc))
            continue
        if a.apply:
            p.write_text(new, encoding="utf-8")
            print("PATCHED  %s (%d call site(s))" % (p, applied))
        else:
            print("WOULD    %s (%d call site(s))" % (p, applied))
    return 0


if __name__ == "__main__":
    sys.exit(main())
