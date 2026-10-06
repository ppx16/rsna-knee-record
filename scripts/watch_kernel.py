"""Poll a Kaggle kernel until it reaches a terminal status, surviving network flakiness.

WHY THIS IS A FILE AND NOT A ONE-LINER
    The first version died before its first poll with
        SSLError ... api.kaggle.com ... SSLEOFError: UNEXPECTED_EOF_WHILE_READING
    raised inside `api.authenticate()` -> `_authenticate_with_access_token()` ->
    `_introspect_token()`.  That path is a single point of failure: authentication itself makes a
    network call, so a transient TLS failure looks exactly like "the poll is broken".  The fix is
    to retry the CONNECT as well as the poll, and never let one bad handshake end the watch.

Usage:
    python scripts/watch_kernel.py ppxl16/rsna-knee-tpu-3dsmoke [--minutes 240]
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
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

TERMINAL = ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED')


def connect(tries=10, delay=10):
    from kaggle.api.kaggle_api_extended import KaggleApi
    for a in range(tries):
        try:
            api = KaggleApi()
            # ⚠️ RETRY THE AUTH CALL ITSELF, not just the import. This line authenticates ONCE at module scope, before
            # any polling loop, so a transient api.kaggle.com SSLEOFError here is instant death -- and for a watcher that
            # means the result it waited hours for is never read. See work/patch_auth_guard.py.
            _auth_last = None
            for _auth_try in range(6):
                try:
                    # ⚠️ RETRY THE AUTH CALL ITSELF, not just the import. This line authenticates ONCE at module scope, before
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
                            _auth_sys.stderr.write("api.authenticate() attempt %d failed: %s\n"
                                                   % (_auth_try + 1, type(_auth_e).__name__))
                            _auth_sys.stderr.flush()
                            _auth_time.sleep(8)
                    else:
                        _auth_sys.stderr.write("COULD NOT AUTHENTICATE after 6 attempts (%r) -- TRANSPORT failure, "
                                               "not a result.\n" % (_auth_last,))
                        raise SystemExit(2)
                    break
                except Exception as _auth_e:  # noqa: BLE001
                    _auth_last = _auth_e
                    print("api.authenticate() attempt %d failed: %s" % (_auth_try + 1, type(_auth_e).__name__),
                          file=sys.stderr, flush=True)
                    time.sleep(8)
            else:
                print("COULD NOT AUTHENTICATE after 6 attempts (%r) -- TRANSPORT failure, not a result."
                      % (_auth_last,), file=sys.stderr)
                raise SystemExit(2)
            return api
        except Exception as e:
            print('  connect retry %d/%d: %s' % (a + 1, tries, type(e).__name__), flush=True)
            time.sleep(delay)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('kernel')
    ap.add_argument('--minutes', type=float, default=240)
    a = ap.parse_args()

    api = connect()
    if api is None:
        print('COULD NOT AUTHENTICATE -- not a kernel problem, a network one')
        return 2

    t0 = time.time()
    last = None
    while time.time() - t0 < a.minutes * 60:
        try:
            st = api.kernels_status(a.kernel)
            s = getattr(st, 'status', None)
            name = getattr(s, 'name', s)
            failure = str(getattr(st, 'failure_message', '') or '')
        except Exception as e:
            name = 'NETWORK-ERR(%s)' % type(e).__name__
            failure = ''
            api = connect(tries=3, delay=8) or api

        if name != last:
            print('%6.0fs  %s' % (time.time() - t0, name), flush=True)
            last = name
        if name in TERMINAL:
            if failure:
                print('failure: %s' % failure[:500], flush=True)
            print('TERMINAL %s' % name, flush=True)
            return 0
        time.sleep(60)

    print('still not terminal after %.0f min' % a.minutes, flush=True)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
