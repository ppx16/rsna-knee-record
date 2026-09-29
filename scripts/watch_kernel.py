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
            api.authenticate()
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
