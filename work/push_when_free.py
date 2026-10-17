"""Push a kernel as soon as the account has a free batch GPU session.

`Maximum batch GPU session count of 2 reached.` is a queueing condition, not a failure (A196 family):
the push must simply be retried until it is accepted.  A lost receipt ("Expecting value: line 1 column 1")
is NOT a failure either -- the session may exist anyway -- so this loop always asks the SERVER for the
kernel's status before deciding whether to push again.

Usage: python work/push_when_free.py <kernel-dir> [max_minutes]
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'state' / 'push_wait.txt'


def status(slug: str) -> str:
    """A kernel we own that EXISTS must never be re-pushed (each retry is a new version + a GPU run).

    `kaggle kernels status` answers 404 for a kernel created moments ago (A201), so existence is
    decided from ListKernels instead; the session status is only used to report RUNNING/COMPLETE.
    """
    try:
        from kstate import state as _exists
        ex = _exists(slug)
    except Exception as exc:  # noqa: BLE001
        ex = 'UNKNOWN (%r)' % (exc,)
    if ex.startswith('EXISTS'):
        r = subprocess.run(['kaggle', 'kernels', 'status', slug], capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=300)
        out = (r.stdout or '') + (r.stderr or '')
        m = re.search(r'has status "KernelWorkerStatus\.(\w+)"', out)
        return m.group(1) if m else 'EXISTS'
    return 'ABSENT' if ex == 'ABSENT' else 'UNKNOWN'


def say(msg: str) -> None:
    line = '%s  %s' % (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), msg)
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    d = Path(argv[1])
    if not d.is_absolute():
        d = ROOT / d
    minutes = float(argv[2]) if len(argv) > 2 else 90.0
    import json
    slug = json.loads((d / 'kernel-metadata.json').read_text(encoding='utf-8'))['id']
    say('waiting to push %s (limit %g min)' % (slug, minutes))
    t_end = time.time() + minutes * 60
    attempt = 0
    while time.time() < t_end:
        st = status(slug)
        if st in ('COMPLETE', 'ERROR', 'RUNNING', 'EXISTS'):
            say('%s EXISTS with status %s -- nothing to push' % (slug, st))
            return 0
        attempt += 1
        r = subprocess.run([sys.executable, str(ROOT / 'work' / 'push_kernel.py'), str(d)],
                           capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800)
        out = (r.stdout or '') + (r.stderr or '')
        if 'PUSH OK' in out:
            say('PUSH OK on attempt %d' % attempt)
            print(out.strip()[-800:])
            return 0
        tail = out.strip().splitlines()[-1] if out.strip() else '(no output)'
        # a lost receipt: ask the server (freshly) before believing the failure
        time.sleep(20)
        try:
            from kstate import forget as _forget
            _forget()
        except Exception:  # noqa: BLE001
            pass
        st2 = status(slug)
        if st2 in ('RUNNING', 'COMPLETE', 'ERROR', 'EXISTS'):
            say('push receipt was lost but the server has %s = %s -- treating as pushed' % (slug, st2))
            return 0
        say('attempt %d not accepted (%s) -- server says %s' % (attempt, tail[:120], st2))
        time.sleep(90)
    say('GAVE UP after %g min' % minutes)
    return 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
