"""Poll a Kaggle kernel until it leaves the RUNNING state, appending every reading to a log.

Usage: python work/watch_kernel.py <owner/slug> <tag> [max_minutes]
Reads the CLI's literal status line; never infers state from an exit code.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def status(ref: str) -> str:
    for attempt in range(4):
        r = subprocess.run(['kaggle', 'kernels', 'status', ref],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        out = ((r.stdout or '') + (r.stderr or '')).strip()
        if 'SSLError' in out or 'Max retries' in out:
            time.sleep(10 * (attempt + 1))
            continue
        return out.splitlines()[-1] if out else '(empty)'
    return '(network) ' + out.splitlines()[-1]


def main() -> int:
    ref, tag = sys.argv[1], sys.argv[2]
    max_min = float(sys.argv[3]) if len(sys.argv) > 3 else 700.0
    log = ROOT / 'state' / ('watch_%s.txt' % tag)
    t0 = time.time()
    last = None
    with log.open('a', encoding='utf-8') as fh:
        while (time.time() - t0) / 60.0 < max_min:
            s = status(ref)
            line = '%s  %s' % (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), s)
            fh.write(line + '\n')
            fh.flush()
            if s != last:
                print(line, flush=True)
                last = s
            m = re.search(r'status\s+(\w+)', s)
            st = m.group(1) if m else ''
            if st and st.upper() not in ('RUNNING', 'QUEUED', ''):
                print('TERMINAL: %s' % s, flush=True)
                return 0
            time.sleep(300)
        print('TIMEOUT after %.0f min; last=%s' % (max_min, last), flush=True)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
