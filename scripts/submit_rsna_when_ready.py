"""Wait for a kernel run to finish, verify the leg actually blended, then submit it.

Submitting is an ACTION (A193): this script only fires when the server says the run is COMPLETE and
the run's own `r9all_leg_report.json` says `blended` -- a fallback run would be byte-identical to the
0.956 we already hold and is not worth one of the five daily slots.

Usage: python work/submit_rsna_when_ready.py <owner/slug> <version> "<message>" [max_minutes]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'state' / 'submit_when_ready.txt'


def say(msg: str) -> None:
    line = '%s  %s' % (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), msg)
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def status(slug: str) -> str:
    r = subprocess.run(['kaggle', 'kernels', 'status', slug], capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=300)
    out = (r.stdout or '') + (r.stderr or '')
    m = re.search(r'has status "KernelWorkerStatus\.(\w+)"', out)
    if m:
        return m.group(1)
    return 'ABSENT' if 'Not Found' in out else 'UNKNOWN'


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 2
    slug, version, message = argv[1], argv[2], argv[3]
    minutes = float(argv[4]) if len(argv) > 4 else 120.0
    out_dir = ROOT / 'scratch' / ('out_' + slug.split('/')[-1])
    t_end = time.time() + minutes * 60
    while time.time() < t_end:
        st = status(slug)
        if st in ('COMPLETE', 'ERROR'):
            say('%s finished with status %s' % (slug, st))
            if st != 'COMPLETE':
                say('not submitting: the run itself failed')
                return 1
            out_dir.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(['kaggle', 'kernels', 'output', slug, '-p', str(out_dir)],
                               capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800)
            rep = out_dir / 'r9all_leg_report.json'
            if not rep.is_file():
                say('not submitting: no r9all_leg_report.json in the output (%.200s)' % ((r.stdout or '') + (r.stderr or '')))
                return 1
            body = json.loads(rep.read_text(encoding='utf-8'))
            say('leg report: status=%s w=%s coverage=%s leg_seconds=%s'
                % (body.get('status'), body.get('w'), body.get('coverage'), body.get('leg_seconds')))
            if body.get('status') != 'blended':
                say('not submitting: the leg fell back, so this artifact equals the 0.956 we already hold')
                return 1
            s = subprocess.run([sys.executable, str(ROOT / 'work' / 'submit_code.py'), slug, version, message],
                               capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
            say((s.stdout or '')[-400:] + (s.stderr or '')[-200:])
            return 0 if 'SUBMITTED' in (s.stdout or '') else 1
        say('%s is %s -- waiting' % (slug, st))
        time.sleep(120)
    say('GAVE UP waiting for %s after %g min' % (slug, minutes))
    return 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
