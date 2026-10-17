"""Morning routine: one command that answers "what happened, what is ready, what should I submit".

Reads only the server and our own run artifacts.  It NEVER submits -- printing the recommendation
and doing the submission are deliberately separate (A193/A200: a Create*/Submit* call is an ACTION).

    1. which submission the platform is crediting right now, our in-band position, medal margins
       (so a new submission is judged by "did LastSubmissionDate move" and "did our position move",
       never by the displayed score -- C173);
    2. today's submission count, counted server-side (never from our own ledger -- C171);
    3. every kernel we own that is COMPLETE and whose leg report says "blended", i.e. safe to submit;
    4. the pre-registered reading rule for each.

Usage: python work/morning.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work'))
import kstate  # noqa: E402


def sh(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
    return (r.stdout or '') + (r.stderr or '')


def main() -> int:
    print('=' * 78)
    print('1. WHICH SUBMISSION IS CREDITED  (work/which_selected.py)')
    print('=' * 78)
    r = subprocess.run([sys.executable, str(ROOT / 'work' / 'which_selected.py')],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(((r.stdout or '') + (r.stderr or '')).strip())

    print()
    print('=' * 78)
    print('2. TODAY\'S SUBMISSIONS (counted server-side, not from our ledger)')
    print('=' * 78)
    out = sh(['kaggle', 'competitions', 'submissions', '-c', 'rsna-knee-abnormality-detection', '--csv'])
    lines = [ln for ln in out.splitlines() if ln and not ln.startswith('Warning') and ',' in ln]
    body = [ln for ln in lines if not ln.startswith('ref,')]
    today = [ln for ln in body if ln.split(',')[2].startswith(__import__('time').strftime('%Y-%m-%d', __import__('time').gmtime()))]
    print('rows today: %d / 5   -> %d left' % (len(today), max(0, 5 - len(today))))
    for ln in today:
        f = ln.split(',')
        print('   %s  %s  %s' % (f[0], f[2][:19], f[4]))

    print()
    print('=' * 78)
    print('3. KERNELS READY TO SUBMIT (COMPLETE + the leg really blended)')
    print('=' * 78)
    for name, k in sorted(kstate.mine().items()):
        if not (name.startswith('rsna-knee-r7-c224')):
            continue
        run = str(k.get('lastRunTime') or '')[:19]
        st = sh(['kaggle', 'kernels', 'status', 'ppxl16/' + name])
        state = 'UNKNOWN'
        for token in ('COMPLETE', 'ERROR', 'RUNNING'):
            if token in st:
                state = token
        print('%-42s %-9s run=%s' % (name, state, run))
    print()
    print('leg reports (a "fallback" means this artifact equals the 0.956 we already hold):')
    for d in sorted((ROOT / 'scratch').glob('out_*')):
        rep = list(d.glob('*leg_report.json'))
        for f in rep:
            try:
                b = json.loads(f.read_text(encoding='utf-8'))
                print('   %-58s %s w=%s cov=%s' % (d.name, b.get('status'), b.get('w'), b.get('coverage')))
            except Exception as exc:  # noqa: BLE001
                print('   %-58s unreadable (%r)' % (d.name, exc))
    print()
    print('reading rule: >=0.957 -> adopt as the new tick-1; =0.956 but LastSubmissionDate moves to it')
    print('-> our in-band position moved (C173); below that the platform keeps our best anyway.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
