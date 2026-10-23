"""Hourly self-check: has anything I launched finished, failed, or silently stalled?

Why this exists: the user is away and asked for an hourly status check.  A background job that polls
is the honest version -- it logs a timestamped line per poll so the record shows WHEN a kernel settled,
not just when I happened to look.

What it does, every POLL_S seconds, for every kernel of ours that ran in the last LOOKBACK_H hours:
  * prints/logs the worker status (COMPLETE / ERROR / RUNNING / QUEUED);
  * when a kernel settles, downloads its output and reports the verdict files it knows about
    (`*leg_report.json`, `blend_receipt.json`, `submission_receipt.json`, `S6_*`/`MIRROR_*` lines);
  * never submits anything (A193: Create*/Submit* are actions, not reads).

Usage: python work/hourly_watch.py [hours_lookback] [poll_seconds]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'state' / 'kernel_watch.txt'
OUT = ROOT / 'scratch' / 'watch_out'
POLL_S = 3600
LOOKBACK_H = 48


def say(msg: str) -> None:
    line = '%s  %s' % (datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), msg)
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
    return 'UNKNOWN' if 'Not Found' not in out else 'ABSENT'


def mine_recent(hours: int) -> list[str]:
    sys.path.insert(0, str(ROOT / 'work'))
    import kstate
    ks = kstate.mine()
    cut = time.time() - hours * 3600
    out = []
    for slug, k in ks.items():
        t = str(k.get('lastRunTime') or '')[:19]
        try:
            ts = datetime.strptime(t, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            continue
        if ts >= cut:
            out.append('ppxl16/' + slug)
    return sorted(out)


def verdict(slug: str) -> str:
    d = OUT / slug.split('/')[-1]
    d.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        subprocess.run(['kaggle', 'kernels', 'output', slug, '-p', str(d)],
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800)
        if any(d.glob('*report.json')) or any(d.glob('blend_receipt.json')):
            break
        time.sleep(20)
    bits = []
    for f in sorted(d.glob('*leg_report.json')):
        try:
            b = json.loads(f.read_text(encoding='utf-8'))
            bits.append('%s=%s' % (f.name.replace('_leg_report.json', ''), b.get('status')))
        except Exception:
            bits.append('%s=unreadable' % f.name)
    br = d / 'blend_receipt.json'
    if br.is_file():
        try:
            b = json.loads(br.read_text(encoding='utf-8'))
            bits.append('blend a224=%s rows=%s' % (b.get('a224'), b.get('rows')))
        except Exception:
            bits.append('blend=unreadable')
    lg = sorted(d.glob('*.log'))
    if lg:
        txt = lg[-1].read_text(encoding='utf-8', errors='replace')
        for key in ('S6_STAGE_COMPLETE', 'S6_STAGE_FALLBACK', 'MIRROR_STAGE_COMPLETE',
                    'MIRROR_STAGE_FALLBACK', 'NOTEBOOK_COMPLETE', 'OURS_STAGE_COMPLETE'):
            for line in txt.splitlines():
                if key in line:
                    bits.append(line.split(key, 1)[1][:170])
                    break
    return ' | '.join(bits) if bits else '(no verdict artifact yet)'


def main(argv: list[str]) -> int:
    hours = float(argv[1]) if len(argv) > 1 else LOOKBACK_H
    poll = float(argv[2]) if len(argv) > 2 else POLL_S
    say('WATCH START  lookback=%.0fh  poll=%.0fs  (log: %s)' % (hours, poll, LOG.name))
    seen: dict[str, str] = {}
    while True:
        try:
            names = mine_recent(hours)
            if not names:
                say('no kernels in the last %.0fh' % hours)
            for slug in names:
                st = status(slug)
                if seen.get(slug) != st:
                    seen[slug] = st
                    say('%-46s %s' % (slug.split('/')[-1], st))
                    if st in ('COMPLETE', 'ERROR'):
                        say('   verdict: %s' % verdict(slug))
                elif st == 'RUNNING':
                    say('%-46s still RUNNING' % slug.split('/')[-1])
        except Exception as exc:  # noqa: BLE001
            say('poll error: %r' % (exc,))
        time.sleep(poll)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
