"""Read the hourly closing-watch log and project the bronze margin at the deadline.

Why a tool rather than a glance: the endgame projections have been wrong twice (C208's -38 from a window that
straddled the band change, C210's +5 from a window that averaged an acceleration). This one uses the only
dense series we have -- one reading an hour for the last day -- and reports the rate on several windows so a
single bad window cannot masquerade as the answer.

Reads state/closing_watch.txt; writes nothing.
"""
import io
import re
import statistics as st
import sys
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
LOG = r'D:\kaggle\rsna-knee\state\closing_watch.txt'
DEADLINE = datetime(2026, 10, 22, 23, 59, tzinfo=timezone.utc)

RANK = re.compile(r'^(\S+Z)\s+RSNA us\s+:\s+rank\s+(\d+)\s+score\s+([\d.]+)')
MARG = re.compile(r'^(\S+Z)\s+RSNA bronze\s+:\s+cut\s+(\d+)\s+->\s+margin\s+([+-]\d+)')

reads = {}
for line in io.open(LOG, encoding='utf-8', errors='replace'):
    line = line.rstrip('\n')
    m = RANK.match(line)
    if m:
        t = datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        reads.setdefault(t, {})['rank'] = int(m.group(2))
        reads[t]['score'] = m.group(3)
        continue
    m = MARG.match(line)
    if m:
        t = datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        reads.setdefault(t, {})['cut'] = int(m.group(2))
        reads[t]['margin'] = int(m.group(3))

series = sorted((t, d) for t, d in reads.items() if 'margin' in d)
if len(series) < 3:
    raise SystemExit('not enough readings (%d)' % len(series))

first_t, first = series[0]
last_t, last = series[-1]
span_h = (last_t - first_t).total_seconds() / 3600
left_h = (DEADLINE - last_t).total_seconds() / 3600

print('readings: %d   %s .. %s   (%.1f h)'
      % (len(series), first_t.strftime('%m-%d %H:%M'), last_t.strftime('%m-%d %H:%M'), span_h))
print('rank   %s -> %s      margin %+d -> %+d      cut %s -> %s'
      % (first.get('rank'), last.get('rank'), first['margin'], last['margin'],
         first.get('cut'), last.get('cut')))

print('\n%-18s %-8s %-9s %s' % ('window', 'hours', 'rate/h', 'projected margin at the deadline'))
for label, n in (('all readings', len(series)), ('last 8', 8), ('last 4', 4)):
    sub = series[-n:]
    h = (sub[-1][0] - sub[0][0]).total_seconds() / 3600
    if h <= 0:
        continue
    rate = (sub[-1][1]['margin'] - sub[0][1]['margin']) / h
    print('%-18s %-8.1f %-9.2f %+.0f' % (label, h, rate, last['margin'] + rate * left_h))

margins = [d['margin'] for _, d in series]
print('\nmargin series: %s' % ' '.join('%+d' % m for m in margins))
diffs = [b - a for a, b in zip(margins, margins[1:])]
if diffs:
    print('hour-to-hour changes: %s' % ' '.join('%+d' % x for x in diffs))
    zero = sum(1 for x in diffs if x == 0)
    print('flat hours: %d of %d   mean change: %.2f   sd: %.2f'
          % (zero, len(diffs), st.mean(diffs), st.stdev(diffs) if len(diffs) > 1 else 0))
print('\nhours to deadline: %.2f' % left_h)
