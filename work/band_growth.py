"""Quantify the 0.958 band's growth from our own trend log.

Why: the whole endgame depended on one unexamined claim -- that the band inflates and therefore our position
decays. That was asserted from a handful of spot readings (C208, C210) and then corrected once when the rate
turned out to be estimated on the wrong window. The trend log now holds every reading the daily check has
taken, so the claim can be measured instead of eyeballed: is the growth linear, accelerating, or saturating,
and what does each imply for the close?

Reads state/daily_trend.jsonl only. Writes nothing.
"""
import io
import json
import statistics as st
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
P = r'D:\kaggle\rsna-knee\state\daily_trend.jsonl'

rows = []
for line in io.open(P, encoding='utf-8'):
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    at = d.get('at') or d.get('board_at')
    if not at:
        continue
    try:
        ts = datetime.fromisoformat(str(at).replace('Z', '+00:00'))
    except ValueError:
        continue
    band = d.get('our_band') or {}
    n_band = band.get('n') if isinstance(band, dict) else None
    lo = band.get('lo') if isinstance(band, dict) else None
    pos = (d.get('our_rank') - lo + 1) if (lo and d.get('our_rank')) else None
    rows.append({
        't': ts, 'n': d.get('n_teams'), 'rank': d.get('our_rank'),
        'bronze': d.get('margin_bronze'), 'band_n': n_band, 'band_pos': pos,
        'score': d.get('our_score'),
    })

rows.sort(key=lambda r: r['t'])
print('readings: %d   span: %s .. %s'
      % (len(rows), rows[0]['t'].strftime('%m-%d %H:%M'), rows[-1]['t'].strftime('%m-%d %H:%M')))

# The series contains TWO different bands: we moved from 0.956 to 0.958 on 10-16, and a "band growth rate"
# computed across that break is measuring two different populations. Split by the score we held.
bands = {}
for r in rows:
    score = r.get('score') or (rows[0].get('score'))
    bands.setdefault(score, []).append(r)
print('\nbands present in the log (by the score we held at the time):')
for score, sub in sorted(bands.items(), key=lambda kv: str(kv[0])):
    print('  %-8s %2d readings  %s .. %s'
          % (score, len(sub), sub[0]['t'].strftime('%m-%d %H:%M'), sub[-1]['t'].strftime('%m-%d %H:%M')))

current = None
for score, sub in bands.items():
    if sub is rows or sub[-1] is rows[-1]:
        current = (score, sub)
if current is None:
    current = sorted(bands.items(), key=lambda kv: kv[1][-1]['t'])[-1]
cur_score, cur = current
rows = cur
print('\n== analysing the band we actually hold now: %s (%d readings) ==\n' % (cur_score, len(rows)))

# only readings that carry the band count
banded = [r for r in rows if r['band_n']]
print('readings with a band count: %d' % len(banded))
if len(banded) < 2:
    raise SystemExit('not enough band readings to say anything')

print('\n%-14s %-6s %-6s %-7s %-7s %s' % ('when', 'N', 'rank', 'band n', 'in-band', 'bronze margin'))
for r in banded:
    print('%-14s %-6s %-6s %-7s %-7s %s'
          % (r['t'].strftime('%m-%d %H:%M'), r['n'], r['rank'], r['band_n'], r['band_pos'], r['bronze']))

# growth of the band, split into halves so an acceleration cannot hide inside one average
def rate(sub):
    if len(sub) < 2:
        return None
    (t0, a0), (t1, a1) = (sub[0]['t'], sub[0]['band_n']), (sub[-1]['t'], sub[-1]['band_n'])
    hours = (t1 - t0).total_seconds() / 3600
    return (a1 - a0) / hours if hours > 0 else None

half = len(banded) // 2
r1, r2 = rate(banded[:half + 1]), rate(banded[half:])
print('\nband growth: first half %.2f teams/h   second half %.2f teams/h' % (r1 or 0, r2 or 0))
if r1 and r2:
    if r2 > r1 * 1.25:
        print('  -> ACCELERATING (the second half is faster by more than 25%%)')
    elif r2 < r1 * 0.75:
        print('  -> DECELERATING (the second half is slower by more than 25%%)')
    else:
        print('  -> roughly LINEAR')

# our position inside the band, and the margin that actually matters
margins = [(r['t'], r['bronze']) for r in rows if r['bronze'] is not None]
if len(margins) >= 2:
    (t0, m0), (t1, m1) = margins[0], margins[-1]
    hours = (t1 - t0).total_seconds() / 3600
    slope = (m1 - m0) / hours if hours else 0
    print('\nbronze margin: %+.0f at %s  ->  %+.0f at %s   (%.2f places/hour)'
          % (m0, t0.strftime('%m-%d %H:%M'), m1, t1.strftime('%m-%d %H:%M'), slope))
    deadline = datetime.fromisoformat('2026-10-22T23:59:00+00:00')
    left = (deadline - t1).total_seconds() / 3600
    if left > 0:
        print('  projecting %.1f h to the deadline at that rate: %+.0f' % (left, m1 + slope * left))

# and the one number that says whether the band is what is hurting us
if len(banded) >= 2:
    (t0, p0), (t1, p1) = (banded[0]['t'], banded[0]['band_pos']), (banded[-1]['t'], banded[-1]['band_pos'])
    if p0 and p1:
        hours = (t1 - t0).total_seconds() / 3600
        print('\nin-band position: %d -> %d over %.0f h (%.2f places/hour inside the band)'
              % (p0, p1, hours, (p1 - p0) / hours if hours else 0))
        print('  -> if this is positive, the band is filling with teams ABOVE us, not merely growing')
