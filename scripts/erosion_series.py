"""Erosion series from the persisted lb_snapshot_*.json files.

Each snapshot already carries our_rank, N, the bronze cut and the band we were in, so this needs
no network at all -- which matters because work/bronze_forecast.py re-downloads the board and can
be slow.  The question it answers is the one that decides strategy: is the bar rising faster than
teams are arriving?
"""
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

snaps = []
for p in glob.glob(str(ROOT / 'state/lb_snapshot_*.json')):
    try:
        d = json.load(open(p, encoding='utf-8'))
    except Exception:
        continue
    d['_file'] = os.path.basename(p)
    snaps.append(d)
snaps.sort(key=lambda d: d.get('at', ''))

print('%-22s %6s %8s %8s %8s %8s  %s' %
      ('at', 'N', 'our_rank', 'score', 'bronze', 'margin', 'our band (score: lo..hi n)'))
for d in snaps[-14:]:
    b = d.get('block') or {}
    print('%-22s %6s %8s %8s %8s %8s  %s: %s..%s n=%s' %
          (d.get('at'), d.get('n_teams'), d.get('our_rank'), d.get('our_score'),
           d.get('bronze_line'), d.get('margin'),
           b.get('score'), b.get('lo'), b.get('hi'), b.get('n')))

if len(snaps) >= 2:
    a, z = snaps[0], snaps[-1]
    import datetime as dt
    def parse(s):
        return dt.datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc)
    hours = (parse(z['at']) - parse(a['at'])).total_seconds() / 3600.0
    print('\nover %.1f h:  N %s -> %s (%+d)   our rank %s -> %s (%+d)   margin %s -> %s (%+d)'
          % (hours, a['n_teams'], z['n_teams'], z['n_teams'] - a['n_teams'],
             a['our_rank'], z['our_rank'], z['our_rank'] - a['our_rank'],
             a['margin'], z['margin'], z['margin'] - a['margin']))
    if hours > 0:
        print('  rate: our rank %+.1f / h, margin %+.1f / h, N %+.1f / h'
              % ((z['our_rank'] - a['our_rank']) / hours,
                 (z['margin'] - a['margin']) / hours,
                 (z['n_teams'] - a['n_teams']) / hours))
    # the LAST pair only, which is the current speed
    y = snaps[-2]
    h2 = (parse(z['at']) - parse(y['at'])).total_seconds() / 3600.0
    if h2 > 0:
        print('  last interval (%.1f h): our rank %+d, margin %+d, N %+d'
              % (h2, z['our_rank'] - y['our_rank'], z['margin'] - y['margin'],
                 z['n_teams'] - y['n_teams']))
