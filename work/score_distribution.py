"""What does the score distribution above us actually look like?

Everything so far has been rank-and-margin bookkeeping. This instead reads the full leaderboard we downloaded
and reports the shape of the top: how many teams sit at each displayed score above ours, how wide the bands
are, and what the next medal actually costs in score rather than in places.

Reads the newest CSV under state/lb_final/; writes nothing.
"""
import collections
import csv
import io
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
LB = Path(r'D:\kaggle\rsna-knee\state\lb_final')

csvs = sorted(LB.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
if not csvs:
    raise SystemExit('no leaderboard CSV under %s -- run the deadline report first' % LB)
path = csvs[0]
rows = list(csv.DictReader(io.open(path, encoding='utf-8-sig')))
print('read %s (%d rows)' % (path.name, len(rows)))

by_score = collections.Counter()
for r in rows:
    s = (r.get('Score') or '').strip()
    if s:
        by_score[float(s)] += 1
n = len(rows)
scored = sum(by_score.values())
print('teams: %d, with a score: %d\n' % (n, scored))

ours = None
for r in rows:
    if 'ppxl16' in ' '.join(str(v) for v in r.values()):
        ours = float(r['Score'])
        break
print('our displayed score: %s\n' % ours)

print('%-8s %-7s %-9s %s' % ('score', 'teams', 'cumulative', 'note'))
cum = 0
for s in sorted(by_score, reverse=True):
    cum += by_score[s]
    if s < (ours or 0) - 0.004:
        break
    note = ''
    if ours is not None and abs(s - ours) < 1e-9:
        note = '<- us'
    print('%-8.3f %-7d %-9d %s' % (s, by_score[s], cum, note))

if ours is not None:
    above = sum(c for s, c in by_score.items() if s > ours)
    print('\nteams strictly above us: %d  (they hold scores %.3f and up)'
          % (above, max((s for s in by_score if s > ours), default=ours)))
    print('teams in our band       : %d' % by_score[ours])
    print('teams strictly below us : %d' % (scored - above - by_score[ours]))
    nxt = max((s for s in by_score if s > ours), default=None)
    if nxt is not None:
        print('\nthe next displayed step is +%.3f. It would pass the %d teams above us.'
              % (nxt - ours, above))
        print('that is the whole gap between us and a higher band: %.3f of AUC.' % (nxt - ours))
