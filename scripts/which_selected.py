"""Which submission is Kaggle actually SELECTING for us, and did our in-band position move?

Two 0.956 submissions can differ in the UNROUNDED score, and the leaderboard's ties are ordered by
that unrounded value (FACTS C113/C141).  So "score unchanged, rank improved" is exactly the
signature of switching to a higher-unrounded 0.956 -- and the `LastSubmissionDate` on our own row
says which submission the platform is currently crediting.
"""
import csv
import glob
import io
import zipfile
from datetime import datetime, timezone

LB = sorted(glob.glob('D:/kaggle/rsna-knee/work/lb_live/*.zip')) or \
     sorted(glob.glob('D:/kaggle/rsna-knee/state/lb_full/*.zip'))
for zp in LB[-1:]:
    z = zipfile.ZipFile(zp)
    stamp = z.namelist()[0][-24:-4]
    raw = z.read(z.namelist()[0]).decode('utf-8-sig', errors='replace')
    rows = []
    for r in csv.DictReader(io.StringIO(raw)):
        try:
            rows.append({'rank': int(r['Rank']), 'score': float(r['Score']),
                         'team': (r.get('TeamName') or '').strip(),
                         'members': (r.get('TeamMemberUserNames') or '').strip(),
                         'last': (r.get('LastSubmissionDate') or '').strip(),
                         'subs': (r.get('SubmissionCount') or '').strip()})
        except (ValueError, KeyError):
            continue
    rows.sort(key=lambda d: d['rank'])
    n = len(rows)
    me = next((d for d in rows if 'ppxl16' in d['members'].lower()), None)
    print('board read %s   N=%d' % (stamp, n))
    if not me:
        print('us: NOT FOUND')
        raise SystemExit(0)
    band = [d for d in rows if d['score'] == me['score']]
    lo, hi = band[0]['rank'], band[-1]['rank']
    print('US: rank %d  score %.3f  subs=%s' % (me['rank'], me['score'], me['subs']))
    print('    LastSubmissionDate = %s   <-- OUR MOST RECENT SUBMISSION (NOT the credited one!)' % me['last'])
    print('        CORRECTION 2026-10-16 (FACTS C189): this column is simply the newest submission of')
    print('        the team. Every observation matches that reading, so it does NOT say which entry the')
    print('        platform credits. The only per-submission evidence is that entry\'s own publicScore')
    print('        column (kaggle competitions submissions --csv); the unrounded order inside one')
    print('        displayed band is NOT observable from the board.')
    print('    our band %.3f = ranks %d..%d (n=%d) -> position %d/%d'
          % (me['score'], lo, hi, len(band), me['rank'] - lo + 1, len(band)))
    cut = int(n * 0.05)
    print('    silver cut %d -> margin %+d ; bronze cut %d -> margin %+d'
          % (cut, cut - me['rank'], int(n * 0.10), int(n * 0.10) - me['rank']))
    # who is immediately around us, to see whether our neighbours' timestamps changed
    i = rows.index(me)
    print('    neighbours:')
    for d in rows[max(0, i - 2):i + 3]:
        mark = '  <== US' if d is me else ''
        print('      rank %4d  %.3f  last %s  %s%s' % (d['rank'], d['score'], d['last'], d['team'][:26], mark))
