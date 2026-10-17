import csv, glob, datetime, itertools

p = sorted(glob.glob('scratch/lb/*.csv'))[0]
rows = list(csv.DictReader(open(p, encoding='utf-8')))
for r in rows:
    r['Rank'] = int(r['\ufeffRank'])
    r['Score'] = float(r['Score'])
    r['t'] = r['LastSubmissionDate']
rows.sort(key=lambda r: r['Rank'])

band = [r for r in rows if r['Score'] == 0.954]
print('0.954 band size', len(band), 'ranks', band[0]['Rank'], '..', band[-1]['Rank'])

# Is the band ordered by submission time ASC?
ts = [r['t'] for r in band]
mono = all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))
print('LastSubmissionDate monotonically non-decreasing with rank?', mono)
viol = sum(1 for i in range(len(ts) - 1) if ts[i] > ts[i + 1])
print('violations:', viol, 'of', len(ts) - 1)

# Spearman rank correlation between position in band and submission time
import math
def spearman(a, b):
    n = len(a)
    ra = {v: i for i, v in enumerate(sorted(set(a)))}
    rb = {v: i for i, v in enumerate(sorted(set(b)))}
    A = [ra[v] for v in a]; B = [rb[v] for v in b]
    ma = sum(A) / n; mb = sum(B) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(A, B))
    den = math.sqrt(sum((x - ma) ** 2 for x in A) * sum((y - mb) ** 2 for y in B))
    return num / den

print('spearman(position_in_band, submission_time) =', round(spearman(list(range(len(band))), ts), 4))

print()
print('first 8 of the band (earliest submits should be here if time-ordered):')
for r in band[:8]:
    print('  rank %4d  %s  %s' % (r['Rank'], r['t'], r['TeamName']))
print('...')
print('our row and its neighbours:')
for r in rows[725:735]:
    print('  rank %4d  %-10s %s  %s' % (r['Rank'], r['Score'], r['t'], r['TeamName']))

# Look at the 0.955 band too, and the 0.950 band (which is huge and recently flooded)
for s in (0.955, 0.953, 0.950, 0.944):
    b = [r for r in rows if r['Score'] == s]
    if len(b) < 3:
        continue
    print('\nscore %.3f  n=%d  ranks %d..%d  spearman(pos,time)=%.3f  mono=%s'
          % (s, len(b), b[0]['Rank'], b[-1]['Rank'],
             spearman(list(range(len(b))), [r['t'] for r in b]),
             all(b[i]['t'] <= b[i + 1]['t'] for i in range(len(b) - 1))))
