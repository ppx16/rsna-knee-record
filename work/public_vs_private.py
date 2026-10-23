"""Does the public half predict the private half?  Fifty submissions, both scores.

This is the question the whole campaign rested on and could not answer until now. Every artifact we built was
judged by its public score, on the assumption that public ordering carried information about private ordering.
Now every submission carries both, so that assumption can be measured directly across fifty submissions that
span the range of things we tried -- from 0.943-era stacks to the 0.958 ensemble.

Reads the submissions table; writes nothing. Reports the correlation, the systematic offset, and whether the
ordering among our own variants survives.
"""
import csv
import io
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = subprocess.run(['kaggle', 'competitions', 'submissions',
                    '-c', 'rsna-knee-abnormality-detection', '--csv'],
                   capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
lines = ((r.stdout or '') + (r.stderr or '')).splitlines()
start = next((i for i, l in enumerate(lines) if l.lstrip('\ufeff').startswith('ref,')), None)
rows = list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))

pairs = []
for x in rows:
    pub = (x.get('publicScore') or '').strip()
    priv = (x.get('privateScore') or '').strip()
    if pub and priv:
        try:
            pairs.append((float(pub), float(priv), x['ref']))
        except ValueError:
            pass

print('submissions with both scores: %d\n' % len(pairs))
pairs.sort(key=lambda t: -t[0])

deltas = [priv - pub for pub, priv, _ in pairs]
import statistics as st
print('public - private gap: mean %+.4f   sd %.4f   min %+.4f   max %+.4f'
      % (st.mean([-d for d in deltas]), st.stdev([-d for d in deltas]),
         min(-d for d in deltas), max(-d for d in deltas)))

pub = [a for a, _, _ in pairs]
priv = [b for _, b, _ in pairs]


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    for pos, i in enumerate(order):
        rk[i] = pos
    return rk


def spear(a, b):
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float('nan')


def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float('nan')


print('\ncorrelation between the halves, across our own submissions:')
print('  pearson  r = %.3f' % pearson(pub, priv))
print('  spearman r = %.3f' % spear(pub, priv))

# how much of the private spread is explained by the public one
print('\nprivate spread among our submissions: %.4f (%.3f .. %.3f)'
      % (max(priv) - min(priv), min(priv), max(priv)))
print('public  spread among our submissions: %.4f (%.3f .. %.3f)'
      % (max(pub) - min(pub), min(pub), max(pub)))

print('\nthe submissions that displayed 0.958, by private score:')
for a, b, ref in pairs:
    if abs(a - 0.958) < 1e-9:
        print('   %s  public %.3f  private %.3f' % (ref, a, b))
