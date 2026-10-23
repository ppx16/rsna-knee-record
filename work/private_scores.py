"""Read our own submissions' private scores -- the column that was empty before the deadline.

The leaderboard download is public-only and the service endpoints need a web session, but the submissions
table has carried a privateScore column all along. If the competition has closed, it should be populated, and
that is the number the analysis needs: not the rank (which is a lottery inside a 1638-team tie) but the score
the private half actually gave us.
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
if start is None:
    raise SystemExit('cannot read the submissions table')
rows = list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
print('columns:', ', '.join(rows[0].keys()))
print('rows   : %d\n' % len(rows))

cols = rows[0].keys()
has_private = any('private' in c.lower() for c in cols)
print('a private column exists: %s' % has_private)
if has_private:
    pcol = next(c for c in cols if 'private' in c.lower())
    filled = [x for x in rows if (x.get(pcol) or '').strip()]
    print('rows with a private score: %d of %d\n' % (len(filled), len(rows)))
    print('%-9s %-19s %-10s %-10s %s' % ('ref', 'date', 'public', 'private', 'delta'))
    for x in rows:
        pub = (x.get('publicScore') or '').strip()
        priv = (x.get(pcol) or '').strip()
        if not pub and not priv:
            continue
        delta = ''
        if pub and priv:
            try:
                delta = '%+.4f' % (float(priv) - float(pub))
            except ValueError:
                pass
        print('%-9s %-19s %-10s %-10s %s' % (x['ref'], x['date'][:19], pub or '-', priv or '-', delta))

# the two we selected
sel = ['57250245', '57478661']
print('\nselected submissions:')
for ref in sel:
    x = next((y for y in rows if y['ref'] == ref), None)
    if x:
        print('  %s  public=%s  private=%s' % (ref, x.get('publicScore') or '-',
                                               (x.get(pcol) if has_private else '') or '-'))
