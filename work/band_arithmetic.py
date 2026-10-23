"""How far can a rank move without the score changing?  Arithmetic of a truncated tie band.

The question this answers: if our private rank fell from ~609 to 1000+, does that require our score to have
fallen, or is it something the 0.958 band does on its own? Everything here is arithmetic on the public band,
which is 1638 teams sharing one displayed score.

Reads the final public CSV; writes nothing.
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
F = Path(r'D:\kaggle\rsna-knee\state\lb_private')
for z in sorted(F.glob('*.zip')):
    with zipfile.ZipFile(z) as zf:
        zf.extractall(F)
p = sorted(F.glob('*.csv'))[-1]
rows = list(csv.DictReader(io.open(p, encoding='utf-8-sig')))
print('file: %s   teams: %d' % (p.name, len(rows)))

me = next(r for r in rows if 'ppxl16' in ' '.join(str(v) for v in r.values()))
our_rank = int(me['Rank'])
our_score = float(me['Score'])

by = {}
for r in rows:
    v = (r.get('Score') or '').strip()
    if v:
        by[float(v)] = by.get(float(v), 0) + 1

above = sum(c for s, c in by.items() if s > our_score)
band = by[our_score]
in_band = our_rank - above
print('our rank %d, score %.3f, teams above %d, our band holds %d, we are %dth inside it'
      % (our_rank, our_score, above, band, in_band))

# the band is one displayed step wide: [score - 0.0005, score + 0.0005)
width = 1.0 / 10 ** 3
print('\ndisplayed step : %.3f' % width)
print('teams per unit : %.0f per 1.000  ->  %.0f per 0.100  ->  %.1f per 1e-4'
      % (band / width, band / width / 10, band / width / 10000))

print('\nhow much unrounded score each move costs, inside this band:')
for places in (100, 200, 400, 600, 1000):
    d = width * places / band
    print('  %5d places  ->  %.6f of AUC' % (places, d))

print('\n=> a fall of 400 places inside this band is %.5f of AUC, i.e. %.0f%% of one displayed step.'
      % (width * 400 / band, 100 * (width * 400 / band) / width))

print('\nwhat the public half tells us about our own unrounded position:')
frac = (in_band - 1) / band
print('  public in-band percentile: %.1f%%  ->  unrounded ~ %.5f (if the band starts at %.4f)'
      % (100 * frac, our_score - width / 2 + frac * width, our_score - width / 2))
print('  the public half is 30%% of 1322 studies ~ %d studies' % round(0.30 * 1322))
print('  the private half is 70%% ~ %d studies' % round(0.70 * 1322))
print('  going from %d to %d studies shrinks the standard error by about %.0f%%,'
      % (round(0.30 * 1322), round(0.70 * 1322), 100 * (1 - (0.30 / 0.70) ** 0.5)))
print('  so our private unrounded value is a *less noisy* estimate of the same skill --')
print('  it can easily sit a few 1e-4 away from the public one without meaning anything changed.')
