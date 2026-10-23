"""Read the final leaderboard and say exactly what changed between the public and private halves.

The question to answer is factual first: what is our private score, what is our private rank, and how does
that differ from the public numbers we recorded. Only then is "overfitting or underfitting" a question that
can be answered rather than narrated -- and the answer may be neither, because the private half is 70% of the
hidden set and re-ranks every tie.

Reads state/lb_private/ and state/lb_final/; writes nothing.
"""
import collections
import csv
import io
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(r'D:\kaggle\rsna-knee\state')


def load(folder: Path) -> tuple[Path, list[dict], list[str]]:
    for z in sorted(folder.glob('*.zip')):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(folder)
    csvs = sorted(folder.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise SystemExit('no CSV in %s' % folder)
    p = csvs[0]
    rows = list(csv.DictReader(io.open(p, encoding='utf-8-sig')))
    return p, rows, list(rows[0].keys()) if rows else []


print('=' * 92)
for label, folder in (('FINAL (post-deadline)', ROOT / 'lb_private'), ('PUBLIC snapshot (pre-deadline)', ROOT / 'lb_final')):
    try:
        p, rows, cols = load(folder)
    except SystemExit as exc:
        print('%-28s %s' % (label, exc))
        continue
    print('\n%s' % label)
    print('  file   : %s' % p.name)
    print('  rows   : %d' % len(rows))
    print('  columns: %s' % ', '.join(cols))

    me = next((r for r in rows if 'ppxl16' in ' '.join(str(v) for v in r.values())), None)
    if me:
        print('  us     : ' + '  '.join('%s=%s' % (k, me[k]) for k in cols))
    # score column, whichever it is called
    score_col = next((c for c in cols if c.lower() in ('score', 'privatescore', 'publicscore')), None)
    if score_col:
        by = collections.Counter()
        for r in rows:
            v = (r.get(score_col) or '').strip()
            if v:
                try:
                    by[float(v)] += 1
                except ValueError:
                    pass
        ours = float(me[score_col]) if me and (me.get(score_col) or '').strip() else None
        above = sum(c for s, c in by.items() if ours is not None and s > ours)
        print('  score hist near us:')
        for s in sorted(by, reverse=True):
            if ours is not None and s < ours - 0.003:
                break
            mark = '  <- us' if ours is not None and abs(s - ours) < 1e-9 else ''
            print('     %.3f  %5d teams%s' % (s, by[s], mark))
        if ours is not None:
            print('  teams strictly above us: %d' % above)
