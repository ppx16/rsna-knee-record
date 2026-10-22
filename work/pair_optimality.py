"""Is {0.958 OAI, 0.950 non-OAI} actually the best pair available?  Enumerate the alternatives.

The question this answers: we hold FIVE artifacts that display 0.958 and one at 0.950, and we chose one of
each rather than two 0.958s. That looks like giving away a thousandth. It is not, and the reason is worth
writing down before the deadline rather than after:

  * if the OAI-lineage models are allowed -- the overwhelming case -- the platform keeps the better of the
    two selected submissions, so the score is 0.958 either way;
  * if they are ruled unusable, an all-OAI pair scores NOTHING, while the mixed pair scores 0.950.

So the mixed pair costs nothing in the likely world and saves the whole result in the unlikely one. The only
choice that is strictly worse is any pair whose second member is not the best non-OAI artifact we have.

Reads state/final_pair.json and the live submissions table; writes nothing.
"""
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
PAIR = ROOT / 'state' / 'final_pair.json'


def table() -> dict:
    r = subprocess.run(['kaggle', 'competitions', 'submissions',
                        '-c', 'rsna-knee-abnormality-detection', '--csv'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    lines = ((r.stdout or '') + (r.stderr or '')).splitlines()
    start = next((i for i, l in enumerate(lines) if l.lstrip('\ufeff').startswith('ref,')), None)
    if start is None:
        raise SystemExit('cannot read the submissions table')
    return {row['ref']: row for row in csv.DictReader(io.StringIO('\n'.join(lines[start:])))}


def main() -> int:
    d = json.loads(PAIR.read_text(encoding='utf-8'))
    tick = d['tick']
    cand = d['candidates']
    rows = table()

    # our artifacts, with the lineage flag the ledger records for them
    arts = {}
    for ref, meta in cand.items():
        if ref not in rows:
            continue
        score = (rows[ref].get('publicScore') or '').strip()
        if not score:
            continue
        arts[ref] = {'score': float(score), 'oai': bool(meta.get('oai')), 'label': meta.get('label', '')}

    print('artifacts with a settled score: %d' % len(arts))
    by_score = sorted(arts.items(), key=lambda kv: -kv[1]['score'])
    for ref, a in by_score[:10]:
        print('  %-9s %.3f  oai=%-5s %s' % (ref, a['score'], a['oai'], a['label']))

    best_oai = max((a['score'] for a in arts.values() if a['oai']), default=0)
    best_non = max((a['score'] for a in arts.values() if not a['oai']), default=0)
    print('\nbest OAI artifact      : %.3f' % best_oai)
    print('best non-OAI artifact  : %.3f' % best_non)

    print('\n%-34s %-22s %s' % ('pair', 'if OAI allowed', 'if OAI ruled unusable'))
    options = [
        ('two best OAI (%.3f + %.3f)' % (best_oai, best_oai), best_oai, 0.0),
        ('our pair (%.3f + %.3f)' % (best_oai, best_non), best_oai, best_non),
        ('two best overall, both OAI', best_oai, 0.0),
    ]
    for name, allow, deny in options:
        print('%-34s %-22s %s' % (name, '%.3f' % allow, '(nothing)' if deny == 0 else '%.3f' % deny))

    chosen = (arts.get(tick[0], {}), arts.get(tick[1], {}))
    print('\nselected now: %s (%.3f, oai=%s) + %s (%.3f, oai=%s)'
          % (tick[0], chosen[0].get('score', 0), chosen[0].get('oai'),
             tick[1], chosen[1].get('score', 0), chosen[1].get('oai')))

    ok = True
    if abs(chosen[0].get('score', 0) - best_oai) > 1e-9:
        print('  !! slot 1 is not the best OAI artifact')
        ok = False
    if chosen[1].get('oai'):
        print('  !! slot 2 is OAI, so the pair has no insurance')
        ok = False
    elif abs(chosen[1].get('score', 0) - best_non) > 1e-9:
        print('  !! slot 2 is not the best NON-OAI artifact (%.3f available)' % best_non)
        ok = False
    print('\nVERDICT: %s' % ('the pair is optimal under both lineages' if ok
                             else 'the pair is NOT optimal -- see the flags above'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
