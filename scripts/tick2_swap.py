"""Swap tick 2 to a new non-OAI artifact, but only if the server says it is actually better.

Why a script instead of a manual edit: the deadline is tomorrow and the pair has to be right the first
time.  The rules this enforces:
  * the new ref must exist and be COMPLETE (a PENDING or ERROR ref must never reach the tick);
  * its public score must be strictly better than the current tick 2's (0.944) -- an equal score is not a
    reason to move, because the tie would be resolved by an unmeasurable unrounded value;
  * it must be flagged non-OAI, because tick 2 exists solely as insurance if the organisers rule the
    OAI-lineage Raptor models unusable;
  * the old tick 2 is demoted, not deleted;
  * running it twice is a no-op.

Usage: python work/tick2_swap.py <ref> [--dry-run]
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIR = ROOT / 'state' / 'final_pair.json'


def submissions() -> dict[str, dict]:
    r = subprocess.run(['kaggle', 'competitions', 'submissions',
                        '-c', 'rsna-knee-abnormality-detection', '--csv'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    import csv
    import io
    lines = ((r.stdout or '') + (r.stderr or '')).splitlines()
    start = next((i for i, l in enumerate(lines) if l.lstrip('\ufeff').startswith('ref,')), None)
    if start is None:
        raise SystemExit('REFUSING: could not read the submissions table')
    return {str(row['ref']): row for row in csv.DictReader(io.StringIO('\n'.join(lines[start:])))}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    ref = argv[1]
    dry = '--dry-run' in argv

    d = json.loads(PAIR.read_text(encoding='utf-8'))
    if d['tick'][1] == ref:
        raise SystemExit('REFUSING: %s is already tick 2' % ref)

    rows = submissions()
    if ref not in rows:
        raise SystemExit('REFUSING: %s is not in the submissions table' % ref)
    row = rows[ref]
    status = (row.get('status') or '').replace('SubmissionStatus.', '')
    score = (row.get('publicScore') or '').strip()
    if status != 'COMPLETE' or not score:
        raise SystemExit('REFUSING: %s is %s with score %r -- a settled score is required'
                         % (ref, status, score))
    new = float(score)
    old_ref = d['tick'][1]
    # Read the outgoing score from the server too. On the first real run the candidate dict had no
    # 'public' field, so the comparison ran against a made-up 0.0 and "passed" for the wrong reason --
    # the decision was right but the evidence in the log was empty. Refuse instead if it is unreadable.
    old_row = rows.get(old_ref) or {}
    old_score = (old_row.get('publicScore') or '').strip()
    if not old_score:
        raise SystemExit('REFUSING: cannot read the current tick 2 (%s) score from the server, so the '
                         'comparison would be vacuous' % old_ref)
    old = float(old_score)
    if new <= old:
        raise SystemExit('REFUSING: %s scores %.3f, not better than tick 2 %s at %.3f'
                         % (ref, new, old_ref, old))
    desc = (row.get('description') or '')
    lowered = desc.lower()
    if not any(k in lowered for k in ('non-oai', 'no oai', 'without oai')):
        raise SystemExit('REFUSING: %s is not described as non-OAI, and tick 2 exists only as the '
                         'OAI-rule insurance' % ref)

    print('new tick 2 : %s  %.3f  (%s)' % (ref, new, desc[:90]))
    print('old tick 2 : %s  %.3f  -> demoted to candidate' % (old_ref, old))
    if dry:
        print('DRY RUN: nothing written')
        return 0

    cand = d['candidates']
    if old_ref in cand:
        cand[old_ref]['label'] = 'NONOAI-SPARE'
        cand[old_ref]['why'] = (cand[old_ref].get('why', '') +
                                '  [%s] Demoted from tick 2: %s scored %.3f as the non-OAI insurance '
                                'and is strictly better.' % ('2026-10-21', ref, new))
    cand[ref] = {
        'label': 'TICK-2',
        'public': '%.3f' % new,
        'oai': False,
        'why': ('Non-OAI insurance: nartaa\'s S6 all-data ConvNeXt run on its own (Raptor half weighted '
                'to zero, compact half entirely S6). Documented as 0.950 alone and built with "No OAI data '
                'was used", which is why it can serve as the floor if the OAI-lineage Raptor models are '
                'ever ruled unusable. Measured here at %.3f.' % new),
        'clashes_with': [],
    }
    d['tick'] = [d['tick'][0], ref]
    d['_tick2_change_2026_10_21'] = {
        'old': old_ref,
        'new': ref,
        'scores': {old_ref: old, ref: new},
        'why': ('tick 2 exists only as insurance against an OAI ruling, so it should be the best non-OAI '
                'artifact we hold. It was 0.944 for three days while a 0.950-class non-OAI artifact '
                '(S6 alone) had never been submitted.'),
    }
    PAIR.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('\ntick is now:', ' + '.join(d['tick']))
    print('now run: python work/verify_final_pair.py')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
