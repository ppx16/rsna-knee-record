"""One command to run at the deadline: verify everything, then print exactly what to click.

The only step no script can do is ticking two submissions in the browser, so this makes that step as
foolproof as possible: it re-verifies the pair against the live server, checks that both ledgers are
committed and pushed, reads the current board, and prints a compact block with the two refs and the words
that must appear in their descriptions (so they can be matched by text rather than by score -- the board
shows 20+ submissions at 0.942/0.943 that must not be picked).

Usage: python work/deadline_report.py
Exit code 0 = safe to tick; 1 = something is wrong, do not tick.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEADLINE = datetime(2026, 10, 22, 23, 59, tzinfo=timezone.utc)


def sh(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    r = subprocess.run(cmd, cwd=str(cwd or ROOT), capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=1800)
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def main() -> int:
    now = datetime.now(timezone.utc)
    left = DEADLINE - now
    local = DEADLINE.astimezone()
    ok = True

    print('=' * 78)
    print('RSNA DEADLINE REPORT   %s' % now.strftime('%Y-%m-%dT%H:%M:%SZ'))
    print('deadline   : %s  (= local %s)' % (DEADLINE.strftime('%Y-%m-%d %H:%MZ'),
                                             local.strftime('%Y-%m-%d %H:%M')))
    print('time left  : %s' % (str(left).split('.')[0] if left.total_seconds() > 0 else 'PASSED'))
    print('=' * 78)

    # 1. the pair, against the live server
    print('\n[1] pair verification (reads state/final_pair.json, checks the live server)')
    rc, out = sh([sys.executable, 'work/verify_final_pair.py'])
    for line in out.splitlines():
        if any(k in line for k in ('TICK-', 'VERDICT', 'PROBLEM', 'TRANSPORT', 'oai flags')):
            print('   ', line.strip())
    if 'VERDICT: pair present, COMPLETE, and legal' not in out:
        print('   *** FAILED -- do not tick until this is green ***')
        ok = False

    # 2. ledgers committed and pushed
    print('\n[2] ledgers committed and pushed')
    # Append-only logs are expected to be dirty: daily_rsna.py adds a trend row on every run, and this
    # report runs it.  Counting those as "uncommitted work" would make the verdict red every single time,
    # which would train the operator to ignore it -- so they are whitelisted, not fixed by suppressing them.
    APPEND_ONLY = ('state/daily_trend.jsonl', 'state/rsna_submissions.jsonl',
                   'state/kernel_watch.txt', 'state/closing_watch.txt')
    for repo in (ROOT, Path(r'D:\kaggle\arc-agi-3-ledger')):
        rc, out = sh(['git', 'status', '--porcelain'], cwd=repo)
        rc2, head = sh(['git', 'rev-parse', '--short', 'HEAD'], cwd=repo)
        rc3, remote = sh(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=repo)
        lines = [l for l in out.splitlines() if l.strip()]
        real = [l for l in lines if not any(f in l for f in APPEND_ONLY)]
        dirty = bool(real)
        only_logs = bool(lines) and not real
        ra = remote.split()[0][:7] if remote.strip() else 'UNREACHABLE'
        h = head.strip()[:7]
        flag = 'OK' if (not dirty and ra == h) else 'CHECK'
        print('    %-22s HEAD=%s remote=%s dirty=%s%s  -> %s'
              % (repo.name, h, ra, dirty, ' (append-only log only)' if only_logs else '', flag))
        if dirty or ra not in (h, 'UNREACHABLE'):
            ok = False

    # 3. the board, right now
    print('\n[3] board (this is a proxy only; medals are decided on the private split)')
    rc, out = sh([sys.executable, 'work/daily_rsna.py'])
    for line in out.splitlines():
        if any(k in line for k in ('us      :', 'bronze  :', 'silver  :', 'our band:', 'none above')):
            print('   ', line.strip())

    # 4. what to click -- with the real description text pulled from the server, because the UI has to be
    # matched by words, not by score: 20+ of our submissions display 0.942/0.943 and the default selection
    # is not ours.
    d = json.loads((ROOT / 'state' / 'final_pair.json').read_text(encoding='utf-8'))
    descs = {}
    scores = {}
    try:
        import csv as _csv
        import io as _io
        rc, out = sh(['kaggle', 'competitions', 'submissions',
                      '-c', 'rsna-knee-abnormality-detection', '--csv'])
        lines = out.splitlines()
        start = next((i for i, l in enumerate(lines) if l.lstrip('\ufeff').startswith('ref,')), None)
        if start is not None:
            for row in _csv.DictReader(_io.StringIO('\n'.join(lines[start:]))):
                ref = str(row.get('ref'))
                descs[ref] = (row.get('description') or '').strip()
                scores[ref] = (row.get('publicScore') or '').strip()
    except Exception as exc:  # noqa: BLE001
        print('   (could not read descriptions: %r)' % (exc,))

    print('\n[4] CLICK THESE TWO  (Kaggle -> My Submissions -> select 2 -> Save/Submit)')
    for slot, ref in zip(('1', '2'), d['tick']):
        c = d['candidates'].get(ref, {})
        print('    slot %s : ref %-9s score %-6s oai=%s'
              % (slot, ref, scores.get(ref) or c.get('public', '?'), c.get('oai')))
        desc = descs.get(ref, '')
        if desc:
            print('             starts with: %s' % desc[:110])
            print('             ends with  : ...%s' % desc[-80:])
        else:
            print('             (no description fetched -- match by the ref number instead)')
    print('\n    then: leave the page, do not submit anything else.')
    print('    note: 5+ of our submissions display 0.958; tick 1 is the artifact whose recipe is')
    print('          entirely someone else\'s published work, i.e. the least self-fitted one.')

    print('\n' + ('RESULT: SAFE TO TICK' if ok else 'RESULT: NOT SAFE -- fix [1]/[2] first'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
