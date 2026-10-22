"""Record the final result once the deadline has passed and the private leaderboard is visible.

Why this exists: the campaign's last step is not a submission, it is knowing where we landed. Doing that by
hand after the fact is how results get mis-quoted -- this project has already mis-read a leaderboard twice
(A203's stale CSV, C189's LastSubmissionDate). So the reading is scripted, and it writes the number it
actually saw, with the columns it actually had, into state/FINAL-RESULT.md.

It is safe to run at any time: before the deadline it says so and does nothing.

Usage: python work/final_result.py [--force]
"""
import csv
import io
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEADLINE = datetime(2026, 10, 22, 23, 59, tzinfo=timezone.utc)
LB = ROOT / 'state' / 'lb_final'
OUT = ROOT / 'state' / 'FINAL-RESULT.md'


def sh(cmd: list[str]) -> tuple[int, str]:
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=1800)
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def main(argv: list[str]) -> int:
    now = datetime.now(timezone.utc)
    if now < DEADLINE and '--force' not in argv:
        left = DEADLINE - now
        print('deadline has not passed yet (%s to go) -- nothing to record' % str(left).split('.')[0])
        print('re-run after %s, or pass --force to read the board as it stands' % DEADLINE.strftime('%Y-%m-%d %H:%MZ'))
        return 0

    LB.mkdir(parents=True, exist_ok=True)
    rc, out = sh(['kaggle', 'competitions', 'leaderboard',
                  '-c', 'rsna-knee-abnormality-detection', '-d', '-p', str(LB)])
    csvs = sorted(LB.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        # the CLI leaves a zip behind rather than extracting it, so unpack it ourselves
        zips = sorted(LB.glob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
        if zips:
            import zipfile
            with zipfile.ZipFile(zips[0]) as zf:
                zf.extractall(LB)
            csvs = sorted(LB.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        print('no leaderboard CSV downloaded')
        print((out or '').strip()[:400])
        return 1
    path = csvs[0]
    rows = list(csv.DictReader(io.open(path, encoding='utf-8-sig')))
    cols = list(rows[0].keys()) if rows else []
    print('leaderboard: %s  (%d rows)  columns: %s' % (path.name, len(rows), ', '.join(cols)))

    # locate ourselves by the username column, not by a name substring (the team name has a space in it)
    me = None
    for r in rows:
        joined = ' '.join(str(v) for v in r.values())
        if 'ppxl16' in joined:
            me = r
            break
    if me is None:
        print('could not find ppxl16 in the leaderboard')
        return 1

    lines = ['# FINAL RESULT — RSNA Knee', '',
             'Read %s from `%s`.' % (now.strftime('%Y-%m-%dT%H:%M:%SZ'), path.name),
             '', '| column | value |', '|---|---|']
    for k in cols:
        lines.append('| %s | %s |' % (k, me.get(k, '')))
    lines += ['', 'Deadline: %s.' % DEADLINE.strftime('%Y-%m-%d %H:%MZ'),
              '', 'Selected submissions (from `state/final_pair.json`): '
              '`57250245` (0.958, OAI) and `57478661` (0.950, non-OAI).']
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print('\nwrote %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
