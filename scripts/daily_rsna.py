"""One command for the daily RSNA check. Zero GPU, zero submissions, ~1 minute.

What it answers, in order:
  1. Where are we, and how thick are the medal margins?   (live board -> the authoritative CSV)
  2. Is the bar moving?                                    (band inflation vs our rank)
  3. Has anyone published a scored notebook BEATING our best?  (the only live upside left)
  4. Is the tick still legal?                              (delegates to verify_final_pair.py)

Design notes learned the hard way:
  * the medal lines and our rank come from the CSV's own Rank column, located by TeamName --
    never from a derived tool (FACTS C140/A187/A190: one script printed two contradictory ranks);
  * the download is retried and bounded, because the CLI loses responses to SSL EOF on this
    machine (A189) and because a hung download previously burned a 10-minute timeout;
  * freshness comes from the timestamp INSIDE the zip's CSV name, not from file mtime (A190).
"""
from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
COMP = 'rsna-knee-abnormality-detection'
TEAM = 'p pxl16'
LIVE = ROOT / 'work' / 'lb_live'
TREND = ROOT / 'state' / 'daily_trend.jsonl'


def fetch_board() -> Path:
    api = KaggleApi()
    last = None
    for attempt in range(6):
        try:
            api.authenticate()
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(8 * (attempt + 1))
    else:
        raise SystemExit('TRANSPORT: could not authenticate (%r)' % (last,))
    LIVE.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            api.competition_leaderboard_download(COMP, path=str(LIVE), quiet=True)
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(10 * (attempt + 1))
    else:
        raise SystemExit('TRANSPORT: could not download the board (%r)' % (last,))
    zp = LIVE / 'rsna-knee-abnormality-detection.zip'
    if not zp.is_file():
        raise SystemExit('download produced no zip')
    return zp


def snapshot_time(zp: Path) -> str:
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.endswith('.csv')]
    m = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', names[0]) if names else None
    return m.group(1) if m else '?'


def board_rows(zp: Path):
    with zipfile.ZipFile(zp) as z:
        raw = z.read(z.namelist()[0]).decode('utf-8-sig', errors='replace')
    rows = []
    for r in csv.DictReader(io.StringIO(raw)):
        try:
            rows.append({'rank': int(r['Rank']), 'score': float(r['Score']),
                         'team': (r.get('TeamName') or '').strip(),
                         'members': (r.get('TeamMemberUserNames') or '').strip()})
        except (ValueError, KeyError):
            continue
    rows.sort(key=lambda d: d['rank'])
    return rows


def bands(rows):
    out = []
    cur_score, first, last = None, None, None
    for r in rows:
        if r['score'] != cur_score:
            if cur_score is not None:
                out.append((cur_score, first, last, last - first + 1))
            cur_score, first = r['score'], r['rank']
        last = r['rank']
    if cur_score is not None:
        out.append((cur_score, first, last, last - first + 1))
    return out


def public_ceiling(ours_best: float, budget_s: float = 240.0):
    """Highest scored public notebook, via the internal API (zero GPU, zero submissions)."""
    import requests
    s = requests.Session()
    s.get('https://www.kaggle.com/competitions', timeout=60)
    hdr = {'x-xsrf-token': s.cookies.get('XSRF-TOKEN', '')}
    t0 = time.time()
    seen, page = {}, 1
    while page < 40 and time.time() - t0 < budget_s:
        body = {'kernelFilterCriteria': {'search': '', 'listRequest': {
                    'competitionId': 154281, 'sortBy': 'SCORE_DESCENDING', 'pageSize': 100,
                    'group': 'EVERYONE', 'page': page}},
                'detailFilterCriteria': {'deletedAccessBehavior': 'RETURN_NOTHING',
                                         'unauthorizedAccessBehavior': 'RETURN_NOTHING',
                                         'excludeResultsFilesOutputs': True},
                'readMask': 'pinnedKernels'}
        try:
            r = s.post('https://www.kaggle.com/api/i/kernels.KernelsService/ListKernels',
                       json=body, headers=hdr, timeout=90)
            if r.status_code != 200:
                break
            got = r.json().get('kernels', [])
        except Exception:  # noqa: BLE001
            break
        if not got:
            break
        for k in got:
            seen[k.get('scriptUrl')] = k
        page += 1
    scored = [{'notebook': u.removeprefix('/code/'), 'title': k.get('title', ''),
               'score': float(k['bestPublicScore']), 'votes': k.get('totalVotes') or 0}
              for u, k in seen.items() if k.get('bestPublicScore') is not None]
    scored.sort(key=lambda d: (-d['score'], -d['votes']))
    better = [d for d in scored if d['score'] > ours_best + 1e-9]
    return len(seen), len(scored), better[:10]


def main() -> int:
    zp = fetch_board()
    rows = board_rows(zp)
    n = len(rows)
    me = next((r for r in rows if r['team'] == TEAM or 'ppxl16' in r['members'].replace(' ', '')), None)
    if me is None:
        print('!! we are not on the board under %r' % TEAM)
        return 1
    bronze, silver, gold = int(n * 0.10), int(n * 0.05), 10
    b = bands(rows)
    ours = next(((s, lo, hi, c) for s, lo, hi, c in b if lo <= me['rank'] <= hi), None)

    print('=' * 78)
    print('RSNA daily check   board read %s   N=%d' % (snapshot_time(zp), n))
    print('  us      : rank %d  score %.3f' % (me['rank'], me['score']))
    print('  bronze  : cut %d -> margin %+d %s' % (bronze, bronze - me['rank'],
                                                   'INSIDE' if me['rank'] <= bronze else 'OUTSIDE'))
    print('  silver  : cut %d -> margin %+d %s' % (silver, silver - me['rank'],
                                                   'INSIDE' if me['rank'] <= silver else 'OUTSIDE'))
    print('  gold    : cut %d -> margin %+d' % (gold, gold - me['rank']))
    if ours:
        s, lo, hi, c = ours
        print('  our band: %.3f  ranks %d..%d (n=%d) -> position %d/%d; the bronze cut is at %s,'
              ' the silver cut at %s'
              % (s, lo, hi, c, me['rank'] - lo + 1, c,
                 'inside' if lo <= bronze <= hi else 'outside',
                 'inside' if lo <= silver <= hi else 'outside'))
    print('  bands   : ' + ' | '.join('%.3f %d..%d(n=%d)' % t for t in b[:6]))

    # trend from our own persisted snapshots (no network)
    hist = []
    for p in sorted((ROOT / 'state').glob('lb_snapshot_*.json'), key=lambda q: q.stat().st_mtime_ns):
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
            hist.append(d)
        except Exception:  # noqa: BLE001
            continue
    if len(hist) >= 2:
        a, z = hist[0], hist[-1]
        def parse(x):
            return datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        hours = (parse(z['at']) - parse(a['at'])).total_seconds() / 3600.0
        if hours > 0:
            print('  trend   : over %.0f h  rank %+0.1f/h  margin %+0.1f/h  N %+0.1f/h'
                  % (hours, (z['our_rank'] - a['our_rank']) / hours,
                     (z['margin'] - a['margin']) / hours,
                     (z['n_teams'] - a['n_teams']) / hours))

    print('-' * 78)
    print('  scanning the public surface for anything ABOVE our %.3f ...' % me['score'])
    try:
        total, scored_n, better = public_ceiling(me['score'])
        print('  %d public notebooks, %d with a real score' % (total, scored_n))
        if better:
            print('  *** ACTION: %d notebook(s) score ABOVE ours ***' % len(better))
            for d in better:
                print('      %-52s %.3f  v=%d' % (d['notebook'][:52], d['score'], d['votes']))
        else:
            print('  none above ours -> we hold the reproducible public ceiling')
    except Exception as exc:  # noqa: BLE001
        print('  census failed (%r) -- not a finding, a transport/tooling problem' % (exc,))

    print('-' * 78)
    v = subprocess.run([sys.executable, str(ROOT / 'work' / 'verify_final_pair.py')],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    verdict = [l for l in (v.stdout or '').splitlines() if 'VERDICT' in l]
    tick = [l.strip() for l in (v.stdout or '').splitlines() if 'TICK-' in l]
    for l in tick:
        print('  ' + l)
    print('  ' + (verdict[0] if verdict else 'pair verifier produced no verdict (rc=%d)' % v.returncode))

    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    rec = {'at': now, 'board_at': snapshot_time(zp), 'n_teams': n, 'our_rank': me['rank'],
           'our_score': me['score'], 'margin_bronze': bronze - me['rank'],
           'margin_silver': silver - me['rank'],
           'our_band': None if not ours else {'score': ours[0], 'lo': ours[1], 'hi': ours[2], 'n': ours[3]}}
    TREND.parent.mkdir(parents=True, exist_ok=True)
    with TREND.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec) + '\n')
    print('  appended to state/%s' % TREND.name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
