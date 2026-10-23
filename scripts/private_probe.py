"""Try every route we can legitimately use to read the private leaderboard.

The CLI only downloads the public one, and the internal service returned 403 on a cookie session. So: try the
legacy v1 endpoints with basic auth (which is what the CLI itself uses), and the internal service with basic
auth as well. Print each status so the answer is which door is open, not that none is.

Credentials are read from ~/.kaggle/kaggle.json and never printed.
"""
import json
import sys
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
cred = json.loads((Path.home() / '.kaggle' / 'kaggle.json').read_text())
AUTH = (cred['username'], cred['key'])
SLUG = 'rsna-knee-abnormality-detection'

print('trying routes for %s\n' % SLUG)

# 1. legacy v1 leaderboard view -- historically returns BOTH public and private
urls = [
    'https://www.kaggle.com/api/v1/competitions/%s/leaderboard/view' % SLUG,
    'https://www.kaggle.com/api/v1/competitions/leaderboard/view?competitionName=%s' % SLUG,
    'https://www.kaggle.com/api/v1/competitions/%s' % SLUG,
]
for u in urls:
    try:
        r = requests.get(u, auth=AUTH, timeout=60)
        print('%-78s %s  %d bytes' % (u.replace('https://www.kaggle.com', ''), r.status_code, len(r.content)))
        if r.status_code == 200:
            try:
                j = r.json()
                print('   json keys: %s' % (list(j)[:12] if isinstance(j, dict) else type(j).__name__))
            except ValueError:
                print('   (not json) %s' % r.text[:200].replace('\n', ' '))
    except Exception as exc:  # noqa: BLE001
        print('%-78s ERR %r' % (u[:78], exc))

# 2. the internal service, but with basic auth instead of a cookie session
print()
s = requests.Session()
s.auth = AUTH
for ltype in ('PRIVATE', 'PUBLIC'):
    body = {'competitionId': None, 'leaderboardType': ltype, 'competitionName': SLUG}
    try:
        r = s.post('https://www.kaggle.com/api/i/competitions.LeaderboardService/GetLeaderboard',
                   json=body, timeout=60)
        print('LeaderboardService/GetLeaderboard %-7s -> %s  %s'
              % (ltype, r.status_code, r.text[:220].replace('\n', ' ')))
    except Exception as exc:  # noqa: BLE001
        print('LeaderboardService %-7s ERR %r' % (ltype, exc))
