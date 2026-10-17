"""Submit a COMPLETED notebook version to the RSNA code competition, with the real error text.

Why not `kaggle competitions submit`: on this client that path collapses every server-side
refusal into a bare "400 Client Error" with no body, which cost us a round trip on 2026-10-13
(the actual reason was "requires an output FileName" / "uses non-versioned datasets").
`CreateCodeSubmission` returns the server's own message, so we always know which gate tripped.

    python work/submit_code.py <owner/slug> <version> "<message>"
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest

ROOT = Path(__file__).resolve().parents[1]
COMP = 'rsna-knee-abnormality-detection'
LEDGER = ROOT / 'state' / 'rsna_submissions.jsonl'


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    ref_slug, version, message = argv[1], int(argv[2]), argv[3]
    owner, slug = ref_slug.split('/')

    api = KaggleApi()
    api.authenticate()
    hc = api.build_kaggle_client()._http_client
    hc._init_session()
    url = hc._get_request_url('competitions.CompetitionApiService', 'CreateCodeSubmission')

    req = ApiCreateCodeSubmissionRequest()
    req.competition_name = COMP
    req.kernel_owner = owner
    req.kernel_slug = slug
    req.kernel_version = version
    req.file_name = 'submission.csv'
    req.submission_description = message

    hr = requests.Request(method='POST', url=url, json=req.__class__.to_dict(req),
                          headers=hc._session.headers, auth=hc._session.auth)
    r = hc._session.send(hr.prepare(), timeout=300)
    body = r.text
    print('status %s' % r.status_code)
    print(body[:600])

    if r.status_code != 200:
        print('\nREFUSED by the server -- this is the real reason, not a transport failure')
        return 1

    ref = re.search(r'"ref"\s*:\s*"?(\d{6,})', body)
    ref = ref.group(1) if ref else None
    row = {
        'date': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'ref': ref,
        'file': 'submission.csv',
        'desc': message,
        'status': 'PENDING',
        'score': None,
        'private': None,
        'kernel': '%s:%d' % (ref_slug, version),
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    print('\nSUBMITTED ref=%s' % ref)
    print('LOGGED to %s' % LEDGER.name)
    print('(score appears in ~10-11 h; work/reconcile_submissions.py refreshes it)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
