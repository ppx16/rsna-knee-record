"""List OUR OWN kernels with their real state, straight from the internal ListKernels API.

`kaggle kernels list --user ppxl16` shows a truncated, oddly ordered view (and hides private
notebook titles), and `kernels status <slug>` returns 404 both for "does not exist" and for
"exists but is not readable that way" -- so neither is a reliable existence check on its own.
This asks the same endpoint the census uses, filtered to our user.

Usage: python work/my_kernels.py [substring]
"""
from __future__ import annotations

import sys

import requests

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelsRequest, KernelsListSortType

needle = (sys.argv[1] if len(sys.argv) > 1 else '').lower()

api = KaggleApi()
api.authenticate()
hc = api.build_kaggle_client()._http_client
hc._init_session()
url = hc._get_request_url('kernels.KernelsApiService', 'ListKernels')


def post(req):
    hr = requests.Request(method='POST', url=url, json=req.__class__.to_dict(req),
                          headers=hc._session.headers, auth=hc._session.auth)
    return hc._session.send(hr.prepare(), timeout=120)


rows, token = [], None
for _ in range(20):
    req = ApiListKernelsRequest()
    req.user = 'ppxl16'
    req.sort_by = KernelsListSortType.DATE_RUN
    req.page_size = 100
    if token:
        req.page_token = token
    r = post(req)
    if r.status_code != 200:
        print('status', r.status_code, r.text[:300])
        break
    d = r.json()
    ks = d.get('kernels', d.get('kernelsList', []))
    rows += ks
    token = d.get('nextPageToken')
    if not token or not ks:
        break

print('kernels:', len(rows))
for k in sorted(rows, key=lambda k: str(k.get('lastRunTime') or ''), reverse=True):
    slug = str(k.get('ref') or k.get('scriptUrl') or '')
    if needle and needle not in slug.lower():
        continue
    print('%-52s run=%-22s score=%-8s votes=%s' % (slug.split('/')[-1][:52], str(k.get('lastRunTime'))[:22],
                                                   k.get('bestPublicScore'), k.get('totalVotes')))
