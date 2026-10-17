import json
import requests

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelsRequest, KernelsListSortType)

api = KaggleApi()
api.authenticate()
hc = api.build_kaggle_client()._http_client
hc._init_session()
url = hc._get_request_url('kernels.KernelsApiService', 'ListKernels')


def post(req):
    hr = requests.Request(method='POST', url=url,
                          json=req.__class__.to_dict(req),
                          headers=hc._session.headers, auth=hc._session.auth)
    return hc._session.send(hr.prepare(), timeout=90)


rows = []
token = None
for page in range(20):
    req = ApiListKernelsRequest()
    req.competition = 'rsna-knee-abnormality-detection'
    req.sort_by = KernelsListSortType.SCORE_DESCENDING
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
    print('page', page, 'got', len(ks), 'total', len(rows), 'token', bool(token))
    if not token or not ks:
        break

json.dump(rows, open('scratch/kernel_scores.json', 'w'), default=str)
print('saved', len(rows))
if rows:
    print('keys:', sorted(rows[0].keys()))
