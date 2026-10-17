import json

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelsRequest, KernelsListSortType)

api = KaggleApi()
api.authenticate()
ks = api.build_kaggle_client().kernels.kernels_api_client

rows = []
token = None
for page in range(12):
    r = ApiListKernelsRequest()
    r.competition = 'rsna-knee-abnormality-detection'
    r.sort_by = KernelsListSortType.SCORE_DESCENDING
    r.page_size = 100
    if token:
        r.page_token = token
    resp = ks.list_kernels(r)
    ks_ = resp.kernels or []
    if not ks_:
        break
    for k in ks_:
        d = k if isinstance(k, dict) else k.to_dict()
        rows.append(d)
    token = resp.next_page_token
    print('page', page, 'got', len(ks_), 'total', len(rows), 'token', bool(token))
    if not token:
        break

json.dump(rows, open('scratch/kernel_scores.json', 'w'), default=str)

seen = set()
print()
print('%-38s %-9s %6s %s' % ('ref', 'score', 'votes', 'title'))
for d in rows:
    ref = d.get('ref')
    if ref in seen:
        continue
    seen.add(ref)
    sc = d.get('bestPublicScore') if d.get('bestPublicScore') is not None else d.get('best_public_score')
    if sc is None:
        continue
    if float(sc) < 0.945:
        continue
    print('%-38s %-9s %6s %s' % (ref, sc, d.get('totalVotes'), str(d.get('title'))[:60]))
