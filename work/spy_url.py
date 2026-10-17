import json, os
import requests

_orig = requests.post
_calls = []


def spy(url, *a, **kw):
    _calls.append((url, kw.get('json')))
    return _orig(url, *a, **kw)


requests.post = spy

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelsRequest, KernelsListSortType)

api = KaggleApi()
api.authenticate()
ks = api.build_kaggle_client().kernels.kernels_api_client
r = ApiListKernelsRequest()
r.competition = 'rsna-knee-abnormality-detection'
r.sort_by = KernelsListSortType.SCORE_DESCENDING
r.page_size = 25
ks.list_kernels(r)
requests.post = _orig
for u, b in _calls:
    print('URL :', u)
    print('BODY:', json.dumps(b))
