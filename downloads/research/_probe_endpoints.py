import json, sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
cv = api.config_values; auth = (cv.get('username'), cv.get('key'))
import requests
s = requests.Session(); s.trust_env = True

# probe several endpoints for a single known kernel to find one that exposes public LB score
refs = ['mattiaangeli/bend-the-knee-to-the-dinosaurs']
for owner, slug in [tuple(r.split('/')) for r in refs]:
    for url in [
        'https://www.kaggle.com/api/v1/kernels/pull/%s/%s' % (owner, slug),
        'https://www.kaggle.com/api/v1/kernels/output/%s/%s' % (owner, slug),
        'https://www.kaggle.com/api/v1/kernels/view/%s/%s' % (owner, slug),
        'https://www.kaggle.com/api/v1/kernels/status/%s/%s' % (owner, slug),
    ]:
        try:
            r = s.get(url, auth=auth, timeout=60)
            body = r.text[:600]
            print('---', url, r.status_code)
            print(body.replace('\n', ' ')[:600])
        except Exception as e:
            print('---', url, 'ERR', e)
