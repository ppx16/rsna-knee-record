import json, sys, time
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
cv = api.config_values; auth = (cv.get('username'), cv.get('key'))
import requests
s = requests.Session(); s.trust_env = True

COMP = 'rsna-knee-abnormality-detection'
out = {}

def fetch(sortBy, page, pageSize=100):
    r = s.get('https://www.kaggle.com/api/v1/kernels/list',
              params={'competition': COMP, 'sortBy': sortBy, 'page': page, 'pageSize': pageSize},
              auth=auth, timeout=120)
    r.raise_for_status()
    return r.json()

for sortBy in ['scoreDescending', 'voteCount', 'dateRun']:
    all_k = []
    for page in range(1, 6):
        try:
            j = fetch(sortBy, page)
        except Exception as e:
            print('ERR', sortBy, page, e, file=sys.stderr); break
        if not j:
            break
        all_k.extend(j)
        print('sortBy=%s page=%d got=%d' % (sortBy, page, len(j)), file=sys.stderr)
        if len(j) < 100:
            break
        time.sleep(1)
    out[sortBy] = all_k

# competition-level score listing (public LB per kernel)
try:
    r = s.get('https://www.kaggle.com/api/v1/competitions/%s/leaderboard/view' % COMP, auth=auth, timeout=120)
    out['_lb_status'] = r.status_code
except Exception as e:
    out['_lb_status'] = str(e)

with open(r'D:\kaggle\rsna-knee\downloads\research\kernels_raw.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print('counts', {k: len(v) for k, v in out.items() if isinstance(v, list)})
