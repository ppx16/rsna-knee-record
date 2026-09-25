import json, os, sys, time
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
cv = api.config_values; auth = (cv.get('username'), cv.get('key'))
import requests
s = requests.Session(); s.trust_env = True

BASE = r'D:\kaggle\rsna-knee\downloads\research'
COMP = 'rsna-knee-abnormality-detection'

def get(url, **params):
    for attempt in range(3):
        try:
            r = s.get(url, params=params, auth=auth, timeout=90)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return {'_text': r.text[:4000]}
            return {'_status': r.status_code}
        except Exception as e:
            last = str(e); time.sleep(2)
    return {'_err': last}

out = {}
# competition overview / rules / data
out['competition_view'] = get('https://www.kaggle.com/api/v1/competitions/view/%s' % COMP)

# forum topics -- several plausible endpoints
for name, url, params in [
    ('forum_list', 'https://www.kaggle.com/api/v1/competitions/%s/forum' % COMP, {'pageSize': 100, 'sortBy': 'recent'}),
    ('forum_topics', 'https://www.kaggle.com/api/v1/forum/topics', {'competitionId': COMP}),
    ('discussion_list', 'https://www.kaggle.com/api/v1/discussions/list', {'competition': COMP}),
    ('topics_list', 'https://www.kaggle.com/api/v1/competitions/topics/%s' % COMP, {}),
]:
    out[name] = get(url, **params)

json.dump(out, open(os.path.join(BASE, 'discussions_raw.json'), 'w', encoding='utf-8'), indent=1)

for k, v in out.items():
    if isinstance(v, dict) and '_status' in v:
        print('%-18s HTTP %s' % (k, v['_status']))
    elif isinstance(v, dict) and '_err' in v:
        print('%-18s ERR %s' % (k, v['_err']))
    elif isinstance(v, list):
        print('%-18s list len=%d  keys=%s' % (k, len(v), sorted(v[0].keys())[:14] if v else []))
    elif isinstance(v, dict):
        print('%-18s dict keys=%s' % (k, sorted(v.keys())[:20]))
