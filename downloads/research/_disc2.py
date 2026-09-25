import json, os, sys, time
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
cv = api.config_values; auth = (cv.get('username'), cv.get('key'))
import requests
s = requests.Session(); s.trust_env = True
BASE = r'D:\kaggle\rsna-knee\downloads\research'
COMP = 'rsna-knee-abnormality-detection'

def get(url, **params):
    try:
        r = s.get(url, params=params, auth=auth, timeout=90)
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return {'_text': r.text[:2000]}
        return {'_status': r.status_code}
    except Exception as e:
        return {'_err': str(e)}

out = {}
cands = [
    ('api_discussions', 'https://www.kaggle.com/api/v1/discussions/list', {'competitionId': COMP, 'pageSize': 100}),
    ('api_discussions2', 'https://www.kaggle.com/api/v1/discussions/list', {'competition': COMP, 'pageSize': 100, 'sortBy': 'top'}),
    ('api_forum', 'https://www.kaggle.com/api/v1/competitions/forum/list', {'competitionId': COMP}),
    ('api_topics', 'https://www.kaggle.com/api/v1/topics/list', {'competitionId': COMP}),
    ('api_search', 'https://www.kaggle.com/api/v1/discussions/search', {'query': 'rsna knee', 'pageSize': 50}),
    ('api_comp_topics', 'https://www.kaggle.com/api/v1/competitions/%s/topics' % COMP, {'pageSize': 100}),
    ('api_kernels_meta', 'https://www.kaggle.com/api/v1/kernels/list', {'competition': COMP, 'sortBy': 'scoreAscending', 'pageSize': 20}),
]
for name, url, params in cands:
    out[name] = get(url, **params)
    v = out[name]
    st = v.get('_status') if isinstance(v, dict) else None
    print('%-18s' % name, 'status=%s' % st if st else ('list=%d' % len(v) if isinstance(v, list) else 'ok'))
json.dump(out, open(os.path.join(BASE, 'discussions_raw.json'), 'w', encoding='utf-8'), indent=1)

# also check datasets published per-user (weights hunt)
for user in ['dreaddevelopment', 'mattiaangeli', 'antoinegg1', 'pilkwang', 'prvsiyan', 'marwanmath', 'renta0426', 'nishantkharga', 'romantamrazov', 'sofiaanjenje']:
    j = get('https://www.kaggle.com/api/v1/datasets/list', user=user, pageSize=100, sortBy='updated')
    if isinstance(j, list):
        print('\n### user %s : %d datasets' % (user, len(j)))
        for d in j:
            print('   %-62s %9s KB  %s  votes=%s' % (d.get('ref'), d.get('totalBytes', 0) // 1024,
                  (d.get('lastUpdated') or '')[:10], d.get('voteCount')))
    else:
        print('\n### user %s : %s' % (user, j))
    time.sleep(0.3)
