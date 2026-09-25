import json, sys, time
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
cv = api.config_values; auth = (cv.get('username'), cv.get('key'))
import requests
s = requests.Session(); s.trust_env = True

SEARCHES = ['rsna-knee', 'knee', 'rsna-knee-weights', 'radimagenet', 'dinov2 knee',
            'knee mri', 'knee-mri', 'raptor-knee', 'knee abnormality', 'rsna knee']
out = {}
for q in SEARCHES:
    rows = []
    for page in range(1, 4):
        try:
            r = s.get('https://www.kaggle.com/api/v1/datasets/list',
                      params={'search': q, 'page': page, 'pageSize': 100, 'sortBy': 'hottest'},
                      auth=auth, timeout=120)
            r.raise_for_status()
            j = r.json()
        except Exception as e:
            print('ERR', q, page, e, file=sys.stderr); break
        if not j:
            break
        rows.extend(j)
        if len(j) < 100:
            break
        time.sleep(1)
    out[q] = rows
    print('search=%r got=%d' % (q, len(rows)), file=sys.stderr)
    time.sleep(1)

with open(r'D:\kaggle\rsna-knee\downloads\research\datasets_raw.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print('done')
