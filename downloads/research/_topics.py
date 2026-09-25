import json, os, sys, time
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
cv = api.config_values; auth = (cv.get('username'), cv.get('key'))
import requests
s = requests.Session(); s.trust_env = True
BASE = r'D:\kaggle\rsna-knee\downloads\research'
COMP = 'rsna-knee-abnormality-detection'

def gp(url, **p):
    for _ in range(3):
        try:
            r = s.get(url, params=p, auth=auth, timeout=90)
            if r.status_code == 200:
                try: return r.json()
                except Exception: return {'_text': r.text[:3000]}
            return {'_status': r.status_code}
        except Exception as e:
            last = str(e); time.sleep(2)
    return {'_err': last}

all_topics = []
seen = set()
for sort in ['recent', 'votes', 'comments']:
    for page in range(1, 12):
        j = gp('https://www.kaggle.com/api/v1/competitions/%s/topics' % COMP,
               pageSize=100, page=page, sortBy=sort)
        if not isinstance(j, dict) or 'topics' not in j:
            print('stop', sort, page, str(j)[:120], file=sys.stderr); break
        ts = j['topics']
        if not ts: break
        new = 0
        for t in ts:
            key = t.get('id') or t.get('topicId')
            if key in seen: continue
            seen.add(key); t['_sort'] = sort; all_topics.append(t); new += 1
        print('sort=%s page=%d got=%d new=%d total=%d/%s' % (sort, page, len(ts), new, len(all_topics), j.get('totalCount')), file=sys.stderr)
        if new == 0 or len(ts) < 100: break
        time.sleep(0.6)

json.dump(all_topics, open(os.path.join(BASE, 'forum_topics.json'), 'w', encoding='utf-8'), indent=1)
if all_topics:
    print('KEYS:', sorted(all_topics[0].keys()))
    print('total unique topics:', len(all_topics))
