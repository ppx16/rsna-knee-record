import json, os, re, sys, time
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
cv = api.config_values; auth = (cv.get('username'), cv.get('key'))
import requests
s = requests.Session(); s.trust_env = True

BASE = r'D:\kaggle\rsna-knee\downloads\research'
rows = json.load(open(os.path.join(BASE, 'kernels_ranked.json'), encoding='utf-8'))
# take top 60 by scoreDescending rank, plus top 40 by voteCount
cand = [r['ref'] for r in rows if r.get('rank_scoreDescending', 9999) <= 60]
vote = sorted(rows, key=lambda r: -r['votes'])[:40]
for r in vote:
    if r['ref'] not in cand:
        cand.append(r['ref'])
print('candidates', len(cand), file=sys.stderr)

def walk(o, path=''):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(walk(v, path + '.' + k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(walk(v, path + '[%d]' % i))
    else:
        out[path] = o
    return out

meta = {}
for ref in cand:
    owner, slug = ref.split('/', 1)
    try:
        r = s.get('https://www.kaggle.com/api/v1/kernels/pull/%s/%s' % (owner, slug), auth=auth, timeout=90)
        if r.status_code != 200:
            meta[ref] = {'_err': r.status_code}; continue
        j = r.json()
        m = j.get('metadata', {}) or {}
        m.pop('text', None)
        flat = walk(m)
        # keep only shallow interesting keys
        keep = {k: v for k, v in flat.items() if k.count('.') <= 2 and k.count('[') <= 1}
        meta[ref] = keep
    except Exception as e:
        meta[ref] = {'_err': str(e)}
    time.sleep(0.35)

json.dump(meta, open(os.path.join(BASE, 'kernel_meta_full.json'), 'w', encoding='utf-8'), indent=1)

# show sample keys
sample = meta.get('mattiaangeli/bend-the-knee-to-the-dinosaurs', {})
print('SAMPLE KEYS:')
for k in sorted(sample):
    print('  ', k, '=', str(sample[k])[:110])
