import json, re
p = r'D:\kaggle\rsna-knee\downloads\research\kernels_raw.json'
d = json.load(open(p, encoding='utf-8'))

# union of all, keep best order info
seen = {}
for key in ['scoreDescending', 'voteCount', 'dateRun']:
    for i, x in enumerate(d[key]):
        r = x['ref']
        if r not in seen:
            seen[r] = {'rank_' + key: i + 1, 'ref': r, 'title': x['title'], 'author': x['author'],
                       'votes': x['totalVotes'], 'lastRun': x['lastRunTime'],
                       'gpu': x['enableGpu'], 'ds': [y['ref'] if isinstance(y, dict) else y for y in x.get('datasetDataSources', [])],
                       'ks': [y['ref'] if isinstance(y, dict) else y for y in x.get('kernelDataSources', [])],
                       }
        else:
            seen[r]['rank_' + key] = i + 1

rows = list(seen.values())

# extract explicit score claims from title
sc = re.compile(r'(0[._-]?9\d{2})')
for r in rows:
    m = sc.findall(r['title'])
    r['title_score'] = None
    if m:
        v = m[0].replace('_', '.').replace('-', '.')
        if v.startswith('09'):
            v = '0.' + v[2:]
        try:
            f = float(v)
            if 0.5 < f < 1.0:
                r['title_score'] = f
        except Exception:
            pass

rows.sort(key=lambda r: (r['rank_scoreDescending'] if 'rank_scoreDescending' in r else 9999))
json.dump(rows, open(r'D:\kaggle\rsna-knee\downloads\research\kernels_ranked.json', 'w', encoding='utf-8'), indent=1)

print('%-4s %-52s %-22s %-6s %-11s %s' % ('#', 'ref', 'author', 'votes', 'lastRun', 'title'))
for r in rows[:60]:
    print('%-4s %-52s %-22s %-6s %-11s %s' % (
        r.get('rank_scoreDescending', '-'), r['ref'][:52], r['author'][:22], r['votes'],
        r['lastRun'][:10], r['title'][:60]))
