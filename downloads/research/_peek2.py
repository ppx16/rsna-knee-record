import json
p = r'D:\kaggle\rsna-knee\downloads\research\kernels_raw.json'
d = json.load(open(p, encoding='utf-8'))
for key in ['scoreDescending', 'voteCount', 'dateRun']:
    refs = [x['ref'] for x in d[key]]
    print('=== %s : %d unique of %d' % (key, len(set(refs)), len(refs)))
    for r in refs[:12]:
        print('   ', r)
print()
a = set(x['ref'] for x in d['scoreDescending'])
b = set(x['ref'] for x in d['voteCount'])
c = set(x['ref'] for x in d['dateRun'])
print('score==vote order?', [x['ref'] for x in d['scoreDescending']][:5] == [x['ref'] for x in d['voteCount']][:5])
print('overlap score/vote', len(a & b), 'score/dateRun', len(a & c), 'union', len(a | b | c))
