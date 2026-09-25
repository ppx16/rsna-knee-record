import json, io, sys
d = json.load(open(r'D:\kaggle\rsna-knee\downloads\research\kernels_raw.json', encoding='utf-8'))
k = d['scoreDescending']
print('total', len(k))
print('sample keys:', sorted(k[0].keys()))
print(json.dumps(k[0], indent=1)[:2500])
