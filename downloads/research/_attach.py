import json, os
BASE = r'D:\kaggle\rsna-knee\downloads\research'
meta = json.load(open(os.path.join(BASE, 'kernel_meta_full.json'), encoding='utf-8'))
rows = json.load(open(os.path.join(BASE, 'kernels_ranked.json'), encoding='utf-8'))
rank = {r['ref']: r.get('rank_scoreDescending', 9999) for r in rows}
votes = {r['ref']: r['votes'] for r in rows}

def arr(v, key):
    out = []
    i = 0
    while True:
        k = '%s[%d]' % (key, i)
        if k in v:
            out.append(v[k]); i += 1
        else:
            break
    return out

recs = []
for ref, m in meta.items():
    if '_err' in m:
        recs.append({'ref': ref, 'err': m['_err']}); continue
    recs.append({
        'ref': ref,
        'rank': rank.get(ref, 9999),
        'votes': m.get('.totalVotes', votes.get(ref)),
        'title': m.get('.title'),
        'lastRun': (m.get('.lastRunTime') or '')[:10],
        'gpu': m.get('.enableGpu'),
        'machine': m.get('.machineShape'),
        'internet': m.get('.enableInternet'),
        'ver': m.get('.currentVersionNumber'),
        'datasets': arr(m, '.datasetDataSources'),
        'kernels': arr(m, '.kernelDataSources'),
        'models': arr(m, '.modelDataSources'),
    })
recs.sort(key=lambda r: r.get('rank', 9999))
json.dump(recs, open(os.path.join(BASE, 'kernel_attach.json'), 'w', encoding='utf-8'), indent=1)

print('== TOP 30 by scoreDescending rank ==')
for r in recs[:30]:
    if 'err' in r:
        print('%-3s %-46s ERR %s' % ('-', r['ref'], r['err'])); continue
    print('#%-3d %-46s v=%-4s %s gpu=%s %s' % (r['rank'], r['ref'], r['votes'], r['lastRun'], r['gpu'], r['machine']))
    print('        ds: %s' % (', '.join(r['datasets']) or '-'))
    print('        ks: %s' % (', '.join(r['kernels']) or '-'))
    print('        mdl: %s' % (', '.join(r['models']) or '-'))

print('\n\n== ALL KERNEL SOURCES USED BY TOP 60 (freq) ==')
from collections import Counter
c = Counter()
for r in recs[:60]:
    for k in r.get('kernels', []):
        c[k] += 1
for k, n in c.most_common(50):
    flag = '' if k in ('sofiaanjenje/rsna-knee-e11-train', 'sofiaanjenje/rsna-knee-e13-train') else '   <== NOT e11/e13'
    print('  %-3d %s%s' % (n, k, flag))

print('\n\n== ALL DATASETS USED BY TOP 60 (freq) ==')
c2 = Counter()
for r in recs[:60]:
    for k in r.get('datasets', []):
        c2[k] += 1
OURS = set('''dreaddevelopment/raptor-knee-maxspan dreaddevelopment/raptor-knee-native384
dreaddevelopment/raptor-knee-native384dense mattiaangeli/knee-mri-fold-weights
mattiaangeli/opencv-python-headless-4120088-x86 marwanmath/resnet-50-radimagenet-marwan
mattiaangeli/rsna-knee-coat-resgated-ep10-top3 antoinegg1/rsna-knee-e11-diverse-heads-v20
antoinegg1/rsna-knee-e9-radimagenet-heads-v15 pilkwang/rsna-knee-llm-labels
prvsiyan/rsna-knee-v52-radimagenet-heads-20260812 pilkwang/rsna-knee-weights'''.split())
for k, n in c2.most_common(80):
    flag = '' if k in OURS else '   <== NOT IN OUR LIST'
    print('  %-3d %s%s' % (n, k, flag))
