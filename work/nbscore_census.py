import json
import re
import time

import requests

COMPETITION = 'rsna-knee-abnormality-detection'

_s = requests.Session()
_s.get('https://www.kaggle.com/competitions', timeout=60)
_h = {'x-xsrf-token': _s.cookies.get('XSRF-TOKEN', '')}
print('xsrf token present:', bool(_h['x-xsrf-token']))


def kaggle_api(service, method, body, tries=4):
    last = None
    for a in range(tries):
        r = _s.post('https://www.kaggle.com/api/i/%s/%s' % (service, method),
                    json=body, headers=_h, timeout=120)
        if r.status_code == 200:
            return r.json()
        last = (r.status_code, r.text[:300])
        time.sleep(2 + 3 * a)
    raise RuntimeError('failed %s %s -> %s' % (service, method, last))


comp = kaggle_api('competitions.CompetitionService', 'GetCompetition',
                  {'competitionName': COMPETITION})
comp = comp.get('competition', comp)
print('comp id', comp['id'], 'teams', comp['totalTeams'], 'deadline', comp['deadline'])


def list_notebooks(sort_by):
    out, page = [], 1
    while page < 40:
        body = {'kernelFilterCriteria': {'search': '', 'listRequest': {
                    'competitionId': comp['id'], 'sortBy': sort_by, 'pageSize': 100,
                    'group': 'EVERYONE', 'page': page}},
                'detailFilterCriteria': {'deletedAccessBehavior': 'RETURN_NOTHING',
                                         'unauthorizedAccessBehavior': 'RETURN_NOTHING',
                                         'excludeResultsFilesOutputs': True},
                'readMask': 'pinnedKernels'}
        got = kaggle_api('kernels.KernelsService', 'ListKernels', body).get('kernels', [])
        if not got:
            break
        out += got
        page += 1
    return out


raw = {}
for sort_by in ('SCORE_DESCENDING', 'DATE_RUN', 'VOTE_COUNT'):
    got = list_notebooks(sort_by)
    print(sort_by, '->', len(got))
    for k in got:
        raw[k['scriptUrl']] = k

recs = []
for k in raw.values():
    recs.append({
        'notebook': k['scriptUrl'].removeprefix('/code/'),
        'title': k.get('title', ''),
        'score': k.get('bestPublicScore'),
        'votes': k.get('totalVotes') or 0,
        'updated': str(k.get('scriptVersionDateCreated', ''))[:10],
    })

json.dump(recs, open('scratch/notebook_scores.json', 'w'), default=str)
scored = [r for r in recs if r['score'] is not None]
print('total public notebooks:', len(recs), ' with a score:', len(scored))

scored.sort(key=lambda r: (-float(r['score']), -r['votes']))
print()
print('%-58s %-8s %6s %s' % ('notebook', 'score', 'votes', 'title'))
for r in scored[:45]:
    print('%-58s %-8s %6s %s' % (r['notebook'][:58], r['score'], r['votes'], r['title'][:52]))
