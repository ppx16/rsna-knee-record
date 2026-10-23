import json
import re
import requests

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.discussions.types.discussions_api_service import ApiListTopicsRequest

api = KaggleApi()
api.authenticate()
hc = api.build_kaggle_client()._http_client
hc._init_session()
url = hc._get_request_url('discussions.DiscussionApiService', 'ListTopics')
print('URL:', url)


def post(req):
    hr = requests.Request(method='POST', url=url, json=req.__class__.to_dict(req),
                          headers=hc._session.headers, auth=hc._session.auth)
    return hc._session.send(hr.prepare(), timeout=90)


FID = 12044884
seen = {}
for sort in ('DATE_CREATED', 'HOTNESS', 'VOTES', 'RECENT_ACTIVITY'):
    req = ApiListTopicsRequest()
    req.forum_slug = 'rsna-knee-abnormality-detection'
    req.page_size = 50
    try:
        req.sort_by = sort
    except Exception as exc:  # noqa: BLE001
        print('sort %s rejected: %s' % (sort, str(exc)[:90]))
        continue
    r = post(req)
    if r.status_code != 200:
        print('sort %s -> status %s %s' % (sort, r.status_code, r.text[:160]))
        continue
    d = r.json()
    tops = d.get('topics') or []
    print('sort %-16s -> %d topics' % (sort, len(tops)))
    for t in tops:
        seen[t.get('id')] = t

print('\ntotal unique topics:', len(seen))
if seen:
    k = next(iter(seen.values()))
    print('topic keys:', sorted(k.keys()))
    items = sorted(seen.values(), key=lambda t: -(t.get('totalVotes') or 0))
    print('\ntop 25 by votes:')
    for t in items[:25]:
        print('  id=%-9s v=%-4s c=%-4s %s' % (t.get('id'), t.get('totalVotes'),
                                              t.get('totalComments'),
                                              (t.get('name') or '')[:105]))
    pat = re.compile(r'oai|osteoarthritis|external|checkpoint|rule|allowed|data', re.I)
    print('\nkeyword hits:')
    for t in items:
        if pat.search(t.get('name') or ''):
            print('  id=%-9s v=%-4s %s' % (t.get('id'), t.get('totalVotes'),
                                           (t.get('name') or '')[:105]))
