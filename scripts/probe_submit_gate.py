"""Zero-cost probe: is the 'non-versioned datasets' refusal about the DATASET or about our fork?

We call CreateCodeSubmission for a kernel that is NOT ours but attaches the same dataset
(le4ves/rsna-knee-0-954-dual-blend-raptor-convnext-s6, credited 0.954).  The server checks
permissions and preconditions; whichever error comes back first tells us which gate tripped.
Nothing is created either way, because we are not the owner.
"""
import json
import requests

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest

api = KaggleApi()
api.authenticate()
hc = api.build_kaggle_client()._http_client
hc._init_session()
url = hc._get_request_url('competitions.CompetitionApiService', 'CreateCodeSubmission')

PROBES = [
    ('le4ves', 'rsna-knee-0-954-dual-blend-raptor-convnext-s6', 'same dataset, NOT our kernel'),
    ('heliosli', 'blend-mine-in-private-lb-luck-to-all-upvoters', 'upstream of our fork, same dataset'),
    ('goodpjw2008', 'rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954', 'attaches rsna-knee-compact-v1'),
    ('ppxl16', 'rsna-knee-blendgold-restored', 'OURS, attaches rsna-knee-compact-v1, submitted fine before'),
]

for owner, slug, note in PROBES:
    req = ApiCreateCodeSubmissionRequest()
    req.competition_name = 'rsna-knee-abnormality-detection'
    req.kernel_owner = owner
    req.kernel_slug = slug
    req.kernel_version = 1
    req.file_name = 'submission.csv'
    req.submission_description = 'probe'
    hr = requests.Request(method='POST', url=url, json=req.__class__.to_dict(req),
                          headers=hc._session.headers, auth=hc._session.auth)
    r = hc._session.send(hr.prepare(), timeout=120)
    try:
        msg = r.json().get('error', {}).get('message', r.text[:200])
    except Exception:  # noqa: BLE001
        msg = r.text[:200]
    print('%-10s %-52s %-3s %s' % (owner, slug[:52], r.status_code, msg[:150]))
    print('           (%s)' % note)
