import json
import requests

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest

api = KaggleApi()
api.authenticate()
hc = api.build_kaggle_client()._http_client
hc._init_session()
url = hc._get_request_url('competitions.CompetitionApiService', 'CreateCodeSubmission')
print('URL:', url)

KERNEL = 'ppxl16/rsna-knee-raptor-s6-fork'
MSG = ('Byte-for-byte fork of the public notebook credited 0.956 (heliosli/blend-mine-in, '
       'sha256 63e92116). Same two legs as our 0.954 (nartaa Raptor CoAtNet-384 + heliosli '
       'compact ConvNeXt-small S6) but at the author 0.5/0.5 rank blend and with the newer '
       'compact_train.py source. All inputs public, no empty slug.')

req = ApiCreateCodeSubmissionRequest()
req.competition_name = 'rsna-knee-abnormality-detection'
req.kernel_owner = KERNEL.split('/')[0]
req.kernel_slug = KERNEL.split('/')[1]
req.kernel_version = 1
req.file_name = 'submission.csv'
req.submission_description = MSG
print('body:', json.dumps(req.__class__.to_dict(req))[:500])

hr = requests.Request(method='POST', url=url, json=req.__class__.to_dict(req),
                      headers=hc._session.headers, auth=hc._session.auth)
r = hc._session.send(hr.prepare(), timeout=180)
print('status', r.status_code)
print(r.text[:900])
