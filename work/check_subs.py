"""Check the pending submission only (one API call, no board download)."""
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
subs = api.competition_submissions("rsna-knee-abnormality-detection")
for s in sorted(subs, key=lambda s: str(getattr(s, "date", "")), reverse=True)[:5]:
    print(f"{getattr(s,'date','')}  ref={getattr(s,'ref','')}  "
          f"status={str(getattr(s,'status','')).split('.')[-1]:9s} "
          f"public={getattr(s,'publicScore',None)}  {str(getattr(s,'description',''))[:52]}")
