import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
subs = api.competition_submissions("rsna-knee-abnormality-detection")
print("total:", len(subs))
for s in subs[:24]:
    print("%-10s pub=%-6s priv=%-6s %-9s %s" % (s.ref, s.public_score or "-", s.private_score or "-",
          str(s.status).split(".")[-1][:9], (s.description or "")[:58]))
