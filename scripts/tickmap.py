import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
subs = {s.ref: s for s in api.competition_submissions("rsna-knee-abnormality-detection")}
want = ["56599932","56599159","56568524","56570144","56570151","56568520","56791368"]
for r in want:
    s = subs.get(r)
    if not s:
        print("  %-10s *** NOT FOUND IN THE SUBMISSION LIST ***" % r); continue
    print("  %-10s pub=%-6s %-9s %s" % (r, s.public_score or "-",
          str(s.status).split(".")[-1][:9], (s.description or "")[:88]))
