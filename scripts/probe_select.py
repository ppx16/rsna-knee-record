import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
m = [x for x in dir(api) if "select" in x.lower() or "final" in x.lower()]
print("select/final-ish methods on KaggleApi:", m)
ss = [x for x in dir(api) if "submission" in x.lower()]
print("submission methods:", ss)
