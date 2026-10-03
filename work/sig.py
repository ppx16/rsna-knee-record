import sys, inspect
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kaggle.api.kaggle_api_extended import KaggleApi
print("competition_submit_code%s" % (inspect.signature(KaggleApi.competition_submit_code),))
src = inspect.getsource(KaggleApi.competition_submit_code)
print(src[:1800])
