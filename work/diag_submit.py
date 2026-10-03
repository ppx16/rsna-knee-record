import sys, traceback
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
print("has competition_submit_code:", hasattr(api, "competition_submit_code"))
print([m for m in dir(api) if "competition" in m and "submit" in m])
try:
    api.competition_submit_code("rsna-knee-abnormality-detection", "test",
        kernel="ppxl16/rsna-knee-0943cfm", kernel_version=1)
    print("OK")
except Exception as e:
    print("EXC %s" % type(e).__name__)
    print(str(e)[:1200])
    r = getattr(e, "response", None)
    if r is not None:
        print("--- body ---"); print(r.text[:1200])
