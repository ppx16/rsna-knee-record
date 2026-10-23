import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
for slug in ["ppxl16/rsna-knee-msft2-cx", "ppxl16/rsna-knee-msft2"]:
    try:
        s = api.kernels_status(slug)
        print("%-32s %s" % (slug, s))
    except Exception as e:
        print("%-32s !! %s" % (slug, str(e)[:160]))
