import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
try:
    r = api.competition_submit_code("submission.csv",
        "0943cfm - 0943cf + MCL added to _RAD_EXCLUDE; +0.000904 macro on n=3802/210pos, 35/35 sign-stable, direction pre-registered by the 58-study audit",
        competition="rsna-knee-abnormality-detection",
        kernel="ppxl16/rsna-knee-0943cfm", kernel_version=1, quiet=True)
    print("SUBMITTED:", r)
except Exception as e:
    print("EXC %s: %s" % (type(e).__name__, str(e)[:400]))
    r = getattr(e, "response", None)
    if r is not None:
        print("--- body ---"); print(r.text[:1500])
