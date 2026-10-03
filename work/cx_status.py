import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
for slug in ["ppxl16/rsna-knee-msft2-cx", "ppxl16/rsna-knee-msft2"]:
    try:
        print("%-32s %s" % (slug, api.kernels_status(slug)))
    except Exception as e:
        print("%-32s !! %s" % (slug, str(e)[:200]))
print()
try:
    fs = api.kernels_output_files("ppxl16/rsna-knee-msft2-cx") if hasattr(api,"kernels_output_files") else None
except Exception:
    fs = None
try:
    st = api.kernels_list_files("ppxl16/rsna-knee-msft2-cx")
    for f in st.files:
        print("   %-46s %12d" % (f.name, f.totalBytes))
except Exception as e:
    print("list_files !! %s" % str(e)[:200])
