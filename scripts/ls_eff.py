import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
for slug in ["tbongc/rsna-knee-effb0mil-sl16-tv7w", "ngoanphamvan/sl16s0b-cache-variant"]:
    try:
        fs = api.dataset_list_files(slug).files
        print("== %s  (%d files)" % (slug, len(fs)))
        for f in fs[:30]: print("     %s" % f.name)
    except Exception as e:
        print("== %s !! %s" % (slug, str(e)[:140]))
