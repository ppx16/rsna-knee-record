import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
for slug in ["apoorvadixit/rsna-knee-weights-v4cx",
             "soyaoki/rsna-knee-my-models",
             "lixin73/rsna-knee-train-checkpoints",
             "dreaddevelopment/raptor-knee-maxspan"]:
    try:
        fs = api.dataset_list_files(slug).files
        print("== %s  (%d files)" % (slug, len(fs)))
        for f in fs[:25]:
            print("     %-58s %12d" % (f.name, f.totalBytes))
    except Exception as e:
        print("== %s  !! %s: %s" % (slug, type(e).__name__, str(e)[:160]))
