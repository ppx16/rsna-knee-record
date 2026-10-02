import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
for slug in ["apoorvadixit/rsna-knee-weights-v4cx", "soyaoki/rsna-knee-my-models",
             "lixin73/rsna-knee-train-checkpoints", "dreaddevelopment/raptor-knee-maxspan"]:
    try:
        fs = api.dataset_list_files(slug).files
        print("== %s" % slug)
        for f in fs:
            print("     %s" % f.name)
    except Exception as e:
        print("== %s !! %s" % (slug, str(e)[:160]))
