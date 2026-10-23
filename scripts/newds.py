import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
targets = [
 "dariushafshar/kaggle-competition-leaderboard-intelligence",
 "arumpuri/rsna-knee-baseline-weights",
 "richyroseyiloveyou/rsna-knee-full-volume-288-npz",
 "sachinhambar/rsna-knee-ft-cache-data",
 "wenqiangao1/e13asdf",
 "codeandodev/rsna-knee-run-base",
 "codeandodev/rsna-knee-procesado-32x224x224",
 "vaprf5/rsna-knee-v11-preprocessed",
 "bijaychdas/rsna-knee-2026-dinov2-small-hf",
]
for t in targets:
    try:
        fs = api.dataset_list_files(t).files
        names = [f.name for f in fs]
        print("== %-58s %d files" % (t, len(names)))
        for n in names[:10]: print("      %s" % n)
        if len(names) > 10: print("      ... +%d more" % (len(names)-10))
    except Exception as e:
        print("== %-58s !! %s" % (t, str(e)[:80]))
