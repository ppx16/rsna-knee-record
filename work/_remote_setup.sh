#!/bin/bash
# Bring the rented 4090 up to the point where the 3D arm can train on it.
#
# The corpus is pulled with PRE-SIGNED URLs rather than kaggle.json, because FACTS A44 forbids putting account
# credentials on a third-party machine. The URLs were resolved locally, where the credentials are.
set -u
echo "=== uploaded files ==="
ls -la /root/urls_corpus.tsv /root/urls_corpus_ext.tsv /root/kaggle_parallel.py 2>&1

echo "=== python / torch ==="
/root/miniconda3/bin/python - <<'PY'
import sys
print("python", sys.version.split()[0])
for m in ("numpy", "torch", "sklearn", "pandas"):
    try:
        mod = __import__(m)
        print("  %-8s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("  %-8s MISSING (%s)" % (m, type(e).__name__))
try:
    import torch
    print("  cuda available:", torch.cuda.is_available())
except Exception:
    pass
PY

echo "=== disk ==="
df -h /root/autodl-tmp | tail -1

echo "=== start the corpus pull in the background ==="
cd /root
nohup /root/miniconda3/bin/python /root/kaggle_parallel.py --url /root/urls_corpus.tsv \
      --dst /kaggle/input/datasets/dreaddevelopment/knee-raptor-corpus --streams 12 \
      > /root/pull_corpus.log 2>&1 &
echo "corpus pid $!"
nohup /root/miniconda3/bin/python /root/kaggle_parallel.py --url /root/urls_corpus_ext.tsv \
      --dst /kaggle/input/datasets/dreaddevelopment/knee-raptor-corpus-ext --streams 12 \
      > /root/pull_ext.log 2>&1 &
echo "ext pid $!"
sleep 20
echo "=== after 20s ==="
ls -la /kaggle/input/datasets/dreaddevelopment/knee-raptor-corpus/ 2>&1 | tail -6
tail -3 /root/pull_corpus.log 2>&1
tail -3 /root/pull_ext.log 2>&1
