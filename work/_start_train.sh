#!/bin/bash
# Start the 3D arm on the rented 4090. Corpus is already complete (all_vols.npy = 15.9 GB).
#
# NOTE: this file is written on Windows and MUST have its CRLF stripped before bash runs it. The previous
# launcher died on exactly that -- 'syntax error near unexpected token $do\r' -- so it never started anything,
# while the corpus finished downloading behind it. The sed below is therefore load-bearing.
set -u
sed -i 's/\r$//' /root/start_train.sh 2>/dev/null
cd /root
PY=/root/miniconda3/bin/python

echo "=== corpus check ==="
ls -la /kaggle/input/datasets/dreaddevelopment/knee-raptor-corpus/all_vols.npy \
       /kaggle/input/datasets/dreaddevelopment/knee-raptor-corpus-ext/extra_vols.npy 2>&1
$PY - <<'PY'
import numpy as np, glob
for pat in ('all_vols.npy', 'extra_vols.npy'):
    h = sorted(glob.glob('/kaggle/input/**/' + pat, recursive=True))
    if h:
        a = np.load(h[0], mmap_mode='r')
        print("  %-16s %s %s" % (pat, a.shape, a.dtype))
    else:
        print("  %-16s MISSING" % pat)
PY

echo "=== start training ==="
nohup $PY /root/arm3d_cuda.py --steps 2000 --bs 8 --depth 32 --hw 128 --width 32 --eval-every 250 \
      > /root/arm3d.log 2>&1 &
echo "pid $!"
sleep 100
echo "=== first 100s of the training log ==="
tail -20 /root/arm3d.log
