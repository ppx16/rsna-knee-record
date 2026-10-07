#!/bin/bash
# Wait for the corpus, then train the 3D arm on the rented 4090. Unattended.
set -u
CORP=/kaggle/input/datasets/dreaddevelopment/knee-raptor-corpus/all_vols.npy
EXT=/kaggle/input/datasets/dreaddevelopment/knee-raptor-corpus-ext/extra_vols.npy
PY=/root/miniconda3/bin/python

echo "[launch] waiting for the corpus ..."
for i in $(seq 1 240); do
  # a file whose size has stopped growing for two consecutive checks is complete
  s1=$(stat -c %s "$CORP" 2>/dev/null || echo 0)
  sleep 30
  s2=$(stat -c %s "$CORP" 2>/dev/null || echo 0)
  e2=$(stat -c %s "$EXT" 2>/dev/null || echo 0)
  if [ "$s1" = "$s2" ] && [ "$s1" -gt 15000000000 ]; then
     echo "[launch] corpus settled at $s2 bytes (ext $e2)"
     break
  fi
  [ $((i % 10)) -eq 0 ] && echo "[launch] $s2 / 14800000000 bytes"
done

echo "[launch] starting the 3D arm"
cd /root
nohup $PY /root/arm3d_cuda.py --steps 2000 --bs 8 --depth 32 --hw 128 --width 32 --eval-every 250 \
      > /root/arm3d.log 2>&1 &
echo "[launch] pid $!"
sleep 120
tail -20 /root/arm3d.log
