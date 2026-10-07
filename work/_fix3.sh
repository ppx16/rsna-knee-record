#!/bin/bash
# Apply the binarisation fix and restart training with a first checkpoint soon enough to read.
set -u
sed -i 's/\r$//' /root/fix3.sh 2>/dev/null
cd /root
pkill -f arm3d_cuda.py 2>/dev/null || true
sleep 2
/root/miniconda3/bin/python /root/_patch_auc.py
echo "=== restart training (1200 steps, eval every 200) ==="
nohup /root/miniconda3/bin/python /root/arm3d_cuda.py --steps 1200 --bs 8 --depth 32 --hw 128 --width 32 \
      --eval-every 200 > /root/arm3d.log 2>&1 &
echo "pid $!"
sleep 240
echo "=== after 4 min ==="
tail -16 /root/arm3d.log
