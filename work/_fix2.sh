#!/bin/bash
# Fix the one stale `vols.shape` reference and start training. One heredoc, one interpreter.
set -u
sed -i 's/\r$//' /root/fix2.sh 2>/dev/null
cd /root
/root/miniconda3/bin/python /root/_patch_vols.py
echo "=== start training ==="
nohup /root/miniconda3/bin/python /root/arm3d_cuda.py --steps 2000 --bs 8 --depth 32 --hw 128 --width 32 \
      --eval-every 250 > /root/arm3d.log 2>&1 &
echo "pid $!"
sleep 110
echo "=== first 110s ==="
tail -22 /root/arm3d.log
