#!/bin/bash
# Chain a much longer PRETRAINED run behind the current one.
#
# The evidence for spending more time: the pretrained backbone climbs ~+0.014 AUC per 1000 steps and is still
# rising at 3000 (0.7541 -> 0.7638 -> 0.7776), while the from-scratch model plateaued AND rolled over
# (0.6927 -> 0.6631 -> 0.6493, best 0.7077). So the prior is doing real work and the only question left is
# where it saturates. On this box that question costs about CNY 2.3 for 40000 steps, which is the cheapest
# open question in the project.
set -u
sed -i 's/\r$//' /root/chain_pre.sh 2>/dev/null
cd /root
echo "[chain] waiting for the current pretrained run to finish ..."
while pgrep -f arm3d_pre.py > /dev/null; do sleep 20; done
echo "[chain] previous run ended; current best:"
grep -E 'HELD-OUT|FINAL' /root/arm3d_pre2.log | tail -4

echo "[chain] starting the long pretrained run: 40000 steps, eval every 2000"
nohup /root/miniconda3/bin/python /root/arm3d_pre.py --steps 40000 --bs 4 --depth 32 --hw 128 \
      --eval-every 2000 > /root/arm3d_pre_long.log 2>&1 &
echo "[chain] pid $!"
