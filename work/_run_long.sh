#!/bin/bash
# The clean follow-up: SAME model and SAME data as the 1200-step run, but ~33x the schedule.
#
# Why this is the right next experiment and not a bigger model: the 1200-step run finished with loss barely
# moving (0.61 -> 0.57) and reached 0.6634, which looks like underfitting rather than '3D carries no signal'.
# Changing the model size AND the schedule at once would make either outcome uninterpretable, and on this box
# time is the cheap resource: 1200 steps took 68 s, so 40000 steps is roughly 40 minutes at CNY ~2.
set -u
sed -i 's/\r$//' /root/run_long.sh 2>/dev/null
cd /root
pkill -f arm3d_cuda.py 2>/dev/null || true
sleep 2
echo "=== long run: 40000 steps, identical model/data ==="
nohup /root/miniconda3/bin/python /root/arm3d_cuda.py --steps 40000 --bs 8 --depth 32 --hw 128 --width 32 \
      --eval-every 4000 > /root/arm3d_long.log 2>&1 &
echo "pid $!"
sleep 200
echo "=== after 200s ==="
tail -12 /root/arm3d_long.log
