#!/bin/bash
# The second axis: CAPACITY. Same data, same soft labels, same binarisation -- bigger model.
#
# The schedule run (same model, 33x steps) is still climbing at 0.6634 -> 0.6927, which says the width-32
# 128px Conv3d is short of training. This asks the orthogonal question: is it also short of CAPACITY? Running
# them in parallel on one 4090 is acceptable because only the AUC is compared, not the wall clock, and the box
# bills by the hour whether it is busy or idle -- so idle GPU is the only genuinely wasted resource here.
set -u
sed -i 's/\r$//' /root/run_big.sh 2>/dev/null
cd /root
echo "=== capacity run: width 64, 160px, bs 4, 20000 steps ==="
nohup /root/miniconda3/bin/python /root/arm3d_cuda.py --steps 20000 --bs 4 --depth 32 --hw 160 --width 64 \
      --eval-every 2000 > /root/arm3d_big.log 2>&1 &
echo "pid $!"
sleep 180
echo "=== after 180s ==="
grep -E 'HELD-OUT|FINAL|Error|error' /root/arm3d_big.log 2>/dev/null | tail -6
grep -E '^\[3darm\] step' /root/arm3d_big.log 2>/dev/null | tail -3
echo "=== schedule run, for comparison ==="
grep -E 'HELD-OUT|FINAL' /root/arm3d_long.log 2>/dev/null | tail -4
grep -E '^\[3darm\] step' /root/arm3d_long.log 2>/dev/null | tail -1
