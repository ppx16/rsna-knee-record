#!/bin/bash
# Run the pretrained-backbone arm. First launch also DOWNLOADS the Kinetics-400 weights (~120 MB), so the
# first minute is network, not compute -- do not read a slow start as a failure.
set -u
sed -i 's/\r$//' /root/run_pre.sh 2>/dev/null
cd /root
echo "=== internet check (weights download) ==="
curl -s -o /dev/null -w 'download.pytorch.org http=%{http_code} %{time_total}s\n' --max-time 25 https://download.pytorch.org/ || true
echo "=== launch: r3d_18 pretrained, 6000 steps, bs 4, 128px ==="
nohup /root/miniconda3/bin/python /root/arm3d_pre.py --steps 6000 --bs 4 --depth 32 --hw 128 --eval-every 1000 \
      > /root/arm3d_pre.log 2>&1 &
echo "pid $!"
sleep 210
echo "=== after 210s ==="
head -6 /root/arm3d_pre.log
grep -E 'HELD-OUT|FINAL|Error|error' /root/arm3d_pre.log 2>/dev/null | tail -6
grep -E '^\[3darm\] step' /root/arm3d_pre.log 2>/dev/null | tail -3
