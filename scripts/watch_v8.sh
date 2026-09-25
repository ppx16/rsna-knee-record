#!/usr/bin/env bash
# 监控 v8 run 到终态 -> 下载 output -> 抽取诊断文件(crash/hb/probe)并打印摘要。
# 用法: bash scripts/watch_v8.sh
set -u
export PYTHONUTF8=1

KERNEL="ppxl16/rsna-knee-tpu-train-v1"
ROOT="/d/kaggle/rsna-knee"
OUT="$ROOT/outputs/tpu_weights/seed2026_v19"
mkdir -p "$OUT"

echo "== watch start $(date +%H:%M:%S) =="
prev=""
while true; do
  s=$(kaggle kernels status "$KERNEL" 2>/dev/null | grep -oE 'KernelWorkerStatus\.[A-Z]+' | head -1)
  [ "$s" != "$prev" ] && { echo "[$(date +%H:%M:%S)] status=$s"; prev="$s"; }
  case "$s" in
    KernelWorkerStatus.COMPLETE|KernelWorkerStatus.ERROR|KernelWorkerStatus.CANCELLED) break ;;
  esac
  sleep 120
done

echo "== terminal $s, waiting 30s for output to finalize =="
sleep 30
for try in 1 2 3; do
  (cd "$ROOT" && kaggle kernels output "$KERNEL" -p "$OUT" 2>/dev/null) && break
  echo "  download retry $try/3"; sleep 60
done

echo "== output files =="
ls -la "$OUT" | head -60

echo "== diagnostic: tpu_probe =="
[ -f "$OUT/tpu_probe.txt" ] && cat "$OUT/tpu_probe.txt" || echo "(no tpu_probe.txt)"

echo "== diagnostic: crash files =="
shopt -s nullglob
crash=( "$OUT"/worker_*_crash.txt )
if [ ${#crash[@]} -gt 0 ]; then
  for f in "${crash[@]}"; do echo "--- $(basename "$f") ---"; cat "$f"; done
else
  echo "(no worker_*_crash.txt)"
fi

echo "== diagnostic: heartbeat files =="
hb=( "$OUT"/worker_*_hb.txt )
if [ ${#hb[@]} -gt 0 ]; then
  for f in "${hb[@]}"; do echo "--- $(basename "$f") ---"; cat "$f"; done
else
  echo "(no worker_*_hb.txt)"
fi

echo "== diagnostic: predict heartbeat (one file per PID, append-only) =="
phb=( "$OUT"/predict_hb_*.txt )
if [ ${#phb[@]} -gt 0 ]; then
  for f in "${phb[@]}"; do
    echo "--- $(basename "$f") ($(wc -l < "$f") lines) ---"
    echo "  first: $(head -1 "$f")"
    echo "  last : $(tail -1 "$f")"
    tail -n 12 "$f" | sed 's/^/    /'
  done
else
  echo "(no predict_hb_*.txt)"
fi

echo "== diagnostic: parent memory (fork baseline) =="
[ -f "$OUT/parent_mem.txt" ] && cat "$OUT/parent_mem.txt" || echo "(no parent_mem.txt)"

echo "== diagnostic: worker memory curves (RSS vs MemAvailable) =="
mem=( "$OUT"/worker_*_mem.txt )
if [ ${#mem[@]} -gt 0 ]; then
  for f in "${mem[@]}"; do echo "--- $(basename "$f") ---"; tail -n 25 "$f"; done
else
  echo "(no worker_*_mem.txt)"
fi

echo "== member weights present? =="
find "$OUT" -name 'manifest.json' -o -name '*.pt' -o -name '*.pth' 2>/dev/null | head -20
echo "== watch end $(date +%H:%M:%S) =="
