#!/usr/bin/env bash
# 监控 GPU 训练 kernel 到终态 -> 下载 output -> 打印 manifest 摘要 + 关键日志行。
# 顺带每轮试一次 `kaggle kernels logs`，记录它 RUNNING 期间到底会不会返回内容
# （写到 $OUT/log_probe.txt），这是判断「能不能实时看训练日志」的实测。
#
#   用法: bash scripts/watch_train.sh
set -u
export PYTHONUTF8=1

KERNEL="ppxl16/rsna-knee-train-v1"
ROOT="/d/kaggle/rsna-knee"
OUT="$ROOT/outputs/gpu_weights/v4_224x12"
mkdir -p "$OUT"
PROBE="$OUT/log_probe.txt"
: > "$PROBE"

echo "== watch start $(date +%H:%M:%S) =="
prev=""
while true; do
  s=$(kaggle kernels status "$KERNEL" 2>/dev/null | grep -oE 'KernelWorkerStatus\.[A-Z]+' | head -1)
  [ "$s" != "$prev" ] && { echo "[$(date +%H:%M:%S)] status=$s"; prev="$s"; }
  if [ "$s" = "KernelWorkerStatus.RUNNING" ]; then
    out=$(timeout 180 kaggle kernels logs "$KERNEL" 2>/dev/null)
    n=$(printf '%s' "$out" | grep -c .)
    last=$(printf '%s' "$out" | grep . | tail -1 | cut -c1-150)
    echo "$(date +%H:%M:%S) RUNNING loglines=$n last=[$last]" >> "$PROBE"
  fi
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
ls -la "$OUT" | head -30
echo "== weights/ =="
ls -la "$OUT/weights" 2>/dev/null | head -20 || echo "(no weights/ dir)"

echo "== log probe (was the log readable while RUNNING?) =="
head -3 "$PROBE"; echo "  ..."; tail -5 "$PROBE"
echo "  RUNNING polls with content: $(grep -c 'loglines=[1-9]' "$PROBE" 2>/dev/null || echo 0)"

echo "== key log lines (cache layout / step / epoch / member) =="
LOG="$OUT/rsna-knee-train-v1.log"
if [ -f "$LOG" ]; then
  grep -a "plan_cache\|cache layout\|memory: \|: cache \|fold .*: train\|epoch \|saved member\|best holdout\|Error\|Traceback\|error" "$LOG" \
    | sed 's/^,*{"stream_name":"stdout","time":[0-9.]*,"data":"//; s/"}$//' \
    | tail -60
else
  echo "(no log file; listing what did arrive)"
  find "$OUT" -type f | head -20
fi
echo "== watch end $(date +%H:%M:%S) =="
