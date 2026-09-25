#!/usr/bin/env bash
# 串行跑多个 seed 的 TPU 训练，每个 seed 的权重下载到 outputs/tpu_weights/seed<SEED>/。
#
#   用法:  bash scripts/queue_seeds.sh 2027 2028 2029
#
# 每个 seed 流程: build_tpu_train.py --seed N -> 覆盖 push 目录 -> kaggle kernels push
# (触发一次 run) -> 轮询 status 到 COMPLETE/ERROR -> kernels output 下载。ERROR 或超时
# 即中断退出(不浪费后续 seed 的 TPU 时间)。全部完成后用 package_weights.py --from-dir
# 合并成一个私有权重数据集。
#
# 启动保护: 若当前 kernel 还有 run 在 QUEUED/RUNNING, 直接退出(避免 push 覆盖取消掉
# 正在跑的 run)。所以等上一个 run COMPLETE 后再启动本脚本。
set -u

KERNEL="ppxl16/rsna-knee-tpu-train-v1"
ROOT="/d/kaggle/rsna-knee"
ACC="$ROOT/outputs/tpu_weights"
POLL_INTERVAL=120
POLL_MAX=400          # ~13h per seed (排队 ~0.5h + 训练 ~8h + 缓冲)
export PYTHONUTF8=1

mkdir -p "$ACC"

# 启动保护: 有 run 在跑就先停, 别把人家 cancel 了。
# FAIL CLOSED 版本 —— 原文是 `if [ "$cur" = QUEUED ] || [ "$cur" = RUNNING ]`, 一旦
# status 读不出来 (例如 PATH 里没有 grep/head, cur 为空) 条件就为假、守卫静默放行。
# DSH 2026-09-11 在 GPU 版上实测踩到过这个 fail-open, 这里一并修掉。
cur=$(kaggle kernels status "$KERNEL" 2>/dev/null | grep -oE 'KernelWorkerStatus\.[A-Z]+' | head -1)
case "$cur" in
  KernelWorkerStatus.COMPLETE|KernelWorkerStatus.ERROR|KernelWorkerStatus.CANCELLED)
    : ;;
  *)
    echo "!! cannot confirm $KERNEL is terminal (status read as '$cur')" >&2
    echo "   refusing to push: a push replaces the version, and Kaggle serves only the" >&2
    echo "   latest version, so pushing over a live run destroys its output." >&2
    exit 1 ;;
esac

for SEED in "$@"; do
  echo "===== seed $SEED @ $(date +%H:%M:%S) ====="

  # 1. 生成该 seed 的 notebook
  python "$ROOT/scripts/build_tpu_train.py" --seed "$SEED" || { echo "!! build failed"; exit 1; }

  # 2. 覆盖 push 目录
  cp "$ROOT/notebooks/rsna-knee-tpu-train-v1.ipynb" \
     "$ROOT/kernel-tpu-train/rsna-knee-tpu-train-v1.ipynb"

  # 3. push (save & run)
  (cd "$ROOT/kernel-tpu-train" && kaggle kernels push) || { echo "!! push failed"; exit 1; }
  echo "  pushed, waiting for terminal status (poll ${POLL_INTERVAL}s)..."

  # 4. 等终态
  s=""
  for i in $(seq 1 "$POLL_MAX"); do
    s=$(kaggle kernels status "$KERNEL" 2>/dev/null | grep -oE 'KernelWorkerStatus\.[A-Z]+' | head -1)
    case "$s" in
      KernelWorkerStatus.COMPLETE) break ;;
      KernelWorkerStatus.ERROR|KernelWorkerStatus.CANCELLED) echo "!! seed $SEED -> $s"; exit 1 ;;
    esac
    sleep "$POLL_INTERVAL"
  done
  if [ "$s" != "KernelWorkerStatus.COMPLETE" ]; then
    echo "!! seed $SEED timeout after $((POLL_MAX*POLL_INTERVAL))s (status=$s)"; exit 1
  fi

  # 5. 下载 output (COMPLETE 后可能还要几秒生成 output, 失败重试)
  outdir="$ACC/seed$SEED"
  mkdir -p "$outdir"
  for try in 1 2 3; do
    if (cd "$ROOT" && kaggle kernels output "$KERNEL" -p "$outdir" 2>/dev/null); then
      break
    fi
    echo "  output download retry $try/3"; sleep 60
  done

  # 6. 验证下载到了带 members 的 manifest
  manfile=$(find "$outdir" -name manifest.json 2>/dev/null | head -1)
  if [ -z "$manfile" ]; then
    echo "!! seed $SEED: no manifest.json under $outdir"; ls -R "$outdir" 2>/dev/null | head -30; exit 1
  fi
  n=$(grep -o '"file"' "$manfile" | wc -l)
  echo "  seed $SEED done: $n member(s) -> $outdir"
done

echo "===== all seeds done @ $(date +%H:%M:%S) ====="
echo "merge + package:"
echo "  python scripts/package_weights.py --from-dir $ACC --dataset ppxl16/rsna-knee-tpu-weights"
