#!/usr/bin/env bash
# Queue several GPU training seeds back to back, downloading each run before starting
# the next. The GPU twin of queue_seeds.sh, which only ever handled the TPU kernel.
#
#   用法:  bash scripts/queue_gpu_seeds.sh 2027 2028 2029
#
# Why this exists: growing the ensemble means pushing the SAME kernel with a different
# SEED, and Kaggle serves the output of the latest version only. Two things make the
# hand-rolled version of that loop expensive:
#
#   1. `kaggle kernels push -p kernel-train` pushes whatever is in kernel-train/, and the
#      generator writes to notebooks/. Forget the copy and you re-run the seed that is
#      already banked - member ids hash fold+seed, so every member deduplicates away and
#      the hours are gone. Step 3 below refuses to push in that case.
#   2. Pushing while a run is in flight cancels it and takes its output with it. The
#      start guard below, and the wait-for-terminal inside the loop, exist for that.
#
# Seed 2026 is the run already in flight (224x12 + dinov2-large; its output lands in
# outputs/gpu_weights/v4_224x12), so start this at 2027.
#
# VARIANT selects the backbone and defaults to large. It is what makes the
# confound-isolating run a single command:
#     VARIANT=small bash scripts/queue_gpu_seeds.sh 2027
# small at 224x12 differs from the banked 224x6 small baseline in exactly ONE variable -
# the slice-group count - and it trains fast enough to reach ~17 epochs inside the same 8h
# budget, where large manages 4-5. HANDOFF 2.5 says when to prefer which.
# REQUIREMENT: kernel-train/kernel-metadata.json must attach a checkpoint for the chosen
# variant, because find_dinov2 falls back to whatever it found first and build_model takes
# the architecture from THAT checkpoint - a mismatch is silent at training time. The
# metadata currently attaches both variants, which is what makes this safe.
#
# EPOCHS is NOT a knob for saving time on the large config. The loop runs
# min(EPOCHS, whatever the per-fold share of TIME_BUDGET affords), and for DINOv2-large at
# 224px that share is ~1.45h against ~18 min/epoch, so only ~4-5 epochs happen whatever
# EPOCHS says - every run costs the full 8h of quota either way.
#
# What EPOCHS does control is the OneCycleLR schedule length
# (total_steps = EPOCHS * steps_per_epoch). Leave it far above what fits and the schedule
# never anneals: with EPOCHS=20 and 4 epochs run, the LR is still near its peak at the
# truncation point, so the members are cut off mid-schedule. Set EPOCHS to what actually
# fits (~5 for large) and the cycle completes inside the same budget.
#
# Read the real per-fold epoch count from the previous run's manifest (`ep` column of
# scripts/manifest_report.py) and pass EPOCHS accordingly:
#     EPOCHS=5 bash scripts/queue_gpu_seeds.sh 2027
set -u

KERNEL="ppxl16/rsna-knee-train-v1"
ROOT="/d/kaggle/rsna-knee"
ACC="$ROOT/outputs/gpu_weights"
EPOCHS="${EPOCHS:-5}"
VARIANT="${VARIANT:-large}"
# BACKBONE selects the visual encoder family. biomedclip ignores VARIANT for the
# architecture (its tower is always a ViT-B/16) but the generated notebook still carries
# BACKBONE_VARIANT="biomedclip-vitb16", which is what step 3 must expect.
#     BACKBONE=biomedclip bash scripts/queue_gpu_seeds.sh 2026
BACKBONE="${BACKBONE:-dinov2}"
POLL_INTERVAL=120
POLL_MAX=320          # ~10.5h per seed (GPU queue is short; training is the time budget)
export PYTHONUTF8=1

mkdir -p "$ACC"

if [ "$#" -eq 0 ]; then
  echo "usage: bash scripts/queue_gpu_seeds.sh <seed> [seed...]" >&2
  exit 2
fi

# 启动保护: 有 run 在跑就先停, 别把人家 cancel 了 (那会连同它的 output 一起丢掉)。
# FAIL CLOSED: 只有确证 kernel 已在终态才继续。原来写成 `if [ "$cur" = QUEUED ] || ...`
# 是 fail-open 的 —— 一旦 status 读不出来 (PATH 缺 grep/head 时 cur 为空), 条件为假,
# 守卫就静默放行。DSH 2026-09-11 实测踩到过。
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
  echo "===== seed $SEED variant $VARIANT @ $(date +%H:%M:%S) ====="

  # 1. 生成该 seed 的 notebook (写到 notebooks/)
  python "$ROOT/scripts/build_train_notebook.py" --seed "$SEED" --epochs "$EPOCHS" \
         --variant "$VARIANT" --backbone "$BACKBONE" \
    || { echo "!! build failed for seed $SEED"; exit 1; }

  # 2. 覆盖 push 目录
  cp "$ROOT/notebooks/rsna-knee-train-v1.ipynb" \
     "$ROOT/kernel-train/rsna-knee-train-v1.ipynb"

  # 3. 验证 push 目录里确实是这个 seed 和这个 backbone —— 忘了第 2 步就要在这里被拦下。
  #    两个都要验: id 是 md5(pixel_group|fold|seed), 而 pixel_group 不含 backbone, 所以换
  #    backbone 而不换 seed 会撞出同样的 id, 被打包时的先到者胜静默丢掉。
  #    期望值要跟生成器实际写进 BACKBONE_VARIANT 的字符串一致 (见 build_train_notebook.py
  #    的 _TAG), 否则 biomedclip 构建会被自己的校验拦下。
  if [ "$BACKBONE" = "biomedclip" ]; then EXPECT_TAG="biomedclip-vitb16"; else EXPECT_TAG="$VARIANT"; fi
  if ! python "$ROOT/scripts/check_notebook_seed.py" \
        "$ROOT/kernel-train/rsna-knee-train-v1.ipynb" "$SEED" --variant "$EXPECT_TAG"; then
    echo "!! refusing to push: kernel-train/ does not carry seed $SEED / backbone $BACKBONE" >&2
    exit 1
  fi

  # 4. push (save & run)
  (cd "$ROOT/kernel-train" && kaggle kernels push) || { echo "!! push failed"; exit 1; }
  echo "  pushed seed $SEED backbone $BACKBONE variant $VARIANT (epochs=$EPOCHS), polling every ${POLL_INTERVAL}s..."

  # 5. 等终态
  s=""
  for i in $(seq 1 "$POLL_MAX"); do
    s=$(kaggle kernels status "$KERNEL" 2>/dev/null | grep -oE 'KernelWorkerStatus\.[A-Z]+' | head -1)
    case "$s" in
      KernelWorkerStatus.COMPLETE) break ;;
      KernelWorkerStatus.ERROR|KernelWorkerStatus.CANCELLED)
        echo "!! seed $SEED -> $s"; exit 1 ;;
    esac
    sleep "$POLL_INTERVAL"
  done
  if [ "$s" != "KernelWorkerStatus.COMPLETE" ]; then
    echo "!! seed $SEED timeout after $((POLL_MAX*POLL_INTERVAL))s (status=$s)"; exit 1
  fi

  # 6. 立刻下载 (只有最新版本的 output 可取)
  outdir="$ACC/seed${SEED}_224x12_${VARIANT}"
  mkdir -p "$outdir"
  echo "  == terminal, waiting 30s for the archive =="
  sleep 30
  ok=""
  for try in 1 2 3 4 5; do
    # 空目录 + exit 0 是 CLI 在 RUNNING 时的行为, 所以数文件而不是看退出码
    (cd "$ROOT" && kaggle kernels output "$KERNEL" -p "$outdir" >/dev/null 2>&1)
    n=$(find "$outdir" -type f | wc -l)
    echo "  download attempt $try/5: $n file(s)"
    [ "$n" -gt 0 ] && { ok=1; break; }
    sleep 60
  done
  [ -n "$ok" ] || { echo "!! seed $SEED: 5 attempts produced no files"; exit 1; }

  # 7. 验证拿到带 members 的 manifest
  manfile=$(find "$outdir" -name manifest.json 2>/dev/null | head -1)
  if [ -z "$manfile" ]; then
    echo "!! seed $SEED: no manifest.json under $outdir"; find "$outdir" -type f | head -20; exit 1
  fi
  echo "  seed $SEED done -> $manfile"
  python "$ROOT/scripts/manifest_report.py" "$manfile" | sed 's/^/    /'
done

echo "===== all seeds done @ $(date +%H:%M:%S) ====="
echo "merge every run under outputs/gpu_weights (v4_224x12 plus the seed dirs) into one"
echo "private dataset. Mixed variants are fine: they differ in pixel_group, so the"
echo "inference side decodes them as separate groups and averages the members:"
echo "  python scripts/package_weights.py --from-dir outputs/gpu_weights \\"
echo "      --dataset ppxl16/rsna-knee-train-weights-224x12 \\"
echo "      --title \"RSNA Knee train weights 224x12\" --dry-run"
