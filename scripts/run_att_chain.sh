#!/usr/bin/env bash
# 注意力臂批次 —— 单卡串行跑批（5 个 run）。
#   顺序：噪声地板 baseline s42/s43/s44 → CBAM+BN s42 → Triplet s42
#   理由（用户 2026-10-15 拍板）：臂的效应只有 ±0.013(pt) 量级，
#   必须先有 3 种子的地板块，否则"涨了/跌了"不可判读。地板先跑。
#
# 口径：全部 workers=0（与被封版报告的 CBAM 臂同管线），640 / 60 轮 / SGD。
#   产物落 D:/2/exp_upscale/runs/（不污染已封版 runs/）。
#   预计 ~70 min/run，5 个 run 约 5.8 h。
#
# 用法（必须用绝对路径解释器；本机 bare python 没有 ultralytics）：
#   bash D:/2/scripts/run_att_chain.sh
set -u

PY='D:/yolo-env/Scripts/python.exe'
TRAIN='D:/2/code/yolov11/train_att_640.py'
EVAL='D:/2/code/tools/eval/eval_att_640.py'
LOGDIR='D:/2/logs'
CHAIN="$LOGDIR/att_chain.log"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN"; }

# GPU 空闲等待（<3000 MiB 才开跑）。
#   ⚠ 阈值不能照抄 run_batch1.sh 的 1000：本机桌面进程常占 ~1.6 GB，
#     空载也降不到 1000 以下（会一直空等到超时）。空载 ≈1.6 G、训练中 ≈6 G，
#     故取 3000 作为分界。
wait_gpu_free() {
  local used
  for _ in $(seq 1 240); do           # 最多等 4 h
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${used:-99999}" -lt 3000 ]; then return 0; fi
    say "GPU 仍占用 ${used} MiB，等 60 s…"
    sleep 60
  done
  say "❌ 等 GPU 超时"
  return 1
}

run_one() {  # $1=att  $2=seed
  local att=$1 seed=$2
  say "START  att=$att seed=$seed"
  "$PY" "$TRAIN" --att "$att" --seed "$seed" \
      >> "$LOGDIR/att_train_${att}_s${seed}.log" 2>&1
  local rc=$?
  say "END    att=$att seed=$seed  rc=$rc"
  return $rc
}

mkdir -p "$LOGDIR"
wait_gpu_free || exit 1

say '================ 注意力臂批次开始 ================'
run_one baseline 42
run_one baseline 43
run_one baseline 44
run_one cbam_bn  42
run_one triplet  42
say '================ 5 个 run 全部训练结束 ================'

# 全量评估（val 选型复核 + test 判读 + 噪声地板 + 臂 vs 地板）
say 'START  eval_att_640（全量）'
"$PY" "$EVAL" >> "$LOGDIR/att_eval.log" 2>&1
say "END    eval_att_640  rc=$?"
say '================ 批次完成 ================'
