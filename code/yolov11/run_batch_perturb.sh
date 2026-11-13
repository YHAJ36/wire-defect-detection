#!/usr/bin/env bash
# =============================================================================
# 数据侧扰动实验链（2026-10-29）
#
#   起因：iou_diagnosis.py 拆出「模型方差」与「标注偏离」两个不同问题
#         break   A=0.695  B=0.662  → 瓶颈在模型方差（小目标）
#         thunder A=0.901  B=0.678  → 瓶颈才像标注（A−B=0.223）
#   所以用三个**不改网络结构**的单变量分头检验，每个 3 种子：
#
#   组 A  jitter  每轮重抖 GT 框（sigma=0.08），阻止模型精确认某一帧的画法
#                 判读：若有效应主要出现在 test（跨段）而非 val（同段）
#   组 B  drop    删掉「三颗种子都不认同」的训练 GT 框（IoU<0.3）
#                 判读：若无效说明噪声是均匀抖动而非局部错标，洗不掉
#   组 C  ms      开 multi_scale（本项目从未试过；直接针对小目标）
#                 判读：break 的小目标方差 A=0.695 若被拉高，就是它
#
#   ⚠ 三组都跑在 dataset_noleak（1401/181/203），与 v3 口径 74.67 直接可比。
#   ⚠ 判据不是「比基线高」，而是**配对 bootstrap CI 不跨零**（见 bootstrap_ci_noleak.py）。
#      单侧 CI 宽 10.4 pt，配对地板 3.4 pt，未配对比较在这套数据上没有意义。
#
# 用法
#   bash code/yolov11/run_batch_perturb.sh --smoke    # 只冒烟（约 6 分钟）
#   bash code/yolov11/run_batch_perturb.sh --chat     # 跑 A + C（约 9.5 小时）
#   bash code/yolov11/run_batch_perturb.sh --all      # A + C + B
#   bash code/yolov11/run_batch_perturb.sh --eval     # 只评估
# =============================================================================
set -uo pipefail

PY="D:/yolo-env/Scripts/python.exe"
TRAIN="D:/2/code/yolov11/train_perturb.py"
DATA="D:/2/dataset_noleak/data.yaml"
PROJ="D:/2/exp_upscale/runs"
LOGDIR="D:/2/exp_upscale/logs"
LOG="$LOGDIR/perturb_chain.log"
DROP="$LOGDIR/unreliable_train.tsv"

IMGSZ=896
EPOCHS=60
JIT=0.08

mkdir -p "$LOGDIR"
log()  { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
rule() { log "--------------------------------------------------------------"; }

epochs_done() {
  local csv="$PROJ/$1/results.csv"
  if [ -f "$csv" ]; then echo $(( $(wc -l < "$csv") - 1 )); else echo 0; fi
}

train_one() {   # $1=tag  $2=seed  $3=name  $4..=extra args
  local tag=$1 seed=$2 name=$3; shift 3
  local got; got=$(epochs_done "$name")
  if [ "$got" -ge "$EPOCHS" ]; then
    log "↷ 跳过 $name（已 $got/$EPOCHS 轮）"; return 0
  fi
  local resume=''
  if [ "$got" -gt 0 ] && [ -f "$PROJ/$name/weights/last.pt" ]; then
    log "⚠ $name 只跑完 $got/$EPOCHS 轮，改为 --resume 续训"
    resume='--resume'
  fi
  log "▶ 开始 $name（$tag, seed=$seed）"
  local t0; t0=$(date +%s)
  "$PY" "$TRAIN" --att baseline --seed "$seed" --epochs "$EPOCHS" \
        --imgsz "$IMGSZ" --data "$DATA" --name "$name" --project "$PROJ" \
        $resume "$@" >> "$LOG" 2>&1
  local rc=$?
  local fin; fin=$(epochs_done "$name")
  if [ $rc -ne 0 ] || [ "$fin" -lt "$EPOCHS" ]; then
    log "✗ $name 失败/未完成（rc=$rc, $fin/$EPOCHS 轮）。续训："
    log "   \"$PY\" $TRAIN --att baseline --seed $seed --epochs $EPOCHS --imgsz $IMGSZ --data $DATA --name $name --project $PROJ --resume $*"
    exit 1
  fi
  log "✓ $name 完成，用时 $(( ($(date +%s)-t0)/60 )) 分钟"
}

preflight() {
  rule
  log "预检"
  [ -x "$PY" ]   || { log "✗ 缺 python $PY"; exit 1; }
  [ -f "$TRAIN" ]|| { log "✗ 缺 $TRAIN"; exit 1; }
  [ -f "$DATA" ] || { log "✗ 缺 $DATA"; exit 1; }
  local ntr; ntr=$(ls D:/2/dataset_noleak/images/train | wc -l)
  log "train 图 $ntr（应为 1401）"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
    | sed 's/^/    GPU: /' | tee -a "$LOG"
}

smoke() {
  rule; log "冒烟：抖动 1 轮 / 3% 数据"
  "$PY" "$TRAIN" --att baseline --epochs 1 --fraction 0.03 --gt-jitter "$JIT" \
        --imgsz 896 --data "$DATA" --name smoke_perturb_jit --project "$PROJ" \
        >> "$LOG" 2>&1 && log "✓ 抖动冒烟通过" || { log "✗ 抖动冒烟失败"; exit 1; }
  log "冒烟：multi_scale 1 轮 / 3% 数据"
  "$PY" "$TRAIN" --att baseline --epochs 1 --fraction 0.03 --multi-scale \
        --imgsz 896 --data "$DATA" --name smoke_perturb_ms --project "$PROJ" \
        >> "$LOG" 2>&1 && log "✓ multi_scale 冒烟通过" || { log "✗ multi_scale 冒烟失败"; exit 1; }
}

group_A() {
  rule; log "组 A：GT 抖动注入 sigma=$JIT（3 种子）"
  for s in 42 43 44; do
    train_one "jit$JIT" "$s" "nf_v11_noleak_jit08_s$s" --gt-jitter "$JIT"
  done
}

group_C() {
  rule; log "组 C：multi_scale（3 种子）"
  for s in 42 43 44; do
    train_one "multi_scale" "$s" "nf_v11_noleak_ms_s$s" --multi-scale
  done
}

group_B() {
  rule; log "组 B：删除不可信框（3 种子）"
  if [ ! -f "$DROP" ]; then
    log "✗ 缺 $DROP，先跑 code/tools/eval/make_unreliable_tsv.py"
    exit 1
  fi
  for s in 42 43 44; do
    train_one "drop" "$s" "nf_v11_noleak_drop_s$s" --drop-file "$DROP"
  done
}

case "${1:---all}" in
  --smoke) smoke ;;
  --chat)  preflight; group_A; group_C; log "完成（A+C）" ;;
  --all)   preflight; group_A; group_C; group_B; log "完成（A+B+C）" ;;
  --a)     preflight; group_A ;;
  --b)     preflight; group_B ;;
  --c)     preflight; group_C ;;
  *) echo "用法: $0 [--smoke|--chat|--all|--a|--b|--c]"; exit 1 ;;
esac
