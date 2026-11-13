#!/usr/bin/env bash
# 批 4-fix：A.9.42 审计后的三臂，按「单变量可归因」设计。
#
# 为什么要一个 ctrl 臂
#   审计把 best.pt 的挑选口径从 val NMS 0.7 改成 0.5（P0 修复）。这一改会动数字，
#   所以任何新臂都不能直接跟旧的 nf_v11_896_s42（0.9197）比 —— 必须先有一个
#   「只改口径、不改别的」的对照，所有新臂都跟它比。
#
# 三臂
#   ctrl  只改 val-iou 0.5                —— 新基线，1.9 h
#   wts   cls 0.5→0.3, dfl 1.5→2.0, mixup 0.1→0 —— 权重配比单变量，1.9 h
#   bud   epochs 60→150, nbs 64→32, close_mosaic 5→12 —— 预算单变量，约 4.5 h
#
# 判读
#   三臂都按 test / NMS 0.5、同 seed 42 配对；先看 wts 与 bud 各自相对 ctrl 的差，
#   再决定是否补 s43/s44。效应 <0.6 pt 视为测不出（地板三种子跨度仅 0.24 pt，见 A.9.41）。
#
# 用法（GPU 空闲后）：
#   bash code/tools/run_batch4_fix.sh            # 顺序跑三臂
#   bash code/tools/run_batch4_fix.sh ctrl       # 只跑某一臂
set -u
cd /d/2 || exit 1

PY=D:/yolo-env/Scripts/python.exe
TRAIN=code/yolov11/train_att_640.py
LOG=exp_upscale/logs
mkdir -p "$LOG"

run_arm() {
  local name="$1"; shift
  echo "=== $(date +%F' '%T)  start $name ==="
  $PY $TRAIN --att baseline --mode neck_sppf --imgsz 896 --seed 42 \
      --name "$name" "$@" > "$LOG/chain_batch4_${name}.log" 2>&1
  local rc=$?
  echo "=== $(date +%F' '%T)  end   $name  rc=$rc ==="
  if [ $rc -ne 0 ]; then
    echo "!! $name 失败，见 $LOG/chain_batch4_${name}.log"
    return $rc
  fi
  $PY code/tools/eval/eval_nms05.py "$name"
}

WHICH="${1:-all}"

if [ "$WHICH" = all ] || [ "$WHICH" = ctrl ]; then
  run_arm nf_v11_896_ctrl_s42 --epochs 60 --val-iou 0.5
fi

if [ "$WHICH" = all ] || [ "$WHICH" = wts ]; then
  run_arm nf_v11_896_wts_s42 --epochs 60 --val-iou 0.5 \
      --cls 0.3 --dfl 2.0 --mixup 0.0
fi

if [ "$WHICH" = all ] || [ "$WHICH" = bud ]; then
  run_arm nf_v11_896_bud_s42 --epochs 150 --val-iou 0.5 \
      --nbs 32 --close-mosaic 12
fi

echo "=== 全部完成 $(date +%F' '%T) ==="
