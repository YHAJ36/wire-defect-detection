#!/usr/bin/env bash
# P2v2 三种子正式训练（串行）
# 结构：code/yolov11/yolo11s-p2v2.yaml（唯一结构变量）
# batch=4 + ultralytics 内置梯度累积（nbs=64 => 累积 16 步，等效 batch 64，
# 与基线 batch8 x accum8 完全等价，weight_decay 不缩放）
# 其余超参与封版 896 基线逐项一致。
set -u
PY=D:/yolo-env/Scripts/python.exe
cd D:/2 || exit 1
LOG=D:/2/exp_upscale/logs/p2v2_3seeds_$(date +%Y%m%d).log
mkdir -p D:/2/exp_upscale/logs

for s in 42 43 44; do
  echo "===== seed ${s} START $(date '+%F %T') =====" >> "$LOG"
  "$PY" code/yolov11/train_phase1_896.py \
      --model D:/2/code/yolov11/yolo11s-p2v2.yaml \
      --pretrained D:/1/models/yolo11s.pt \
      --batch 4 --seed "${s}" --epochs 60 --imgsz 896 \
      --name "p2v2_v11_896_s${s}" >> "$LOG" 2>&1
  rc=$?
  echo "===== seed ${s} END rc=${rc} $(date '+%F %T') =====" >> "$LOG"
  if [ $rc -ne 0 ]; then
    echo "!!!!! seed ${s} 失败，中止后续种子，等待人工检查" >> "$LOG"
    exit $rc
  fi
done
echo "===== ALL DONE $(date '+%F %T') =====" >> "$LOG"
