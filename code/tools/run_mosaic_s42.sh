#!/usr/bin/env bash
# mosaic 削弱两臂（A.9.45 唯一未试且机理对得上的方向）
#
# 单因素隔离：与 nf_v11_896_s42 逐项一致，只改一处
#   基线：mosaic 0.8 / close_mosaic 5 / mixup 0.1 / copy_paste 0.1 / scale 0.5
#         translate 0.1 / erasing 0.4 / randaugment / SGD lr0 .01 lrf .005 cos_lr
#         seed 42 / epochs 60 / batch 8 / imgsz 896 / nbs 64 / box 7.5 cls .5 dfl 1.5
#   臂1：--mosaic 0.4        （只降拼接总量，收尾仍关 5 轮）
#   臂2：--close-mosaic 20   （只让末段干净，拼接总量不变）
#
# 两臂机理不同，都必须跑：
#   只有臂2有效 ⇒ 问题在"末段仍被拼接污染"
#   只有臂1有效 ⇒ 问题是"整体拼接过度"
#
# 判定：357 图合并集（_valtest），break AP50 提升 ≥1.6 pt（A.9.49 实测种子跨度）才算数。
#       last.pt 为主对比（无挑选），best.pt 作参考。
set -u
cd /d/2 || exit 1

PY=D:/yolo-env/Scripts/python.exe
LOG=exp_upscale/logs/chain_mosaic_s42.log
COMMON="--att baseline --mode neck_sppf --imgsz 896 --seed 42 --epochs 60"

echo "=== [1/2] mosaic 0.4  start $(date '+%F %H:%M:%S') ===" | tee -a "$LOG"
$PY code/yolov11/train_att_640.py $COMMON --mosaic 0.4 \
    --name nf_v11_896_mos04_s42 2>&1 | tee -a "$LOG"
echo "=== [1/2] done  $(date '+%F %H:%M:%S') ===" | tee -a "$LOG"

echo "=== [2/2] close_mosaic 20  start $(date '+%F %H:%M:%S') ===" | tee -a "$LOG"
$PY code/yolov11/train_att_640.py $COMMON --close-mosaic 20 \
    --name nf_v11_896_cm20_s42 2>&1 | tee -a "$LOG"
echo "=== [2/2] done  $(date '+%F %H:%M:%S') ===" | tee -a "$LOG"
echo "=== ALL DONE  $(date '+%F %H:%M:%S') ===" | tee -a "$LOG"
