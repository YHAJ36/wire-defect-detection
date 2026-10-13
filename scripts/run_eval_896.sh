#!/bin/bash
# 896 组 test 集评估（第三步）—— 此前只有 val 读数（v11 0.8990 / v5 0.9090），
# 按项目口径 test 是必补项。
#
# 口径：split=test / batch=8 / workers=0 / device=0（与 640 组一致），imgsz=896。
# 896 组只有 v11 baseline + v5 两组，仅作同构子集对比（v11 vs v11、v5 vs v5），
# 不与 640 的 7 组均值混算。
#
# 日志：logs/eval_896.log
D:/yolo-env/Scripts/python.exe D:/2/scripts/fix_settings.py

echo "=========== 896 test: YOLOv11 基线 ==========="
D:/yolo-env/Scripts/python.exe D:/2/code/tools/eval/eval_test_896.py

echo "=========== 896 test: YOLOv5 基线 ==========="
cd /d/2/code/baselines/yolov5
D:/2/envs/yolov5-env/Scripts/python.exe eval_test_v5_896.py

echo "=========== 896 test 评估完成 ==========="
