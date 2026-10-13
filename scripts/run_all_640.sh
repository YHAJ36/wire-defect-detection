#!/bin/bash
# 640 全套：v11×3 (CBAM 消融) → v12×2 → v5×1，各 60ep 串行，末尾自动跑 test 评估。
# 说明：train_v11_640.py (baseline) 已于 17:05 单独跑完，故此处只补缺的 3 组 v11。
# 日志：D:/2/train_640_all.log
D:/yolo-env/Scripts/python.exe D:/2/scripts/fix_settings.py   # settings 被重置时自动恢复

cd /d/2/code/yolov11
D:/yolo-env/Scripts/python.exe train_v11_640_neck.py
D:/yolo-env/Scripts/python.exe train_v11_640_backbone.py
D:/yolo-env/Scripts/python.exe train_v11_640_both.py

cd /d/2/code/yolov12
D:/yolo-env/Scripts/python.exe train_v12_640.py
D:/yolo-env/Scripts/python.exe train_v12_640_cbam.py

cd /d/2/code/baselines/yolov5
D:/2/envs/yolov5-env/Scripts/python.exe train_v5_640.py

# ===== 跑完自动 test 评估 =====
echo "=========== 640 全套 test 评估 ==========="
D:/yolo-env/Scripts/python.exe /d/2/eval_test_640.py

cd /d/2/code/baselines/yolov5
D:/2/envs/yolov5-env/Scripts/python.exe eval_test_v5_640.py
echo "=========== 640 全套 完成 ==========="
