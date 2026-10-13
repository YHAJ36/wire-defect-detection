#!/bin/bash
# AdamW 7 组 test 集评估（6 组 ultralytics + 1 组 YOLOv5），串行，避免 GPU 争用与 cache 竞争。
# 日志：D:/2/eval_test_adamw.log
D:/yolo-env/Scripts/python.exe D:/2/scripts/fix_settings.py   # settings 被重置时自动恢复

D:/yolo-env/Scripts/python.exe /d/2/eval_test_adamw.py

cd /d/2/code/baselines/yolov5 || exit 1
D:/2/envs/yolov5-env/Scripts/python.exe eval_test_v5_adamw.py
