#!/bin/bash
# 统一优化器为 SGD 后的重跑（v5 不动，v11 4 组 + v12 2 组）
cd /d/2/code/yolov11
D:/yolo-env/Scripts/python.exe train_v11_ablation.py --mode baseline --name wire_v11_v3
D:/yolo-env/Scripts/python.exe train_v11_ablation.py --mode backbone --name wire_v11_backbone_v3
D:/yolo-env/Scripts/python.exe train_v11_ablation.py --mode neck_sppf --name wire_v11_neck_v3
D:/yolo-env/Scripts/python.exe train_v11_ablation.py --mode both --name wire_v11_both_v3

cd /d/2/code/yolov12
D:/yolo-env/Scripts/python.exe train_v12_ablation.py --name wire_v12s_v4
D:/yolo-env/Scripts/python.exe train_v12_ablation.py --cbam --name wire_v12s_cbam_v4
