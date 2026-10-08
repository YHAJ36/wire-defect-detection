#!/bin/bash
cd /d/2/code/baselines/yolov5
D:/2/envs/yolov5-env/Scripts/python.exe train_v5.py

cd /d/2/code/yolov11
D:/yolo-env/Scripts/python.exe train_v11_ablation.py --mode baseline --name wire_v11_v2
D:/yolo-env/Scripts/python.exe train_v11_ablation.py --mode backbone --name wire_v11_backbone_v2
D:/yolo-env/Scripts/python.exe train_v11_ablation.py --mode neck_sppf --name wire_v11_neck_v2
D:/yolo-env/Scripts/python.exe train_v11_ablation.py --mode both --name wire_v11_both_v2
