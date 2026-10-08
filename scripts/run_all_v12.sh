#!/bin/bash
cd /d/2/code/yolov12
D:/yolo-env/Scripts/python.exe train_v12_ablation.py --name wire_v12s_v3
D:/yolo-env/Scripts/python.exe train_v12_ablation.py --cbam --name wire_v12s_cbam_v3
