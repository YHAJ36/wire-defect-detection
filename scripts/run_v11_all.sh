#!/bin/bash
D:/yolo-env/Scripts/python.exe D:/2/scripts/fix_settings.py   # settings 被重置时自动恢复

cd /d/2/code/yolov11
D:/yolo-env/Scripts/python.exe train_ablation.py --mode baseline --name wire_v11
D:/yolo-env/Scripts/python.exe train_ablation.py --mode backbone --name wire_v11_backbone
D:/yolo-env/Scripts/python.exe train_ablation.py --mode neck_sppf --name wire_v11_neck
D:/yolo-env/Scripts/python.exe train_ablation.py --mode both --name wire_v11_both
