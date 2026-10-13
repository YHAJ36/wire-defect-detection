#!/bin/bash
D:/yolo-env/Scripts/python.exe D:/2/scripts/fix_settings.py   # settings 被重置时自动恢复

cd /d/2/code/yolov11
D:/yolo-env/Scripts/python.exe train_v11_ablation_adamw.py --mode baseline --name wire_v11_adamw
D:/yolo-env/Scripts/python.exe train_v11_ablation_adamw.py --mode backbone --name wire_v11_backbone_adamw
D:/yolo-env/Scripts/python.exe train_v11_ablation_adamw.py --mode neck_sppf --name wire_v11_neck_adamw
D:/yolo-env/Scripts/python.exe train_v11_ablation_adamw.py --mode both --name wire_v11_both_adamw

cd /d/2/code/yolov12
D:/yolo-env/Scripts/python.exe train_v12_ablation_adamw.py --name wire_v12s_adamw
D:/yolo-env/Scripts/python.exe train_v12_ablation_adamw.py --cbam --name wire_v12s_cbam_adamw

cd /d/2/code/baselines/yolov5
D:/2/envs/yolov5-env/Scripts/python.exe train_v5_adamw.py
