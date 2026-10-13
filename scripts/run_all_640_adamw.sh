#!/bin/bash
# 640 + AdamW 全套：v11×4（baseline + CBAM 消融三组）→ v5×1，各 60ep 串行，
# 末尾自动跑 test 评估并打印 5 组 mAP。
#
# 启动（后台，建议用 nohup，日志见下）：
#   cd /d/2 && nohup bash scripts/run_all_640_adamw.sh > logs/train_640_adamw_all.log 2>&1 &
# 查进度：
#   tail -f D:/2/logs/train_640_adamw_all.log
#   grep -E "epochs completed|>>> CBAM injected" D:/2/logs/train_640_adamw_all.log
#
# 口径：AdamW + lr0=0.001 + seed 42 + workers 0 + imgsz 640 + batch 8，
#       与 640 SGD 那批逐一对应，仅优化器/学习率不同。
# 预计耗时：单组约 72 分钟（640 SGD 实测均值），5 组约 6 小时。
# 注意：勿在运行时编辑本文件（bash 按字节偏移边读边执行）。

D:/yolo-env/Scripts/python.exe D:/2/scripts/fix_settings.py   # settings 被重置时自动恢复

set -u

# ---------- v11 四组（baseline + CBAM 消融） ----------
cd /d/2/code/yolov11 || exit 1
echo "=========== [1/5] v11_640 baseline AdamW ==========="
D:/yolo-env/Scripts/python.exe train_v11_640_adamw.py
echo "=========== [2/5] v11_640 backbone AdamW ==========="
D:/yolo-env/Scripts/python.exe train_v11_640_backbone_adamw.py
echo "=========== [3/5] v11_640 neck AdamW ==========="
D:/yolo-env/Scripts/python.exe train_v11_640_neck_adamw.py
echo "=========== [4/5] v11_640 both AdamW ==========="
D:/yolo-env/Scripts/python.exe train_v11_640_both_adamw.py

# ---------- v5 一组 ----------
cd /d/2/code/baselines/yolov5 || exit 1
echo "=========== [5/5] v5_640 AdamW ==========="
D:/2/envs/yolov5-env/Scripts/python.exe train_v5_640_adamw.py

# ---------- 跑完自动 test 评估 ----------
echo "=========== 640 AdamW test 评估（v11×4）==========="
D:/yolo-env/Scripts/python.exe /d/2/code/tools/eval/eval_test_640_adamw.py

cd /d/2/code/baselines/yolov5 || exit 1
echo "=========== 640 AdamW test 评估（v5）==========="
D:/2/envs/yolov5-env/Scripts/python.exe eval_test_v5_640_adamw.py

echo "=========== 640 + AdamW 全套 完成 ==========="
