#!/bin/bash
# 修复臂 @20 轮 + SA 探针（与原版@20 配对）
#
# 触发条件已满足：原版@20 的 |z| 为 层4 = 2.947、层6 = 3.476（均 > 2.0 失败区），
# 层2 = 0.383。⇒ 20 轮窗口**在层 4/6 上具备区分力**，配对比较成立。
#
# 判据（层 4/6）：修复@20 的 |z| 是否明显低于原版@20（低 50% 以上）？
#   是 → 修复有持续性，再跑完整 60 轮
#   否 → 修复只是延后，省下 40 轮
#
# ⚠ 已知局限（必须在报告里写明）：**层 2 在 20 轮窗口内不具区分力**。
#   原版层 2 走的是"晚长"路径：0.568(@0) → 0.227(@5) → 0.383(@20) → **4.94(@60)**，
#   真正的增长发生在 @20 之后。故即便修复@20 的层 2 很小，也**不能**据此判层 2 已修好
#   —— 层 2 的结论要等 60 轮（即边界 B 在层 2 上单独成立）。
#
# 日志：logs/train_e20_fixed.log
D:/yolo-env/Scripts/python.exe D:/2/scripts/fix_settings.py

echo "=========== 修复臂 @20 轮（wire_v11_640_cbamfix_probe20）==========="
cd /d/2/code/yolov11
D:/yolo-env/Scripts/python.exe train_v11_640_cbamfix.py \
    --epochs 20 --name wire_v11_640_cbamfix_probe20

echo "=========== 修复@20 SA 探针 ==========="
D:/yolo-env/Scripts/python.exe /d/2/code/tools/analyze_cbam_degradation.py \
    --skip-weights --skip-grad --sa-input \
    --sa-run wire_v11_640_cbamfix_probe20 \
    --out D:/2/logs/cbam_fixed20_probe.txt

echo "=========== 修复@20 完成，等三方（@0/@5/@20 × 两臂）对比 ==========="
