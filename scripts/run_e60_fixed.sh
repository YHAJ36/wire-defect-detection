#!/bin/bash
# 修复臂完整 60 轮 + SA 探针 + bias 轨迹（第四步）
#
# 触发条件已满足（预钉判据）：修复@20 的 |z| 比原版@20 低 73%（层4）/ 85%（层6），
# 门槛是 50% ⇒ 修复有持续性，跑完整 60 轮。
#
# 只跑修复臂：**原版@60 已经存在**（runs/wire_v11_640_both，60 轮完整跑过），
# 直接拿它做 @60 的对照，省掉 65 分钟。
#
# 这趟要回答的问题（见 WORKLOG / archive/2026-10-12.md）：
#   已知修复版 DC ≡ SA conv 的 bias。所以本趟 = 检验 **bias 会不会像原版 DC 那样饱和**。
#     原版 DC 的饱和行为：层4 2.95(@20) → 3.15(@60)、层6 3.48 → 4.41，@20 后基本停住。
#     bias 若同样停住 → 修复在 60 轮尺度上成立（但按边界 C 仍要看是否 < 0.5）
#     bias 若继续长   → 修复只是延后，边界 B 成立
#
# 🔑 中间权重：save_period=20 已在 train_v11_640_cbamfix.py 里设为默认，
#    故会落 epoch20.pt / epoch40.pt。这**同时修掉了 20 轮探针的口径缺口** ——
#    20 轮探针跑的是压缩过的余弦 LR 调度（20 轮内走完 lr0→lrf），
#    其 epoch 20 与 60 轮调度下的 epoch 20 不是同一个点；
#    本趟的 epoch20.pt 才是"60 轮调度下真正的 @20"，可与原版@20 直接对齐。
#
# 日志：logs/train_e60_fixed.log
D:/yolo-env/Scripts/python.exe D:/2/scripts/fix_settings.py

echo "=========== 修复臂 60 轮（wire_v11_640_cbamfix）==========="
cd /d/2/code/yolov11
D:/yolo-env/Scripts/python.exe train_v11_640_cbamfix.py \
    --epochs 60 --name wire_v11_640_cbamfix

echo "=========== 修复@60 SA 探针 ==========="
D:/yolo-env/Scripts/python.exe /d/2/code/tools/analyze_cbam_degradation.py \
    --skip-weights --skip-grad --sa-input \
    --sa-run wire_v11_640_cbamfix \
    --out D:/2/logs/cbam_fixed60_probe.txt

echo "=========== bias 轨迹（@0/@20 探针/@60 + 本趟的 ep20/ep40）==========="
D:/yolo-env/Scripts/python.exe D:/2/code/tools/check_sa_bias.py \
    --ckpt ep20=D:/2/runs/wire_v11_640_cbamfix/weights/epoch20.pt \
    --ckpt ep40=D:/2/runs/wire_v11_640_cbamfix/weights/epoch40.pt \
    2>&1 | tee D:/2/logs/sa_bias_trajectory.txt

echo "=========== 修复@60 完成，等 边界B/边界C 判读 ==========="
