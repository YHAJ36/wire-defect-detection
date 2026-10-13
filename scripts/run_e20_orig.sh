#!/bin/bash
# 原版臂 @20 轮 + SA 探针（决策点：20 轮窗口到底具不具备区分力）
#
# 为什么先只跑原版臂：
#   @5 的实测显示原版的 DC 轨迹非单调 —— 层 2 的 |z| 0.568(@0) → 0.227(@5) → 4.94(@60)，
#   即 @0 的 DC 是随机初始化产物，训练**先洗掉它**，之后才长出新的饱和 DC。
#   @5 时原版还在"洗掉期"，所以 5 轮窗口测在了错误阶段。
#   ⇒ 先单独确认 20 轮时原版进没进"长出来期"：
#       原版@20 |z| 仍 < 1.0  → 20 轮同样不具区分力，直接跳到 60 轮（省掉修复臂 25 分钟）
#       原版@20 已 > 2.0      → 20 轮可用，再跑修复臂@20 做配对
#   总串行耗时与"两臂都跑到 20"相同，但决策点提前且可能省一半。
#
# 命名：wire_v11_640_both_probe20（不写 wire_v11_640_both —— 那是已完成的 60 轮产物）。
# save_period=20 已设为默认，本轮会在 epoch 20 落一次盘（对 20 轮这趟是冗余的，
# last.pt 本身就是 @20；真正的用处是将来的 60 轮那趟）。
#
# 日志：logs/train_e20_orig.log
D:/yolo-env/Scripts/python.exe D:/2/scripts/fix_settings.py

echo "=========== 原版臂 @20 轮（wire_v11_640_both_probe20）==========="
cd /d/2/code/yolov11
D:/yolo-env/Scripts/python.exe train_v11_640.py \
    --mode both --epochs 20 --name wire_v11_640_both_probe20

echo "=========== 原版@20 SA 探针 ==========="
D:/yolo-env/Scripts/python.exe /d/2/code/tools/analyze_cbam_degradation.py \
    --skip-weights --skip-grad --sa-input \
    --sa-run wire_v11_640_both_probe20 \
    --out D:/2/logs/cbam_orig20_probe.txt

echo "=========== 原版@20 完成，等人工判定 ==========="
