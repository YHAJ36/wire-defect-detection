#!/bin/bash
# 896 全套 + cbamfix 5 轮短训探针，串行一条链（单卡 8188 MiB，禁止并发训练）。
#
# 链路：
#   1) v11 baseline 896  60ep  ≈ 1.7 h   （smoke 实测 101.7 s/epoch）
#   2) cbamfix 5ep 探针 + SA 探针      ≈ 7 min  （插在两组之间的空档，不与 896 抢卡）
#   3) v5 baseline 896   60ep  ≈ 1.6 h   （smoke 实测 ~90 s/epoch）
#   合计 ≈ 3.4 h
#
# 批次口径：smoke 实测 896 下 v11 用 3.84 G、v5 用 3.25 G 显存，
# **batch 8 不 OOM**，故 896 组与 640 组同为 batch 8，口径一致（未触发降 batch 4 的预案）。
#
# 注意：探针用 --name wire_v11_640_cbamfix_probe 单独建目录，
# 以免 5 轮产物污染将来可能跑的 60 轮正式版（默认名 wire_v11_640_cbamfix）。
# 探针的 SA 结果写到 logs/cbam_sa_input_wire_v11_640_cbamfix_probe.txt，不覆盖基线证据。
#
# 日志：logs/train_896.log
D:/yolo-env/Scripts/python.exe D:/2/scripts/fix_settings.py   # settings 被重置时自动恢复

echo "=========== [1/3] v11 baseline 896 (60ep) ==========="
cd /d/2/code/yolov11
D:/yolo-env/Scripts/python.exe train_v11_896.py

echo "=========== [2/3] cbamfix 5ep 探针（空档期）==========="
cd /d/2/code/yolov11
D:/yolo-env/Scripts/python.exe train_v11_640_cbamfix.py --epochs 5 --name wire_v11_640_cbamfix_probe
D:/yolo-env/Scripts/python.exe /d/2/code/tools/analyze_cbam_degradation.py \
    --skip-weights --skip-grad --sa-input \
    --sa-run wire_v11_640_cbamfix_probe \
    --out D:/2/logs/cbam_fixed_probe.txt

echo "=========== [3/3] v5 baseline 896 (60ep) ==========="
cd /d/2/code/baselines/yolov5
D:/2/envs/yolov5-env/Scripts/python.exe train_v5_896.py

echo "=========== 896 全套 + cbamfix 探针 完成 ==========="
