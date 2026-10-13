#!/bin/bash
# 原版@5轮配对对照 —— 在 896 链（PID 601）跑完之后自动执行。
#
# 为什么需要它：修复版@5轮的 |z| < 1 有两种解释 ——
#   (a) 修复有效（DC 被结构性地压住了）
#   (b) 5 轮本身还没到饱和期（DC 只是还没长出来）
# 绝对阈值分不开这两种；只有"原版@5轮"这个同轮次基线能分开。
# 配对性已验证：train_v11_640.py 与 train_v11_640_cbamfix.py 的 model.train(...) 块
# **逐字一致**（diff 退出码 0），只差注入的 CBAM 实现。warmup=3/5 两臂同样吃，配对后不构成干扰。
#
# 命名说明：**不写 wire_v11_640_both**（那是已跑完的 60 轮产物，不能覆盖）；
# 用 wire_v11_640_both_probe5 新建目录，原版 cbam.py + mode both + 5 轮。
#
# 等待策略（不依赖日志编码/文件解析）：
#   连续 5 分钟 GPU 显存 < 1500 MiB 视为链已结束。
#   依据：链内部的间隔（v11 收尾→探针启动、探针→v5 启动）实测只有数十秒到约 1 分钟，
#   连续 5 分钟空闲只可能是整条链真的结束了。超时上限 5 小时，超时即退出并留记录。
#
# 日志：logs/train_probe5_control.log
set -u
cd /d/2

PY=D:/yolo-env/Scripts/python.exe
THRESH=1500          # MiB：基线占用约 780，训练中 3200~4000
NEED=5               # 连续多少次 60s 检查都空闲才算结束
DEADLINE=$(( $(date +%s) + 5*3600 ))

echo "=== 守望启动 $(date '+%F %T')，等 896 链结束（连续 $NEED 分钟 GPU 空闲）==="

idle=0
while [ $idle -lt $NEED ]; do
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "!! 等待超时（5h），退出。链可能崩了，需人工查 logs/train_896.log"
        exit 1
    fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    # nvidia-smi 偶发取不到数时按"忙"处理，宁可多等一轮也不误判为结束
    if ! [[ "$used" =~ ^[0-9]+$ ]]; then
        echo "  nvidia-smi 读数异常（'$used'），按忙处理"
        idle=0
        sleep 60
        continue
    fi
    if [ "$used" -lt "$THRESH" ]; then
        idle=$((idle + 1))
        echo "  GPU 空闲 $used MiB  ($idle/$NEED)  $(date '+%T')"
    else
        [ $idle -gt 0 ] && echo "  GPU 又占用了 $used MiB，计数清零"
        idle=0
    fi
    sleep 60
done

echo
echo "=== 896 链已结束，开始补测原版@5轮  $(date '+%F %T') ==="
$PY /d/2/scripts/fix_settings.py

echo "--- [1/2] 训练原版（cbam.py）@5轮，name=wire_v11_640_both_probe5 ---"
$PY /d/2/code/yolov11/train_v11_640.py --mode both --epochs 5 --name wire_v11_640_both_probe5

echo "--- [2/2] 对 wire_v11_640_both_probe5 跑 SA 探针 ---"
$PY /d/2/code/tools/analyze_cbam_degradation.py \
    --skip-weights --skip-grad --sa-input \
    --sa-run wire_v11_640_both_probe5 \
    --out D:/2/logs/cbam_orig5_probe.txt

echo "=== 原版@5轮补测完成  $(date '+%F %T') ==="
