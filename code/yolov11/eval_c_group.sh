#!/usr/bin/env bash
# =============================================================================
# C 组（multi_scale）收尾脚本 —— 为「无人值守的一夜」写的
#
# 背景：multi_scale 每轮 5.3 min（A 组同口径 1.53 min），三种子跑不完。
#       已定：只保留 s42 + s43 两个种子，s44 一旦启动就立刻掐掉，把 GPU 让出来。
#
# 做的事（全部确定性，不需要人判断）
#   1. 轮询等 ms_s43 跑满 60 轮（最多等 4 小时）
#   2. 若 ms_s44 已经在跑，按命令行精确匹配（含 ms_s44）杀掉那一个 python 进程
#      —— 不用 taskkill /IM python.exe，避免误杀别的 python
#   3. 给 s42 / s43 生成决策缓存并评估（203 图段级 test，imgsz=896）
#   4. 跑配对 bootstrap（jit 那套逻辑，--target ms --seeds 42,43）
#
# 用法：bash code/yolov11/eval_c_group.sh
#       日志：exp_upscale/logs/c_group_finish.log
# =============================================================================
set -uo pipefail

PY="D:/yolo-env/Scripts/python.exe"
PROJ="D:/2/exp_upscale/runs"
LOGDIR="D:/2/exp_upscale/logs"
LOG="$LOGDIR/perturb_chain.log"
OUT="$LOGDIR/c_group_finish.log"

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$OUT"; }
epochs() { local c="$PROJ/$1/results.csv"; if [ -f "$c" ]; then echo $(( $(wc -l < "$c") - 1 )); else echo 0; fi; }

say "==== C 组收尾开始 ===="
say "ms_s42=$(epochs nf_v11_noleak_ms_s42)/60  ms_s43=$(epochs nf_v11_noleak_ms_s43)/60  ms_s44=$(epochs nf_v11_noleak_ms_s44)/60"

# 1) 等 s43 跑满
for _ in $(seq 1 240); do
  if [ "$(epochs nf_v11_noleak_ms_s43)" -ge 60 ]; then break; fi
  sleep 60
done
say "等待结束：ms_s43=$(epochs nf_v11_noleak_ms_s43)/60"
if [ "$(epochs nf_v11_noleak_ms_s43)" -lt 60 ]; then
  say "⚠ s43 未跑满，仍按已完成部分评估（结果要标注）"
fi

# 2) 掐掉 s44（若已启动）
STARTED=0
if [ "$(epochs nf_v11_noleak_ms_s44)" -gt 0 ]; then STARTED=1; fi
if grep -aq "nf_v11_noleak_ms_s44" "$LOG" 2>/dev/null; then STARTED=1; fi
if [ "$STARTED" -eq 1 ]; then
  PIDS=$(powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*nf_v11_noleak_ms_s44*' } | ForEach-Object { \$_.ProcessId }" 2>/dev/null | tr -d '\r')
  if [ -n "${PIDS:-}" ]; then
    say "终止 ms_s44 的 python 进程：$PIDS"
    powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*nf_v11_noleak_ms_s44*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" 2>/dev/null
    sleep 15
  else
    say "ms_s44 无活动进程（可能尚未进入训练循环）"
  fi
else
  say "ms_s44 尚未启动，无需终止"
fi
say "GPU 现状：$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr -d '\r')"

# 3) 评估 s42 / s43
cd /d/2 || exit 1
say "评估 ms_s42 / ms_s43 ..."
"$PY" code/tools/eval/cache_perturb_tp.py nf_v11_noleak_ms_s42 nf_v11_noleak_ms_s43 >>"$OUT" 2>&1
say "评估结束 rc=$?"

# 4) 配对 bootstrap
say "配对 bootstrap（--target ms --seeds 42,43）..."
"$PY" code/tools/eval/bootstrap_paired_jit.py --B 3000 --target ms --seeds 42,43 >>"$OUT" 2>&1
say "bootstrap 结束 rc=$?"

say "==== 完成，结果在 $LOGDIR/bootstrap_paired_ms.tsv ===="
