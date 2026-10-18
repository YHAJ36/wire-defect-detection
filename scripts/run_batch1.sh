#!/bin/bash
# ============================================================================
# 第一批 v11 提升实验 · 链式脚本：c1 → c2 → c3 → c4 → c5 串行 + 每组落盘
#
# 依据：用户 2026-10-14 裁定（WORKLOG A.9.7）
#   · 顺序 1→6 串行（单卡不可并行）
#   · 跑完一组立即 **val 选型 + test 一次**，产物落 exp_upscale/
#   · **第一批跑到第 5 组即停下看结果** ⇒ 本脚本到 c5 为止，**不含 c6（Triplet）**
#
# 组表
#   c1  wire_v11_1024         v11s 1024 b8              分辨率余量
#   c2  wire_v11m_896_b4      v11m 896 b4 (nbs=64)      容量瓶颈（⚠ BN micro-batch 4≠8）
#   c3  wire_v11_1024_aug     v11s 1024 b8 + 增强上调    增强
#   c4  wire_v11_1024_cpm     v11s 1024 b8 + wear cp     手动 copy-paste（自建数据集）
#   c5  wire_v11_896_adamw    v11s 896 b8 AdamW lr0=1e-3 优化器排他项
#
# 三道设计闸门
#   1. **完成判据 = results.csv 行数 ≥ 60**，不是"best.pt 存在"。
#      ultralytics 第 1 轮就会写出 best.pt，用后者判会把只训了几轮的 run 误判为已完成。
#   2. **被中断则从 last.pt 续训**，不从头重跑（不丢已训轮次）。
#   3. **训练前等 GPU 空闲**（nvidia-smi 显存 < 1000 MiB）——
#      这样本脚本可以与"已在后台跑的 c1"共存：它会等 c1 训完再接上，而不是抢卡。
#
# 可断点续跑：随时 Ctrl-C / 重启，已完成的组会跳过训练直接评估。
#
# 日志（本脚本自身的输出全在这里，与 nohup 的重定向无关）
#   exp_upscale/logs/batch1_chain.log      链总日志
#   exp_upscale/logs/train_<cfg>_<run>.log 各组训练日志
#   exp_upscale/logs/eval_batch1.txt       追加式评估记录（含探索口径声明）
# ============================================================================
set -u
cd /d/2 || { echo "无法进入 D:/2"; exit 1; }

PY=D:/yolo-env/Scripts/python.exe
LOGD=exp_upscale/logs
RUNS=exp_upscale/runs
EPOCHS=60
M_WEIGHT=D:/1/models/yolo11m.pt

mkdir -p "$LOGD"
# 自检模式不重定向，直接在终端看；正式跑才把全部输出收进链日志
if [ "${1:-}" != "--dry-run" ]; then
  exec >> "$LOGD/batch1_chain.log" 2>&1
fi

say() { echo "[$(date '+%F %T')] $*"; }

# ---------- GPU 占用（MiB）----------
gpu_used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr -dc '0-9'
}

# ---------- 等 GPU 空闲：$1 = 最多等多少秒 ----------
wait_gpu_free() {
  local limit=${1:-21600} waited=0 u
  while :; do
    u=$(gpu_used)
    if [ -n "$u" ] && [ "$u" -lt 1000 ]; then
      [ "$waited" -gt 0 ] && say "    GPU 已空闲（等了 ${waited}s）"
      return 0
    fi
    sleep 30; waited=$((waited + 30))
    if [ "$waited" -ge "$limit" ]; then
      say "    ⚠ 等待 GPU 超时（${limit}s），仍被占用 ${u:-?} MiB，直接继续"
      return 1
    fi
    [ $((waited % 600)) -eq 0 ] && say "    仍在等 GPU… 当前占用 ${u:-?} MiB"
  done
}

# ---------- 已完成 epoch 数（results.csv 扣掉表头）----------
epochs_done() {
  local f="$RUNS/$1/results.csv" n
  [ -f "$f" ] || { echo 0; return; }
  n=$(wc -l < "$f")
  [ "$n" -gt 0 ] && echo $((n - 1)) || echo 0
}

# ---------- c2 权重闸门：必须是"能建出模型"，不是"文件存在"或"是个 zip"
#             （踩过：断点续传产出 10.8MB 半截文件，ts 报 exit 0 但 torch.load 失败）
need_m() {
  [ -f "$M_WEIGHT" ] || return 1
  $PY -c "
import sys
sys.path.insert(0, 'D:/2/code/yolov11')
from ultralytics import YOLO
n = sum(p.numel() for p in YOLO('$M_WEIGHT').model.parameters())
sys.exit(0 if n > 15_000_000 else 1)   # YOLO11m 应约 20.1M
" >/dev/null 2>&1
}

# ---------- 跑一组：训练（按需）+ 评估 ----------
# run_group <cfg> <run名> <imgsz>
run_group() {
  local cfg=$1 run=$2 img=$3 done_n
  say "---------- 组 $cfg ($run, imgsz $img) ----------"

  # ⚠ 顺序关键：**先等卡，再判完成度**。
  # 若先判后等，脚本启动时 c1 正训到第 25 轮就会记住"25"，
  # 等 c1 训满 60 轮后反倒去"续训"一个已完成的 run。（自检时抓到的 bug）
  wait_gpu_free 21600 || true
  done_n=$(epochs_done "$run")

  if [ "$done_n" -ge "$EPOCHS" ]; then
    say "  训练已完成 $done_n/$EPOCHS 轮，跳过训练"
  elif [ "$done_n" -gt 0 ] && [ -f "$RUNS/$run/weights/last.pt" ]; then
    say "  续训：从 $done_n/$EPOCHS 轮接上"
    $PY code/yolov11/train_v11_batch1.py --config "$cfg" --resume \
        > "$LOGD/train_${cfg}_${run}.log" 2>&1
    say "  训练退出码 $?；结束于 $(epochs_done "$run")/$EPOCHS 轮"
  else
    say "  从头训练"
    $PY code/yolov11/train_v11_batch1.py --config "$cfg" \
        > "$LOGD/train_${cfg}_${run}.log" 2>&1
    say "  训练退出码 $?；结束于 $(epochs_done "$run")/$EPOCHS 轮"
  fi

  # 评估要用卡；训练刚结束时卡是空的，但若训练被跳过则未必 ⇒ 再等一次（上限 30 min）
  wait_gpu_free 1800 || true
  if [ -f "$RUNS/$run/weights/best.pt" ]; then
    say "  评估：val 选型复核 + test"
    $PY code/upscale/eval_batch1.py "$run" "$img"
    say "  评估退出码 $?"
  else
    say "  ⚠ 无 best.pt，跳过评估"
  fi
}

# ============================ 自检（--dry-run）============================
# 只打印"当前状态 + 将要做什么"，不训练、不评估、不写任何结果。
if [ "${1:-}" = "--dry-run" ]; then
  echo "==================== 自检（--dry-run，不做任何训练/评估）===================="
  echo "工作目录      : $(pwd)"
  echo "解释器        : $PY  $([ -x "$PY" ] && echo '✅' || echo '❌ 不存在')"
  echo "GPU 当前占用  : $(gpu_used) MiB  （<1000 视为空闲）"
  echo
  printf '%-6s %-22s %-6s %-8s %-10s %s\n' 组 run名 imgsz 已完成 训练动作 评估
  for spec in "c1 wire_v11_1024 1024" "c2 wire_v11m_896_b4 896" \
              "c3 wire_v11_1024_aug 1024" "c4 wire_v11_1024_cpm 1024" \
              "c5 wire_v11_896_adamw 896"; do
    set -- $spec
    n=$(epochs_done "$2")
    if [ "$n" -ge "$EPOCHS" ]; then act="跳过（已完成）"
    elif [ "$n" -gt 0 ] && [ -f "$RUNS/$2/weights/last.pt" ]; then act="续训（从 $n 轮）"
    else act="从头训练"; fi
    ev=""
    [ -f "$RUNS/$2/weights/best.pt" ] && ev="best.pt 已存在" || ev="训练后才会有"
    printf '%-6s %-22s %-6s %-8s %-10s %s\n' "$1" "$2" "$3" "$n/$EPOCHS" "$act" "$ev"
  done
  echo '（上表「训练动作」是**此刻的快照**。实际运行时会先等 GPU 空闲、再重判一次完成度——'
  echo '  所以若 c1 此刻正在训练，届时它会显示为「跳过（已完成）」而不是「续训」。）'
  echo
  echo "c2 权重闸门   : $M_WEIGHT"
  if need_m; then echo "                ✅ 可加载成模型（参数量 > 15M）"
  else echo "                ❌ 不可用 —— 跑到 c2 时会等最多 60 min 再决定是否跳过"; fi
  echo
  echo "c4 数据集     : D:/2/dataset_cpm_wear/data_cpm.yaml"
  if [ -f D:/2/dataset_cpm_wear/data_cpm.yaml ]; then
    echo "                ✅ 存在（train $(ls D:/2/dataset_cpm_wear/images/train 2>/dev/null | wc -l) 张）"
  else echo "                ❌ 缺失，需先跑 code/upscale/make_cp_wear.py"; fi
  echo
  echo "本脚本不含 c6(Triplet)；到 c5 停下看结果。"
  echo "==========================================================="
  exit 0
fi

# ============================ 主流程 ============================
say "==================== 第一批流水线启动 ===================="
say "顺序 c1→c5（不含 c6）；组内：训练（按需）→ val 选型 + test → 落盘 exp_upscale/"

# c1：可能正在后台训练 —— wait_gpu_free 会等它训完，再按 results.csv 判是否要续训
run_group c1 wire_v11_1024 1024

# c2：先过权重闸门
if need_m; then
  run_group c2 wire_v11m_896_b4 896
else
  say "yolo11m.pt 不可用，等下载/修复（最多 60 min）…"
  w=0
  while [ "$w" -lt 3600 ]; do
    sleep 60; w=$((w + 60))
    need_m && break
    [ $((w % 300)) -eq 0 ] && say "  仍在等 $M_WEIGHT …"
  done
  if need_m; then
    run_group c2 wire_v11m_896_b4 896
  else
    say "❌ $M_WEIGHT 60 min 内仍不可用，**跳过 c2**（c3~c5 继续）"
  fi
fi

run_group c3 wire_v11_1024_aug  1024
run_group c4 wire_v11_1024_cpm  1024
run_group c5 wire_v11_896_adamw  896

say "==================== 第 1~5 组完成，停下看结果 ===================="
say "c6 (Triplet) 按用户规则视前 5 组结果决定，本脚本不自动执行。"
say "⚠ 所有数字为**探索口径**：单种子、未做显著性检验；只报方向与幅度。"
say "  苗头阈值 ±0.5pt（行动阈值，非统计显著性阈值）。详见 WORKLOG A.9.7。"
