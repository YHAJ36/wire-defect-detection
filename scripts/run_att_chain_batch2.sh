#!/usr/bin/env bash
# =============================================================================
# 第二批链：896（地板 + CBAM+BN + Triplet，种子 42/43/44）+ 补齐 640 的种子
#   共 11 个 run，单卡串行；预估 896 78.5 s/epoch、640 60 s/epoch ⇒ ~13.5h
#
# 设计目标（用户 2026-10-15 要求："每跑完一次都先记录，防止崩溃后全部重跑"）
#   1) **可重入**：随时重跑本脚本，已完成的 run 直接跳过，不会重复训练；
#   2) **崩溃续训**：有 last.pt 但未跑满 ⇒ 用 --resume 接上（不是从头）；
#   3) **每 run 落账**：训练完立刻 ① 核注入参数量 ② 评 val+test ③ 追加 ledger/state；
#      于是任何时刻中断，已完成的 run 的数字都已落盘，只需续未完成的；
#   4) **不删任何东西**：只追加。重跑同一 run 用 ultralytics 的 exist_ok。
#
# 产物：
#   exp_upscale/runs/<run>/                训练
#   exp_upscale/eval/<run>_<split>/        评估（独立目录，不混进训练目录）
#   exp_upscale/logs/batch2_ledger.md      人读账本（每 run 一节）
#   exp_upscale/logs/batch2_state.tsv      机器读账本（每 run 一行）
#   exp_upscale/logs/batch2_eval.tsv       评估机器读表（可重入判读用）
#   exp_upscale/logs/chain_batch2_*.log    每个 run 的原始训练日志
#
# 用法：
#   bash D:/2/scripts/run_att_chain_batch2.sh            # 正式跑
#   DRY=1 bash D:/2/scripts/run_att_chain_batch2.sh      # 只打印计划与判定，不训练
# =============================================================================
set -u

PY=D:/yolo-env/Scripts/python.exe
TRAIN=D:/2/code/yolov11/train_att_640.py
EVAL=D:/2/code/tools/eval/eval_att_batch2.py
GATE=D:/2/code/tools/check_run_params.py
RUNS=D:/2/exp_upscale/runs
LOGS=D:/2/exp_upscale/logs
LEDGER=$LOGS/batch2_ledger.md
STATE=$LOGS/batch2_state.tsv
EPOCHS=60
FREE_MIB=3000          # 桌面常占 ~1.25G，阈值不能低于此（A.9.16 踩过 1000 的坑）
DRY=${DRY:-0}

# run|att|imgsz|seed   —— 896 先跑（主线），640 补种子在后
JOBS=(
  "nf_v11_896_s42|baseline|896|42"
  "att_v11_896_cbam_bn_s42|cbam_bn|896|42"
  "nf_v11_896_s43|baseline|896|43"
  "nf_v11_896_s44|baseline|896|44"
  "att_v11_896_cbam_bn_s43|cbam_bn|896|43"
  "att_v11_896_cbam_bn_s44|cbam_bn|896|44"
  "att_v11_896_triplet_s42|triplet|896|42"
  "att_v11_640_cbam_bn_s43|cbam_bn|640|43"
  "att_v11_640_cbam_bn_s44|cbam_bn|640|44"
  "att_v11_640_triplet_s43|triplet|640|43"
  "att_v11_640_triplet_s44|triplet|640|44"
)

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() { echo "[$(ts)] $*"; }

wait_gpu_free() {
  while :; do
    local u
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')
    if [ -n "$u" ] && [ "$u" -lt "$FREE_MIB" ]; then return 0; fi
    echo "[$(ts)]   GPU 占用 ${u:-?} MiB ≥ ${FREE_MIB}，等 60s"
    sleep 60
  done
}

# 训练完成的判据：results.csv 数据行 ≥ EPOCHS 且 best.pt 在
run_done() {
  local csv="$RUNS/$1/results.csv"
  [ -f "$csv" ] || return 1
  local n
  n=$(awk 'END{print NR-1}' "$csv" 2>/dev/null || echo 0)
  [ "$n" -ge "$EPOCHS" ] 2>/dev/null && [ -f "$RUNS/$1/weights/best.pt" ]
}

run_epochs() {
  local csv="$RUNS/$1/results.csv"
  [ -f "$csv" ] && awk 'END{print NR-1}' "$csv" || echo 0
}

# 账本一行：run att imgsz seed status params_ok test_map50 val_map50 epochs ts
record_state() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "${7:-}" "${8:-}" "${9:-}" "$(ts)" >> "$STATE"
}

record_ledger() {   # $1=run $2=title $3=body
  {
    echo ""
    echo "## $1 — $2   （$(ts)）"
    echo ""
    echo '```'
    echo "$3"
    echo '```'
  } >> "$LEDGER"
}

mkdir -p "$LOGS" "$RUNS"
{
  echo ""
  echo "# 第二批链账本（896 主线 + 640 补种子）"
  echo ""
  echo "- 启动：$(ts)"
  echo "- 计划：${#JOBS[@]} 个 run，单卡串行；896 78.5 s/epoch、640 60 s/epoch"
  echo "- 判据：每 run 训练完 ⇒ 注入参数量核验 + val/test 评估（评估 imgsz 按 run 名自识）"
  echo "- 纪律：只追加不删除；可随时重跑本脚本，已完成的 run 自动跳过"
} >> "$LEDGER"

FAIL_EARLY=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r RUN ATT IMGSZ SEED <<< "$job"

  # --- 已是完成态：跳过 ---
  if run_done "$RUN"; then
    log "skip $RUN（已完成 $EPOCHS 轮 + best.pt）"
    record_state "$RUN" "$ATT" "$IMGSZ" "$SEED" SKIP - - - "$(run_epochs "$RUN")"
    continue
  fi

  # --- 定动作 ---
  LAST="$RUNS/$RUN/weights/last.pt"
  if [ -f "$LAST" ]; then
    ACTION=resume
    HAVE=$(run_epochs "$RUN")
    log "== $RUN : 续训（已有 $HAVE/$EPOCHS 轮）=="
  else
    ACTION=fresh
    log "== $RUN : 全新开跑（无 last.pt）=="
  fi

  if [ "$DRY" = "1" ]; then
    log "   [DRY] 将执行：$TRAIN --att $ATT --imgsz $IMGSZ --seed $SEED --name $RUN${ACTION:+ }$( [ "$ACTION" = resume ] && echo --resume )"
    continue
  fi

  RLOG="$LOGS/chain_batch2_${RUN}.log"
  T0=$(date +%s)
  STATUS=OK

  # --- 训练（失败则自动再续训一次） ---
  for try in 1 2; do
    wait_gpu_free
    if [ "$try" = "1" ] && [ "$ACTION" = "resume" ]; then
      "$PY" "$TRAIN" --att "$ATT" --mode neck_sppf --imgsz "$IMGSZ" --seed "$SEED" \
        --name "$RUN" --resume >> "$RLOG" 2>&1
    else
      "$PY" "$TRAIN" --att "$ATT" --mode neck_sppf --imgsz "$IMGSZ" --seed "$SEED" \
        --name "$RUN" >> "$RLOG" 2>&1
    fi
    rc=$?
    if [ "$rc" = "0" ] && run_done "$RUN"; then STATUS=OK; break; fi
    log "   ✗ 第 $try 次未跑满（rc=$rc，当前 $(run_epochs "$RUN") 轮）→ 重试"
    STATUS=RETRY_FAIL
    sleep 20
  done

  T1=$(date +%s)
  EP=$(run_epochs "$RUN")
  log "   训练结束：$EP/$EPOCHS 轮，状态 $STATUS，用时 $(( (T1-T0)/60 )) min"

  if ! run_done "$RUN"; then
    log "   ❌ $RUN 未跑满，账本记为 FAIL，继续下一个（可事后重跑本脚本补上）"
    record_state "$RUN" "$ATT" "$IMGSZ" "$SEED" FAIL - - - "$EP"
    record_ledger "$RUN" "FAIL（$EP/$EPOCHS 轮）" "$(tail -25 "$RLOG" | tr '\r' '\n' | tail -15)"
    continue
  fi

  # --- 闸门：注入参数量 ---
  wait_gpu_free
  GOUT=$("$PY" "$GATE" "$RUN" 2>&1); grc=$?
  if [ "$grc" = "0" ]; then POK=1; else POK=0; fi

  # --- 评估：val + test（imgsz 按 run 名自识） ---
  wait_gpu_free
  EOUT=$("$PY" "$EVAL" "$RUN" 2>&1) || EOUT="$EOUT (eval 非零退出)"
  TM=$(echo "$EOUT" | awk '/ test /{print $NF}' | tail -1)
  VM=$(echo "$EOUT" | awk '/  val /{print $NF}' | tail -1)

  record_state "$RUN" "$ATT" "$IMGSZ" "$SEED" "$STATUS" "$POK" \
    "$(echo "$EOUT" | grep -E '^(att_|nf_)' | awk '$2=="test"{print $4}' | tail -1)" \
    "$(echo "$EOUT" | grep -E '^(att_|nf_)' | awk '$2=="val"{print $4}' | tail -1)" "$EP"
  record_ledger "$RUN" "$ATT @$IMGSZ seed=$SEED  $STATUS" \
"$GOUT
---
$EOUT"

  log "   注入核验：$([ "$POK" = 1 ] && echo OK || echo '★ 不符')"

  if [ "$POK" = "0" ] && [ "$FAIL_EARLY" = "0" ]; then
    log "   ⚠ 第一个 run 的注入核验就不符 ⇒ 判为系统性问题，**停止整批**，避免再废 ~13h"
    record_ledger "整批中止" "注入闸门在第一个 run 即失败" "$GOUT"
    FAIL_EARLY=1
    break
  fi
done

log "==== 链结束（$(ts)）===="
if [ "$DRY" != "1" ]; then
  # 补评全清单：第一批已训好的 640 run（地板 s42/43/44 + 两条臂 s42）也补进 TSV，
  # 否则判读会把 640 误报成"样本不足"。可重入，已评过的自动跳过。
  wait_gpu_free
  log ">>> 补评全清单（含第一批已训好的 640 run）"
  "$PY" "$EVAL" >> "$LOGS/eval_att_batch2.txt" 2>&1
  wait_gpu_free
  log ">>> 生成判读汇总"
  "$PY" "$EVAL" --summary >> "$LOGS/eval_att_batch2.txt" 2>&1
  log ">>> 判读已追加到 $LOGS/eval_att_batch2.txt"
fi
