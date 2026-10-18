#!/usr/bin/env bash
# =============================================================================
# 第三批链：YOLO26s @896 补种子（s43/s44）+ 896 Triplet 补种子（s43/s44）
#   共 4 个 run，单卡串行；YOLO26 ~1.5h/run、Triplet ~1.3h/run ⇒ 约 5.6h
#
# 起点：`nf_yolo26_896_s42` 已由**另一个后台任务**在跑（本脚本**不接管、不重跑**）。
#       本脚本先等它跑完 → 补评它 → 再依次跑后面四个。
#
# 设计目标（沿用 run_att_chain_batch2.sh，用户要求"崩溃后不能全部重跑"）
#   1) **可重入**：随时重跑本脚本，已完成的 run 直接跳过，不会重复训练；
#   2) **崩溃续训**：有 last.pt 但未跑满 ⇒ 用 --resume 接上（不是从头）；
#   3) **每 run 落账**：训练完立刻 ① 闸门核验 ② 评 val+test ③ 追加 ledger/state；
#   4) **不删任何东西**：只追加。
#
# ★ 两代用**两个解释器**，这不是笔误（见 WORKLOG A.9.22）
#     yolo26  → D:/yolo26-env  (ultralytics 8.4.178，认得 YOLO26)
#     triplet → D:/yolo-env    (ultralytics 8.3.180，与 batch2 同版本 ⇒ 同口径)
#   故 **YOLO26 的数字与 v11 的数字来自不同版本的评测库**：差值可报，归因不可报。
#
# 产物：
#   exp_upscale/runs/<run>/                  训练
#   exp_upscale/eval/<run>_<split>/          评估（独立目录，不混进训练目录）
#   exp_upscale/logs/batch3_ledger.md        人读账本
#   exp_upscale/logs/batch3_state.tsv        机器读账本
#   exp_upscale/logs/batch3_eval.tsv         YOLO26 评估表（新环境产出）
#   exp_upscale/logs/batch2_eval.tsv         Triplet 评估表（旧环境产出，续写 batch2 的同一文件）
#   exp_upscale/logs/eval_yolo26_896.txt     YOLO26 判读输出
#   exp_upscale/logs/eval_att_batch2.txt     Triplet 判读输出（续写）
#   exp_upscale/logs/chain_batch3_<run>.log  每 run 原始训练日志
#   exp_upscale/logs/chain_batch3.pid        本链 PID（外部可用于判断是否还在跑）
#
# 用法：
#   bash D:/2/scripts/run_batch3.sh            # 正式跑
#   DRY=1 bash D:/2/scripts/run_batch3.sh      # 只打印计划与判定，不训练不评估
# =============================================================================
set -u

PY26=D:/yolo26-env/Scripts/python.exe     # 新环境（8.4.178）
PY11=D:/yolo-env/Scripts/python.exe       # 旧环境（8.3.180，与 batch2 同版本）
T26=D:/2/code/yolov11/train_yolo26_896.py
T11=D:/2/code/yolov11/train_att_640.py
E26=D:/2/code/tools/eval/eval_yolo26_896.py
E11=D:/2/code/tools/eval/eval_att_batch2.py
GATE=D:/2/code/tools/check_run_params.py
RUNS=D:/2/exp_upscale/runs
LOGS=D:/2/exp_upscale/logs
LEDGER=$LOGS/batch3_ledger.md
STATE=$LOGS/batch3_state.tsv
EPOCHS=60
FREE_MIB=3000          # 桌面常占 ~1.25G；训练中 >4.7G。3000 能挡住"另一个 run 还在跑"
DRY=${DRY:-0}

S42=nf_yolo26_896_s42

# run|kind|seed    —— kind ∈ {yolo26, triplet}
JOBS=(
  "nf_yolo26_896_s43|yolo26|43"
  "nf_yolo26_896_s44|yolo26|44"
  "att_v11_896_triplet_s43|triplet|43"
  "att_v11_896_triplet_s44|triplet|44"
)

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

wait_gpu_free() {
  while :; do
    local u
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')
    if [ -n "$u" ] && [ "$u" -lt "$FREE_MIB" ]; then return 0; fi
    echo "[$(ts)]   GPU 占用 ${u:-?} MiB ≥ ${FREE_MIB}，等 60s（可能有另一个 run 在跑）"
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

# 等某个 run 跑满。$1=run $2=max_minutes(默认240) $3=stall_minutes(默认30)
wait_run_done() {
  local run=$1 max=${2:-240} stall_max=${3:-30} w=0 last=-1 stall=0 ep
  while :; do
    run_done "$run" && { log "  $run 已跑满 $EPOCHS 轮"; return 0; }
    ep=$(run_epochs "$run")
    if [ "$ep" = "$last" ]; then stall=$((stall + 1)); else stall=0; last=$ep; fi
    if [ $((w % 10)) -eq 0 ]; then log "  等 $run：$ep/$EPOCHS 轮（已等 ${w}min）"; fi
    if [ "$stall" -ge "$stall_max" ]; then
      log "  ★ $run 连续 ${stall_max}min 无新轮次（疑似卡住），停止等待"; return 1
    fi
    if [ "$w" -ge "$max" ]; then
      log "  ★ $run 等待超过 ${max}min，停止等待"; return 1
    fi
    sleep 60; w=$((w + 1))
  done
}

# 账本一行：run kind seed status gate test_map50 val_map50 epochs ts
record_state() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$3" "$4" "$5" "${6:-}" "${7:-}" "${8:-}" "$(ts)" >> "$STATE"
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

# 从评估输出里抠某个 split 的 mAP50。
# 两个 eval 脚本的"逐 run 进度行"格式一致（两个前导空格 + run + split + ... mAP50=）：
#     "  <run>  <split>  imgsz=896  [params=...]  mAP50=0.xxxx  mAP50-95=...  P=...  R=..."
# ★ 必须用"两个前导空格 + run + split"锚定：两代脚本的**汇总表行**是
#   "run  split  imgsz  mAP50值 ..."（**无前导空格、无 mAP50= 字面**），
#   而汇总表里还有"每个 run 一行"的判读行。用 mAP50= 字面 + 前导空格双重锚定，
#   才只命中进度行（batch2 的脚本用 ^(att_|nf_) 锚汇总表、取 $4，是另一套等价做法）。
pick_map50() {   # $1=eval输出 $2=run $3=split
  printf '%s\n' "$1" | grep -E "^ +$2 +$3 " \
    | sed -E 's/.*mAP50=([0-9.]+).*/\1/' | tail -1
}

# 训练 + 闸门 + 评估，落账。$1=run $2=kind $3=seed
do_run() {
  local RUN=$1 KIND=$2 SEED=$3
  local TRAIN_PY TRAIN RLOG EOUT GOUT POK EP STATUS
  local -a EXTRA

  case "$KIND" in
    yolo26)
      TRAIN_PY=$PY26; TRAIN=$T26
      EXTRA=(--seed "$SEED" --name "$RUN")
      ;;
    triplet)
      TRAIN_PY=$PY11; TRAIN=$T11
      EXTRA=(--att triplet --mode neck_sppf --imgsz 896 --seed "$SEED" --name "$RUN")
      ;;
    *) log "   ★ 未知 kind：$KIND"; return 1 ;;
  esac

  # --- 已是完成态：跳过训练，但仍补评（可能上次评估前崩了） ---
  if run_done "$RUN"; then
    log "  skip 训练：$RUN 已完成 $EPOCHS 轮 + best.pt"
  else
    local ACTION=fresh
    if [ -f "$RUNS/$RUN/weights/last.pt" ]; then
      ACTION=resume
      log "  $RUN：续训（已有 $(run_epochs "$RUN")/$EPOCHS 轮）"
    else
      log "  $RUN：全新开跑"
    fi

    if [ "$DRY" = "1" ]; then
      log "     [DRY] $TRAIN_PY $TRAIN ${EXTRA[*]} $([ "$ACTION" = resume ] && echo --resume)"
      return 0
    fi

    RLOG="$LOGS/chain_batch3_${RUN}.log"
    local T0 T1
    T0=$(date +%s); STATUS=OK
    for try in 1 2; do
      wait_gpu_free
      if [ "$try" = "1" ] && [ "$ACTION" = "resume" ]; then
        "$TRAIN_PY" "$TRAIN" "${EXTRA[@]}" --resume >> "$RLOG" 2>&1
      else
        "$TRAIN_PY" "$TRAIN" "${EXTRA[@]}" >> "$RLOG" 2>&1
      fi
      rc=$?
      if [ "$rc" = "0" ] && run_done "$RUN"; then STATUS=OK; break; fi
      log "     !! 第 $try 次未跑满（rc=$rc，当前 $(run_epochs "$RUN") 轮）→ 重试"
      STATUS=RETRY_FAIL
      sleep 20
    done
    T1=$(date +%s)
    EP=$(run_epochs "$RUN")
    log "     训练结束：$EP/$EPOCHS 轮，状态 $STATUS，用时 $(( (T1 - T0) / 60 )) min"

    if ! run_done "$RUN"; then
      log "     !! $RUN 未跑满，账本记 FAIL，继续下一个（可事后重跑本脚本补）"
      record_state "$RUN" "$KIND" "$SEED" FAIL - - - "$EP"
      record_ledger "$RUN" "FAIL（$EP/$EPOCHS 轮）" \
        "$(tail -25 "$RLOG" 2>/dev/null | tr '\r' '\n' | tail -15)"
      return 1
    fi
  fi

  if [ "$DRY" = "1" ]; then return 0; fi

  # --- 闸门 ---
  # triplet 有注入 ⇒ 必须核参数量（防"注入被静默丢弃、跑成基线"）。
  # yolo26 无注入 ⇒ 用"评估能加载 best.pt"作等价闸门（加载失败即 eval 非零）。
  if [ "$KIND" = "triplet" ]; then
    wait_gpu_free
    GOUT=$("$PY11" "$GATE" "$RUN" 2>&1); grc=$?
    [ "$grc" = "0" ] && POK=1 || POK=0
    log "     注入核验：$([ "$POK" = 1 ] && echo OK || echo '★ 不符')"
  else
    GOUT="（YOLO26 无注入层，跳过量具闸门）"; POK=NA
  fi

  # --- 评估 val + test ---
  wait_gpu_free
  case "$KIND" in
    yolo26)  EOUT=$("$PY26" "$E26" "$RUN" 2>&1) || EOUT="$EOUT (eval 非零退出)" ;;
    triplet) EOUT=$("$PY11" "$E11" "$RUN" 2>&1) || EOUT="$EOUT (eval 非零退出)" ;;
  esac

  local TM VM
  TM=$(pick_map50 "$EOUT" "$RUN" test)
  VM=$(pick_map50 "$EOUT" "$RUN" val)
  record_state "$RUN" "$KIND" "$SEED" OK "$POK" "$TM" "$VM" "$(run_epochs "$RUN")"
  record_ledger "$RUN" "$KIND seed=$SEED OK" "$GOUT
---
$EOUT"

  log "     test mAP50 = ${TM:-?}   val mAP50 = ${VM:-?}"

  if [ "$KIND" = "triplet" ] && [ "$POK" = "0" ] && [ "$FAIL_EARLY" = "0" ]; then
    log "     ⚠ 注入核验不符 ⇒ 判为系统性问题，停止整批（避免再废数小时）"
    record_ledger "整批中止" "注入闸门失败" "$GOUT"
    FAIL_EARLY=1
  fi
  return 0
}

mkdir -p "$LOGS" "$RUNS"
echo $$ > "$LOGS/chain_batch3.pid"
{
  echo ""
  echo "# 第三批链账本（YOLO26s@896 + 896 Triplet 补种子）"
  echo ""
  echo "- 启动：$(ts)   PID $(cat "$LOGS/chain_batch3.pid")"
  echo "- 计划：先等 $S42 跑完并补评，再串行 ${#JOBS[@]} 个 run"
  echo "- 解释器：yolo26 → $PY26（8.4.178）；triplet → $PY11（8.3.180）"
  echo "- 判据：每 run 训练完 ⇒ 闸门 + val/test 评估；test 为主判据，val 为一致性检查"
  echo "- 纪律：只追加不删除；可随时重跑本脚本，已完成的 run 自动跳过"
} >> "$LEDGER"

FAIL_EARLY=0

log "==== 第三批链启动（PID $$）===="
log ">>> 第 0 棒：等 $S42（由先前的后台任务持有）"
if [ "$DRY" = "1" ]; then
  log "   [DRY] 将等 $S42 跑满 $EPOCHS 轮，然后评它"
elif run_done "$S42"; then
  log "   $S42 已跑满"
  do_run "$S42" yolo26 42
else
  if wait_run_done "$S42" 240 30; then
    do_run "$S42" yolo26 42
  else
    # 等不到（可能它那个后台任务随会话结束被杀了）。交给 do_run：
    # 它会看到 last.pt ⇒ 从断点续训，而不是从头重跑，也不是丢掉这个种子。
    # 安全性：若旧进程其实还活着，do_run 里的 wait_gpu_free 会一直挡住
    #（训练占 >4.7G ≥ 阈值 3000），故**不会**出现同一 run 双进程抢卡。
    log "   !! $S42 未在等待窗内跑满 ⇒ 改由本链接手续训（有 last.pt 则断点续训）"
    do_run "$S42" yolo26 42
  fi
fi

for job in "${JOBS[@]}"; do
  IFS='|' read -r RUN KIND SEED <<< "$job"
  log "== $RUN （$KIND, seed=$SEED）=="
  do_run "$RUN" "$KIND" "$SEED"
done

log "==== 链结束（$(ts)）===="

if [ "$DRY" != "1" ]; then
  # Triplet 三个种子到齐 ⇒ 让 batch2 的判读脚本重算 896 Triplet 臂区间（n=1 → n=3）
  wait_gpu_free
  log ">>> 重算 batch2 判读汇总（Triplet@896 臂将由 n=1 变为 n=3）"
  # 【2026-10-17 修复·日志双写】原先这里写 >> "$LOGS/eval_att_batch2.txt" 2>&1。
  # 缺陷有两层：① E11/E26 自己就把同一份输出追加到该文件（脚本内 LOG.open('a')），
  # 于是每条判读被写两遍；② 2>&1 把 **stderr** 也并进记录文件，而脚本只对
  # sys.stdout 做了 reconfigure(encoding='utf-8')，stderr 仍是 GBK ——
  # GBK 字节混入 UTF-8 文件 ⇒ eval_yolo26_896.txt 里出现非法字节 0x9b。
  # 故判读仍由脚本自记，链级 stdout/stderr 另行收到独立文件，不再污染记录。
  "$PY11" "$E11" --summary >> "$LOGS/chain_summary_stdout.log" 2>&1
  wait_gpu_free
  log ">>> 重算 YOLO26 判读汇总"
  "$PY26" "$E26" >> "$LOGS/chain_summary_stdout.log" 2>&1
  log ">>> 判读已追加到 $LOGS/eval_att_batch2.txt 与 $LOGS/eval_yolo26_896.txt"
fi
