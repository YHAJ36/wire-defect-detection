#!/usr/bin/env bash
# =============================================================================
# 段级无泄漏重训链（2026-10-26）
#
#   stage 0  冒烟：baseline + triplet 各 1 轮 / 3% 数据，验证新 data 路径与注入
#   stage 1  正式：6 个 run 串行（基线 ×3 种子、Triplet ×3 种子）
#   stage 2  评估：新 203 图 test 上逐 run 评估 + 两族三种子 WBF 融合
#
# 数据：D:/2/dataset_noleak（build_noleak_dataset.py 切成，1080 段零跨 split）
# 口径：imgsz=896 / epochs=60 / batch=8 / workers=0 / SGD lr0=0.01 / 其余同
#       code/yolov11/train_att_640.py 的 BASE（即历史 896 各臂的管线）
#
# 用法
#   bash code/yolov11/run_batch_noleak.sh --all       # 冒烟 + 训练 + 评估
#   bash code/yolov11/run_batch_noleak.sh --smoke     # 只冒烟（约 3 分钟）
#   bash code/yolov11/run_batch_noleak.sh --train     # 只训练（约 7.8 小时）
#   bash code/yolov11/run_batch_noleak.sh --eval      # 只评估（约 15 分钟）
#
# 日志：D:/2/exp_upscale/logs/noleak_chain.log（实时追加）
# 中断恢复：单个 run 失败即停，脚本会打印该 run 的 --resume 命令；也可 --train 重跑，
#           已完成的 run（results.csv 轮数 >= epochs）会被跳过；
#           中断的 run 若仍有 last.pt，自动改为 --resume 续训而不是重跑。
#           ⚠ 判据不是「best.pt 存在」——ultralytics 每轮覆写 best.pt，中断的 run 也有它。
# =============================================================================
set -uo pipefail

PY="D:/yolo-env/Scripts/python.exe"
TRAIN="D:/2/code/yolov11/train_att_640.py"
EVAL="D:/2/code/tools/eval/eval_wbf_tta.py"
DATA="D:/2/dataset_noleak/data.yaml"
PROJ="D:/2/exp_upscale/runs"
LOGDIR="D:/2/exp_upscale/logs"
LOG="$LOGDIR/noleak_chain.log"
TSV="$LOGDIR/eval_noleak_final.tsv"

IMGSZ=896
EPOCHS=60

mkdir -p "$LOGDIR"

log()  { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
rule() { log "--------------------------------------------------------------"; }
secs() { printf '%dh%02dm' $(($1 / 3600)) $((($1 % 3600) / 60)); }

# 顺序即执行顺序：先基线三颗（= 新噪声地板），再 Triplet 三颗
NAMES=(
  "baseline:42:nf_v11_noleak_s42"
  "baseline:43:nf_v11_noleak_s43"
  "baseline:44:nf_v11_noleak_s44"
  "triplet:42:att_v11_noleak_triplet_s42"
  "triplet:43:att_v11_noleak_triplet_s43"
  "triplet:44:att_v11_noleak_triplet_s44"
)

# ---------------------------------------------------------------------------
# 预检：数据、脚本、权重、GPU
# ---------------------------------------------------------------------------
preflight() {
  rule
  log "预检"
  local fail=0
  for f in "$DATA" "D:/2/dataset_noleak/images/train" "D:/2/dataset_noleak/images/val" \
           "D:/2/dataset_noleak/images/test" "$TRAIN" "$EVAL" "D:/1/models/yolo11s.pt"; do
    if [ -e "$f" ]; then log "  ✓ $f"; else log "  ✗ 缺失 $f"; fail=1; fi
  done
  for sp in train val test; do
    local i l
    i=$(ls -1 "D:/2/dataset_noleak/images/$sp" | wc -l | tr -d ' ')
    l=$(ls -1 "D:/2/dataset_noleak/labels/$sp" | grep -c '\.txt$')
    log "  $sp: $i 图 / $l 标签"
    [ "$i" = "$l" ] || { log "  ✗ $sp 图标签数不等"; fail=1; }
  done
  log "  数据缓存检查：$(ls D:/2/dataset_noleak/labels/*.cache 2>/dev/null | wc -l | tr -d ' ') 个 .cache（应为 0）"
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "  GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | head -1)"
  fi
  [ $fail -eq 0 ] || { log "✗ 预检失败，终止"; exit 1; }
  log "预检通过"
}

# ---------------------------------------------------------------------------
# stage 0：冒烟
# ---------------------------------------------------------------------------
smoke_one() {
  local att=$1 name=$2
  local t0; t0=$(date +%s)
  log "▶ 冒烟 $name（att=$att, epochs=1, fraction=0.03）"
  "$PY" "$TRAIN" --att "$att" --seed 42 --imgsz "$IMGSZ" --epochs 1 \
      --fraction 0.03 --data "$DATA" --name "$name" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  local dt=$(( $(date +%s) - t0 ))
  if [ $rc -ne 0 ]; then
    log "✗ 冒烟失败 $name  rc=$rc  用时 $(secs $dt)"
    exit $rc
  fi
  # 机器判据：best.pt 存在 + args.yaml 的 data 指向新数据集 + 注入层参数量
  local wp="$PROJ/$name/weights/best.pt"
  local ay="$PROJ/$name/args.yaml"
  if [ -f "$wp" ]; then log "  ✓ weights/best.pt 生成"; else log "  ✗ 无 best.pt（epochs=1 可能只存 last.pt）"; fi
  if [ -f "$ay" ]; then
    grep -E '^(data|imgsz|epochs|workers|fraction):' "$ay" | sed 's/^/    /' | tee -a "$LOG"
    grep -q 'dataset_noleak' "$ay" && log "  ✓ args.yaml 的 data 指向 dataset_noleak" \
                                   || { log "  ✗ args.yaml 的 data 未指向新数据集"; exit 1; }
  fi
  log "✓ 冒烟 $name 通过，用时 $(secs $dt)"
}

stage_smoke() {
  rule
  log "STAGE 0 冒烟测试"
  smoke_one baseline smoke_noleak_base
  smoke_one triplet  smoke_noleak_triplet
  log "STAGE 0 通过：新数据路径与 Triplet 注入均无报错"
}

# ---------------------------------------------------------------------------
# stage 1：6 个 run 串行
# ---------------------------------------------------------------------------
# 真实轮数 = results.csv 行数 - 表头。
# ⚠ 判据绝不能用「best.pt 存在」：ultralytics 每轮都覆写 best.pt，
#   中断在第 3 轮的 run 也照样有 best.pt，用它判断完成会静默跳过未训练的 run。
epochs_done() {
  local csv="$PROJ/$1/results.csv"
  if [ -f "$csv" ]; then
    echo $(( $(wc -l < "$csv") - 1 ))
  else
    echo 0
  fi
}

train_one() {
  local att=$1 seed=$2 name=$3
  local wp="$PROJ/$name/weights/best.pt"
  local got; got=$(epochs_done "$name")

  if [ "$got" -ge "$EPOCHS" ]; then
    log "↷ 跳过 $name（results.csv 已 $got/$EPOCHS 轮，$([ -f "$wp" ] && echo 'best.pt 在位'))"
    return 0
  fi

  local resume_arg=''
  if [ "$got" -gt 0 ] && [ -f "$PROJ/$name/weights/last.pt" ]; then
    log "⚠ $name 上次只跑完 $got/$EPOCHS 轮，改为从 last.pt 续训（不重跑）"
    resume_arg='--resume'
  elif [ "$got" -gt 0 ]; then
    log "⚠ $name 有 $got 轮记录但无 last.pt，只能从头重训"
  fi

  local t0; t0=$(date +%s)
  log "▶ 开始 $name（att=$att seed=$seed imgsz=$IMGSZ epochs=$EPOCHS${resume_arg:+ resume}）"
  "$PY" "$TRAIN" --att "$att" --seed "$seed" --imgsz "$IMGSZ" --epochs "$EPOCHS" \
      --data "$DATA" --name "$name" $resume_arg 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  local dt=$(( $(date +%s) - t0 ))
  local fin; fin=$(epochs_done "$name")
  if [ $rc -ne 0 ] || [ "$fin" -lt "$EPOCHS" ]; then
    log "✗ $name 失败 rc=$rc，轮数 $fin/$EPOCHS，用时 $(secs $dt)"
    log "  续训：$PY $TRAIN --att $att --seed $seed --imgsz $IMGSZ --data $DATA --name $name --resume"
    exit 1
  fi
  log "✓ $name 完成（$fin/$EPOCHS 轮），用时 $(secs $dt)；best.pt $(du -h "$wp" | cut -f1)"
}

stage_train() {
  rule
  log "STAGE 1 正式训练（6 个 run 串行）"
  # 冒烟（fraction=0.03）会留下只含 42 张的 train.cache。ultralytics 的 get_labels 里有
  # 硬 hash 校验（cache["hash"] != get_hash(label_files+im_files) ⇒ 重建），理论上不会误用，
  # 但删掉能让日志从「Scanning ... 1401 images」开始，留下可查的证据。
  rm -f D:/2/dataset_noleak/labels/*.cache
  log "已清理冒烟遗留 cache（正式训练将重新扫描 1401 张）"
  local t0; t0=$(date +%s)
  local i=0
  for item in "${NAMES[@]}"; do
    i=$((i + 1))
    IFS=':' read -r att seed name <<< "$item"
    log "[$i/6] $name"
    train_one "$att" "$seed" "$name"
  done
  log "STAGE 1 全部完成，总耗时 $(secs $(( $(date +%s) - t0 )))"
}

# ---------------------------------------------------------------------------
# stage 2：评估 + WBF
# ---------------------------------------------------------------------------
stage_eval() {
  rule
  log "STAGE 2 新 test 评估（203 图闭卷，零视觉段跨 split）"
  # 第三族 old896 是**对照组**：旧权重（179 图训练所得）跑同一张新 203 图卷。
  # 它不等于「干净底座」——203 张里有 171 张原本就在旧 train 里，旧权重见过。
  # 但它能回答一个别处答不了的问题：重训后分数是回到原位，还是被真掉了一截。
  local fam='{"base":["nf_v11_noleak_s42","nf_v11_noleak_s43","nf_v11_noleak_s44"],"triplet":["att_v11_noleak_triplet_s42","att_v11_noleak_triplet_s43","att_v11_noleak_triplet_s44"],"old896":["nf_v11_896_s42","nf_v11_896_s43","nf_v11_896_s44"]}'
  local t0; t0=$(date +%s)
  NOLEAK_YAML="$DATA" \
  NOLEAK_FAMILIES="$fam" \
  NOLEAK_CACHE="$LOGDIR/wbf_cache_noleak" \
  NOLEAK_EVALDIR="D:/2/exp_upscale/eval_noleak" \
  NOLEAK_TSV="$TSV" \
  NOLEAK_JSON="$LOGDIR/eval_noleak_final.json" \
  NOLEAK_SKIP_TTA=1 \
    "$PY" "$EVAL" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then log "✗ 评估失败 rc=$rc"; exit $rc; fi
  log "STAGE 2 完成，用时 $(secs $(( $(date +%s) - t0 )))；结果 $TSV"
}

# ---------------------------------------------------------------------------
MODE="${1:---all}"
case "$MODE" in
  --smoke) preflight; stage_smoke ;;
  --train) preflight; stage_train ;;
  --eval)  stage_eval ;;
  --all)   preflight; stage_smoke; stage_train; stage_eval
           rule; log "全链完成 ★ 结果见 $TSV" ;;
  *) echo "用法: bash $0 [--all|--smoke|--train|--eval]"; exit 2 ;;
esac
