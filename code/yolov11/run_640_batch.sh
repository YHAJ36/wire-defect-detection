#!/usr/bin/env bash
# 640 新基点：分离「分辨率」与「增强」两个因素，再给部署候选 n 档
# 用法: bash code/yolov11/run_640_batch.sh [1|2|3|all]
# 断点续跑：已满 epochs 的 run 自动跳过；未满则 --resume
# 防撞名：若目标目录已存在且 args.yaml 里的 data 不是 dataset_noleak，直接拒跑
set -u
cd /d/2 || exit 1
PY="D:/yolo-env/Scripts/python.exe"
DATA="D:/2/dataset_noleak/data.yaml"
PROJ="D:/2/exp_upscale/runs"
LOG="D:/2/exp_upscale/logs/batch640.log"
EPOCHS=60
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

epochs_done(){
    local csv="$PROJ/$1/results.csv"
    [ -f "$csv" ] && echo $(( $(wc -l < "$csv") - 1 )) || echo 0
}

# 防撞名：目录已存在但不是本次数据集 → 拒绝
guard(){
    local name="$1" ax="$PROJ/$1/args.yaml"
    [ -f "$ax" ] || return 0
    if ! grep -q "dataset_noleak" "$ax"; then
        log "!! 拒绝：$name 已存在且不是 dataset_noleak 数据集（$(grep -m1 '^data:' "$ax")）"
        return 1
    fi
    return 0
}

# $1=说明 $2=name $3=weights $4..=额外参数
train_one(){
    local desc="$1" name="$2" w="$3"; shift 3
    guard "$name" || return 1
    local done_n; done_n=$(epochs_done "$name")
    if [ "$done_n" -ge "$EPOCHS" ]; then
        log "跳过 $name（已 $done_n 轮）"; return 0
    fi
    local resume=""
    [ "$done_n" -gt 0 ] && resume="--resume"
    log ">>> $desc | $name | 已 $done_n 轮 ${resume:+(续训)}"
    "$PY" code/yolov11/train_perturb.py \
        --att baseline --seed 42 --epochs "$EPOCHS" --imgsz 640 \
        --data "$DATA" --name "$name" --project "$PROJ" \
        --weights "$w" $resume "$@" >> "$LOG" 2>&1
    log "<<< $name 结束，轮数 $(epochs_done "$name")"
}

W_S="D:/1/models/yolo11s.pt"
W_N="D:/1/models/yolo11n.pt"

# 组 1：只改分辨率，增强保持历史 BASE（mosaic0.8 / mixup0.1 / copy_paste0.1 / scale0.5 / close5）
# 目的：单独隔离「896→640」这一个变量
group1(){ log "=== 组1: s@640 历史增强（隔离分辨率因素）"
    train_one "s@640 base-aug" "nf_v11_nl640_s42" "$W_S" --batch 8; }

# 组 2：640 + 小目标增强（mosaic降、mixup/copy_paste关、scale降、close_mosaic延长）
# 目的：在 640 上单独隔离「增强配置」这一个变量
group2(){ log "=== 组2: s@640 小目标增强（隔离增强因素）"
    train_one "s@640 opt-aug" "nf_v11_nl640opt_s42" "$W_S" \
        --batch 16 --mosaic 0.5 --mixup 0.0 --copy-paste 0.0 \
        --scale 0.2 --close-mosaic 20; }

# 组 3：n 档 + 小目标增强（K230 部署候选）
group3(){ log "=== 组3: n@640 小目标增强（部署候选）"
    train_one "n@640 opt-aug" "nf_v11n_nl640opt_s42" "$W_N" \
        --batch 16 --mosaic 0.5 --mixup 0.0 --copy-paste 0.0 \
        --scale 0.2 --close-mosaic 20; }

case "${1:-all}" in
    1) group1 ;;
    2) group2 ;;
    3) group3 ;;
    all) group1; group2; group3 ;;
    *) echo "用法: $0 [1|2|3|all]"; exit 1 ;;
esac
log "批次结束"
