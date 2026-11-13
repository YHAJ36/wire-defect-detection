#!/usr/bin/env bash
# 150 轮预算对照：跑完即评（不训练、不改任何已封版数字，只产诊断）
# 用法：bash D:/2/code/tools/run_e150_eval.sh
# 前提：GPU 上训练已结束（nvidia-smi 里没有 train_att_640.py 占显存）
set -u

PY="D:/yolo-env/Scripts/python.exe"
RUNS="D:/2/exp_upscale/runs"
A="nf_v11_896_s42"           # 60 轮基线（同种子 s42）
B="nf_v11_896_e150_s42"      # 150 轮（只改 epochs）
LOGD="D:/2/report/_p2_diag/logs"
mkdir -p "$LOGD"

echo "======================================================"
echo " 0. 前置检查：训练是否真的结束"
echo "======================================================"
N=$("$PY" -c "import csv;print(len(list(csv.DictReader(open('$RUNS/$B/results.csv')))))")
echo "  $B 已完成 epoch 数：$N / 150"
if [ "$N" -lt 150 ]; then
  echo "  ⛔ 还没跑完，退出。等 last.pt 稳定后再执行。"
  exit 1
fi
MN=$("$PY" - "$RUNS/$B/weights/last.pt" <<'EOF'
import os, sys, time
print(int((time.time() - os.path.getmtime(sys.argv[1])) // 60))
EOF
)
echo "  last.pt 最后写入距今 ${MN} 分钟"
if [ "$MN" -lt 2 ]; then
  echo "  ⛔ last.pt 还在写，可能正在收尾 val，等 2 分钟再跑。"
  exit 1
fi
echo "  ✅ 训练已结束"

echo
echo "======================================================"
echo " 1. 按交付口径（NMS 0.5）在 val 上重挑检查点 —— 60 轮"
echo "======================================================"
"$PY" "D:/2/code/tools/eval/reselect_nms05.py" --run "$A" --imgsz 896 --iou 0.5 \
  --device 0 --report-test 2>&1 | tee "$LOGD/reselect_${A}.log"

echo
echo "======================================================"
echo " 2. 按交付口径（NMS 0.5）在 val 上重挑检查点 —— 150 轮"
echo "======================================================"
"$PY" "D:/2/code/tools/eval/reselect_nms05.py" --run "$B" --imgsz 896 --iou 0.5 \
  --device 0 --report-test 2>&1 | tee "$LOGD/reselect_${B}.log"

echo
echo "======================================================"
echo " 3. 主对比：last.pt vs last.pt（无挑点偏差，最干净）"
echo "======================================================"
"$PY" "D:/2/code/tools/eval/eval_ckpts_test.py" \
  --ckpt "$A/weights/last.pt" \
  --ckpt "$B/weights/last.pt" \
  --imgsz 896 --iou 0.5 --split test --device 0 --tag "lastpt_60v150" \
  2>&1 | tee "$LOGD/lastpt_60v150.log"

echo
echo "======================================================"
echo " 4. 参考：两边 best.pt（受训练期 iou=0.7 影响，仅参考）"
echo "======================================================"
"$PY" "D:/2/code/tools/eval/eval_ckpts_test.py" \
  --ckpt "$A/weights/best.pt" \
  --ckpt "$B/weights/best.pt" \
  --imgsz 896 --iou 0.5 --split test --device 0 --tag "bestpt_60v150_ref" \
  2>&1 | tee "$LOGD/bestpt_60v150_ref.log"

echo
echo "全部完成。日志在 $LOGD，json 在 D:/2/report/_p2_diag/"
echo "判读用 A.9.47 三档：≥+3.7 真瓶颈 / 0~3.7 不可判（不停线）/ ≤0 无收益"
