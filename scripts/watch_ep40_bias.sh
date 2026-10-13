#!/usr/bin/env bash
# P3：等 epoch40.pt 落地，立刻读 SA conv 的 bias，与 epoch20.pt 对比算 Δbias/20轮。
#
# 为什么必须自动、不能事后看图说话：预钉判据②/③的输入就是 Δbias/20轮。
# 若等训练全部结束再回头翻，就变成"按结果挑解释"。
#
# 为什么必须等文件写稳：torch.save 写 38MB 权重有几百毫秒窗口，
# 期间文件存在但内容不全，读它会得到损坏的 state_dict。
# 故先等存在，再等**大小连续两次相同**，才认为写稳。
#
# 为什么只用这个 run 的 epoch20/epoch40：60 轮与 20 轮跑的 LR 调度不同
# （20 轮是压缩余弦），跨 run 比 Δbias 没有意义。probe20 那个 run 的
# checkpoint **不可用于本对比**。
#
# 注意：不碰 logs/train_e60_fixed.log，不碰任何 runs/ 下的文件（只读）。
set -u
cd /d/2 || exit 1

CKPT="runs/wire_v11_640_cbamfix/weights"
OUT="logs/sa_bias_ep40.txt"
PY="D:/yolo-env/Scripts/python.exe"

echo "[watch] 等待 $CKPT/epoch40.pt ..."
for _ in $(seq 1 720); do   # 最多等 60 分钟
    if [ -f "$CKPT/epoch40.pt" ]; then
        a=$(stat -c %s "$CKPT/epoch40.pt" 2>/dev/null || echo 0)
        sleep 6
        b=$(stat -c %s "$CKPT/epoch40.pt" 2>/dev/null || echo 0)
        if [ "$a" = "$b" ] && [ "$a" -gt 1000000 ]; then
            echo "[watch] epoch40.pt 已写稳（$a 字节），开始读数"
            break
        fi
        continue
    fi
    sleep 5
done

if [ ! -f "$CKPT/epoch40.pt" ]; then
    echo "[watch] 超时：epoch40.pt 未出现，放弃" | tee "$OUT"
    exit 1
fi

# 同 run 的 ep20 与 ep40；另附 ep0 作初值锚点（同 run，口径一致）
{
  echo "===== P3：修复臂@60 的 bias 轨迹（同 run，60 轮 LR 调度）====="
  echo "run   : runs/wire_v11_640_cbamfix  （60 轮调度）"
  echo "输入  : 预钉判据 ②/③ 的 Δbias/20轮"
  echo "依据  : check_sa_bias.py 的理论 —— 中心化后 mean_spatial(z) = SA conv bias"
  echo "        （bias 轨迹即 DC 轨迹，差值实测 <0.047，见 archive/2026-10-12.md）"
  echo
  "$PY" code/tools/check_sa_bias.py \
      --ckpt "ep0=$CKPT/epoch0.pt" \
      --ckpt "ep20=$CKPT/epoch20.pt" \
      --ckpt "ep40=$CKPT/epoch40.pt"
} 2>&1 | tee "$OUT"

echo
echo "=========== P3 bias 读数完成，输出 $OUT ==========="
