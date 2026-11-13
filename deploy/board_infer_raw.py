# 板上第一步：在真实数据上验证模型输出对不对（不碰摄像头、不碰屏幕）
#
# 依赖两个文件，都在 /sdcard：
#   yolo11s_640_u8r.kmodel   11.08 MB，uint8 NHWC 输入 [1,640,640,3]，归一化写进图里
#   testcase.bin             1228800 字节 = 640*640*3，uint8 RGB，已 letterbox
#
# 判读基准（来自 PC 上同一输入跑 ONNX，见 board_case/expect.json）：
#   top1 分数 0.597，类别 thunder(class 1)
#   框在 **640 letterbox 空间**：cx≈342 cy≈258 w≈29 h≈29
#     （注意：expect.json 里的 604/773 是原图 1080x1920 像素坐标，
#      换算 = 原图*0.3333 + pad_left(140) / pad_top(0)，别拿原图数直接比）
#   超过 0.10 的 anchor 数量：PC 上 ONNX 为 9 个
#
# INT8 量化会让分数略低、框略有台阶（约 2.5 px），同类别 + 框位置相近 + anchor 数量同量级即通过。

import time

import nncase_runtime as nn
import ulab.numpy as np

KM = "/sdcard/yolo11s_640_u8r.kmodel"
BIN = "/sdcard/testcase.bin"
W = H = 640
NA = 8400
CONF = 0.10

print("=== 1. 加载 kmodel ===")
t0 = time.ticks_ms()
kpu = nn.kpu()
kpu.load_kmodel(KM)
print("加载 %d ms" % (time.ticks_ms() - t0))
# inputs_desc 在不同固件版本上签名不一致：有的要 index，有的不接受参数，
# 这里逐个试，任何一个能用就打印，全失败也不中断（描述只是辅助信息）
for call in (lambda: kpu.inputs_desc(0), lambda: kpu.inputs_desc()):
    try:
        print("  in :", call())
        break
    except Exception as e:
        print("  inputs_desc 试法失败：%s" % type(e).__name__)
for call in (lambda: kpu.outputs_desc(0), lambda: kpu.outputs_desc()):
    try:
        print("  out:", call())
        break
    except Exception as e:
        print("  outputs_desc 试法失败：%s" % type(e).__name__)

print("=== 2. 读输入并构造 tensor ===")
t0 = time.ticks_ms()
f = open(BIN, "rb")
raw = f.read()
f.close()
print("  读到 %d 字节（应为 %d），读盘 %d ms" % (len(raw), W * H * 3, time.ticks_ms() - t0))
if len(raw) != W * H * 3:
    raise SystemExit("输入文件长度不对，重新上传 testcase.bin")

arr = np.frombuffer(raw, dtype=np.uint8).reshape((1, H, W, 3))
print("  张量形状", arr.shape, "元素数", arr.size)

try:
    ten = nn.from_numpy(arr)
except Exception as e:
    print("  nn.from_numpy 直接吃 ulab 数组失败（%s），改用 bytes 中转" % type(e).__name__)
    ten = nn.from_numpy(bytes(raw))
kpu.set_input_tensor(0, ten)

print("=== 3. 推理 ===")
t0 = time.ticks_ms()
kpu.run()
t1 = time.ticks_ms()
print("  首次推理 %d ms" % (t1 - t0))

out = kpu.get_output_tensor(0).to_numpy()
print("  输出形状", out.shape, "类型", type(out))
y = out.reshape((7, NA))

print("=== 4. 解码（纯 Python 循环，先求正确再求快）===")
t0 = time.ticks_ms()
n_hit = 0
best_score = -1.0
best_idx = 0
best_cls = -1
hots = []          # [(score, cls, idx)]，只留超阈值的
for i in range(NA):
    s0 = float(y[4, i])
    s1 = float(y[5, i])
    s2 = float(y[6, i])
    s = s0
    c = 0
    if s1 > s:
        s = s1
        c = 1
    if s2 > s:
        s = s2
        c = 2
    if s > best_score:
        best_score = s
        best_idx = i
        best_cls = c
    if s > CONF:
        n_hit += 1
        hots.append((s, c, i))
print("  解码耗时 %d ms" % (time.ticks_ms() - t0))

hots.sort(reverse=True)
print("=== 5. 结果 ===")
print("  top1: score=%.3f cls=%d   [期望 0.597 / 1]" % (best_score, best_cls))
i = best_idx
cx = float(y[0, i]); cy = float(y[1, i]); bw = float(y[2, i]); bh = float(y[3, i])
print("  top1 框 cx=%.1f cy=%.1f w=%.1f h=%.1f   [期望 ≈342/258/29/29]" % (cx, cy, bw, bh))
print("  超 %.2f 的 anchor 数 = %d   [期望 9]" % (CONF, n_hit))
print("  前 8 个：")
for s, c, k in hots[:8]:
    print("    score=%.3f cls=%d  box=(%.1f, %.1f, %.1f, %.1f)"
          % (s, c, float(y[0, k]), float(y[1, k]), float(y[2, k]), float(y[3, k])))

print("=== 6. 连跑 5 次测速（预热后）===")
t0 = time.ticks_ms()
for _ in range(5):
    kpu.run()
t1 = time.ticks_ms()
ms = (t1 - t0) / 5.0
print("  平均 %.1f ms/次  ->  %.1f fps（纯推理，不含采集/后处理）" % (ms, 1000.0 / ms))

print("=== 完成 ===")
