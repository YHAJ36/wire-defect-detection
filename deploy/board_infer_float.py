# 板上验证：手动编译的那个 kmodel（float32 NCHW 输入）
#
# 和 board_infer_raw.py 的唯一区别：这个 kmodel 要的是 0~1 的小数、通道在前，
# 不是 0~255 的整数、通道在后。所以板上要多干一步格式转换。
# 这一步要花多少时间，脚本会打出来——这就是"用 uint8 输入"能省下来的东西。
#
# 需要 /sdcard 上的两个文件：
#   yolo11n_640_manual.kmodel   3.42 MB，输入 [1,3,640,640] float32
#   testcase.bin                1228800 字节 = 640*640*3，uint8 RGB
#
# 期望值（PC 模拟器 04_verify_kmodel.py 给的）：
#   top1 分数 0.713  类别 1 (thunderbolt)
#   框 cx≈341.0 cy≈258.2 w≈27 h≈29        [640 letterbox 空间]
#   分数 > 0.10 的 anchor 数：9
#
# 运行：
#   python -m mpremote connect COM8 run D:/2/deploy/board_infer_float.py

import gc
import time

import nncase_runtime as nn
import ulab.numpy as np

KM = "/sdcard/yolo11n_640_manual.kmodel"
BIN = "/sdcard/testcase.bin"
W = H = 640
NA = 8400
CONF = 0.10

# ulab 各固件对 float32 的叫法不统一，先探一下叫什么
FT = getattr(np, "float32", None) or getattr(np, "float")
print("float 类型名：", FT, " 可用内存", gc.mem_free())


def to_float_chw(raw):
    """uint8 的 640x640x3 -> float32 的 1x3x640x640。

    三件事：整数变小数的（除 255）、通道从最后挪到最前、最外面套一层 batch。
    每步都留了退路：ulab 的 API 在不同固件上不完全一样，硬写死一种会直接报错。
    """
    a = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))

    try:
        f = a.astype(FT)
        how = "astype"
    except Exception as e:
        print("  astype 不可用(%s)，改用 array" % type(e).__name__)
        f = np.array(a, dtype=FT)
        how = "np.array"
    f = f / 255.0

    try:
        t = np.transpose(f, (2, 0, 1))
        how += " + transpose"
    except Exception as e:
        print("  transpose 不可用(%s)，改为逐通道搬运" % type(e).__name__)
        t = np.zeros((3, H, W), dtype=FT)
        for c in range(3):
            t[c] = f[:, :, c]
        how += " + 逐通道"
    return t.reshape((1, 3, H, W)), how


print("=== 1. 加载 kmodel ===")
t0 = time.ticks_ms()
kpu = nn.kpu()
kpu.load_kmodel(KM)
print("  加载 %d ms" % (time.ticks_ms() - t0))
for call in (lambda: kpu.inputs_desc(0), lambda: kpu.inputs_desc()):
    try:
        print("  in :", call())
        break
    except Exception as e:
        print("  inputs_desc 试法失败：%s" % type(e).__name__)

print("=== 2. 读图 + 转 float ===")
t0 = time.ticks_ms()
f = open(BIN, "rb")
raw = f.read()
f.close()
print("  读到 %d 字节，读盘 %d ms，free %d KB"
      % (len(raw), time.ticks_ms() - t0, gc.mem_free() // 1024))
if len(raw) != W * H * 3:
    raise SystemExit("输入文件长度不对，重新上传 testcase.bin")

t0 = time.ticks_ms()
arr, how = to_float_chw(raw)
print("  转换 %d ms（%s），free %d KB"
      % (time.ticks_ms() - t0, how, gc.mem_free() // 1024))
print("  形状", arr.shape)
kpu.set_input_tensor(0, nn.from_numpy(arr))
del arr
gc.collect()
print("  喂进去后 free %d KB" % (gc.mem_free() // 1024))

print("=== 3. 推理 ===")
t0 = time.ticks_ms()
kpu.run()
print("  首次推理 %d ms" % (time.ticks_ms() - t0))

out = kpu.get_output_tensor(0).to_numpy()
print("  输出形状", out.shape)
y = out.reshape((7, NA))

print("=== 4. 解码 ===")
t0 = time.ticks_ms()
hots = []
best_score, best_idx, best_cls = -1.0, 0, -1
for i in range(NA):
    s0 = float(y[4, i]); s1 = float(y[5, i]); s2 = float(y[6, i])
    s, c = s0, 0
    if s1 > s:
        s, c = s1, 1
    if s2 > s:
        s, c = s2, 2
    if s > best_score:
        best_score, best_idx, best_cls = s, i, c
    if s > CONF:
        hots.append((s, c, i))
print("  解码 %d ms" % (time.ticks_ms() - t0))

print("=== 5. 结果核对 ===")
print("  top1: score=%.3f cls=%d      [期望 0.713 / 1]" % (best_score, best_cls))
i = best_idx
print("  框 cx=%.1f cy=%.1f w=%.1f h=%.1f   [期望 341.0 / 258.2 / 27 / 29]"
      % (float(y[0, i]), float(y[1, i]), float(y[2, i]), float(y[3, i])))
print("  超 %.2f 的 anchor 数 = %d        [期望 9]" % (CONF, len(hots)))
hots.sort(reverse=True)
for s, c, k in hots[:8]:
    print("    score=%.3f cls=%d box=(%.1f, %.1f, %.1f, %.1f)"
          % (s, c, float(y[0, k]), float(y[1, k]), float(y[2, k]), float(y[3, k])))

print("=== 6. 连跑 5 次测速 ===")
t0 = time.ticks_ms()
for _ in range(5):
    kpu.run()
ms = (time.ticks_ms() - t0) / 5.0
print("  纯推理 %.1f ms/次  ->  %.1f fps" % (ms, 1000.0 / ms))

del out, y
gc.collect()
print("  结束时 free %d KB" % (gc.mem_free() // 1024))
print("=== 完成 ===")
