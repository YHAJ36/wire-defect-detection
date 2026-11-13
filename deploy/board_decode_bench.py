# 解码向量化对比：纯 Python 8400 循环 vs ulab 向量化
import time
import nncase_runtime as nn
import ulab.numpy as np

kpu = nn.kpu()
kpu.load_kmodel("/sdcard/yolo11s_640_u8r.kmodel")
raw = open("/sdcard/testcase.bin", "rb").read()
kpu.set_input_tensor(0, nn.from_numpy(np.frombuffer(raw, dtype=np.uint8).reshape((1, 640, 640, 3))))
kpu.run()
y = kpu.get_output_tensor(0).to_numpy().reshape((7, 8400))

# A: 纯 Python 循环
t0 = time.ticks_ms()
n = 0
best = -1.0
for i in range(8400):
    s0 = float(y[4, i]); s1 = float(y[5, i]); s2 = float(y[6, i])
    s = s0
    if s1 > s: s = s1
    if s2 > s: s = s2
    if s > best: best = s
    if s > 0.10: n += 1
tA = time.ticks_ms() - t0
print("A 纯循环   %d ms  best=%.3f n=%d" % (tA, best, n))

# B: ulab 向量化
t0 = time.ticks_ms()
cls = y[4:7]                       # (3,8400)
sc = np.max(cls, axis=0)           # (8400,)
tB1 = time.ticks_ms() - t0
t0 = time.ticks_ms()
bestv = float(np.max(sc))
idx = int(np.argmax(sc))
tB2 = time.ticks_ms() - t0
t0 = time.ticks_ms()
hits = sc > 0.10
nh = 0
for v in hits:
    if v: nh += 1
tB3 = time.ticks_ms() - t0
print("B 向量化   %d ms  (max %d + argmax %d + 计数 %d)  best=%.3f n=%d"
      % (tB1 + tB2 + tB3, tB1, tB2, tB3, bestv, nh))

# C: 只要 top1 时不做计数
t0 = time.ticks_ms()
sc2 = np.max(y[4:7], axis=0)
j = int(np.argmax(sc2))
cx = float(y[0, j]); cy = float(y[1, j])
tC = time.ticks_ms() - t0
print("C 只取top1 %d ms  score=%.3f idx=%d cx=%.1f cy=%.1f" % (tC, float(sc2[j]), j, cx, cy))
