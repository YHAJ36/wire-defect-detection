# 02_check_onnx.py —— 检查刚导出的 ONNX 是不是"能用"
#
# 什么叫"能用"?三个条件,缺一不可:
#   1. 输入形状对: 必须是 [1, 3, 640, 640]   —— 喂得进去
#   2. 输出形状对: 必须是 [1, 7, 8400]       —— 拿得出来 (7 = 4个框参数 + 3个类别分数)
#   3. 数值对      : 和之前跑通的旧 ONNX 喂同一张图,输出几乎一样 —— 算得对
#
# 第 3 条最关键。形状对但数值错,说明导出时哪里漏了,这种模型上板会"跑得动但框全歪"。

import numpy as np
import onnxruntime as ort

NEW = r'D:/2/manual/yolo11n_640.onnx'      # 你自己刚导出的
OLD = r'D:/2/deploy/yolo11n_640.onnx'      # 之前验证过、能跑的
BIN = r'D:/2/deploy/board_case/testcase.bin'   # 同一张测试图(640x640x3, uint8, 未归一化)

# ---------- 读图: testcase.bin 是裸二进制, 长度正好 640*640*3 ----------
raw = np.fromfile(BIN, dtype=np.uint8)
print(f'testcase.bin  字节数 {raw.size}  ({raw.size / 1024:.0f} KB)')
img = raw.reshape(640, 640, 3)          # 板上是 NHWC 排列: 高, 宽, 通道
x = img.transpose(2, 0, 1)[None]        # 转成 ONNX 要的 NCHW: 1, 通道, 高, 宽
x = x.astype(np.float32) / 255.0        # 归一化到 0~1
print(f'喂进去的形状 {x.shape}, 取值 {x.min():.3f} ~ {x.max():.3f}')

# ---------- 两个模型各跑一遍 ----------
outs = {}
for name, path in (('新', NEW), ('旧', OLD)):
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    i = sess.get_inputs()[0]
    o = sess.get_outputs()[0]
    print(f'\n[{name}] {path}')
    print(f'   输入 {i.name} {i.shape} {i.type}')
    print(f'   输出 {o.name} {o.shape} {o.type}')
    outs[name] = sess.run(None, {i.name: x})[0]

a, b = outs['新'], outs['旧']
print(f'\n两个模型的输出最大差异 {np.abs(a - b).max():.6f}')
if np.abs(a - b).max() < 1e-4:
    print('=> 数值一致,导出正确')
else:
    print('=> 数值不一致!导出过程有问题,不要往下走')

# ---------- 顺便看一眼它到底"看到"了什么 ----------
y = a[0]                                  # (7, 8400)
cls = y[4:7]                              # 后 3 行 = 3 个类别的分数
best = cls.max(0)                         # 每个格子的最高类别分
k = int(best.argmax())                    # 分数最高的那个格子
cx, cy, bw, bh = y[0, k], y[1, k], y[2, k], y[3, k]   # 前 4 行 = 框参数

names = ['break', 'thunderbolt', 'wear']
print(f'\n最高分的格子 第 {k} 号: 类别 {names[int(cls[:, k].argmax())]}  '
      f'分数 {best[k]:.3f}')
print(f'   框中心 ({cx:.1f}, {cy:.1f})  宽高 ({bw:.1f}, {bh:.1f})  [640 空间]')
print('\n对照: 板上实测中心 (342.3, 257.3), 差 %.1f / %.1f px'
      % (342.3 - cx, 257.3 - cy))
