# 板上冒烟测试：只验证「kmodel 能不能加载 + 能不能跑 + 跑多快」
# 不碰摄像头、不碰屏幕、不写 flash。最坏结果是抛异常退出，板子不受影响。
#
# 用法（CanMV IDE 里直接运行，或 mpremote run）：
#   mpremote connect COM8 run D:/2/deploy/board_smoke.py
#
# 输入张量的 dtype/形状从 kpu.inputs_desc() 里解析，不写死 ——
# 因为 float32 NCHW 版和 uint8 NHWC 版的 desc 不一样，写死必然对不上。

import re
import time

import nncase_runtime as nn
import ulab.numpy as np

KM = "/sdcard/yolo11s_640_u8r.kmodel"


def build_input(desc):
    """按 desc 里的 dtype 和形状造一个全零输入。"""
    s = str(desc)
    if 'float32' in s:
        dt = np.float32
    elif 'int8' in s and 'uint8' not in s:
        dt = np.int8
    else:
        dt = np.uint8
    nums = [int(x) for x in re.findall(r'\d+', s)]
    # desc 里还带着别的数字（对齐、量化参数等），取最后 4 个 >=1 的当形状不可靠，
    # 改成穷举合法 4 维组合里最像 NCHW/NHWC 的那组
    cand = [tuple(nums[i:i + 4]) for i in range(len(nums) - 3)]
    shape = None
    for c in cand:
        if len(c) == 4 and c[0] == 1 and 640 in c:
            shape = c
    if shape is None:
        shape = (1, 640, 640, 3) if 'uint8' in s or 'int8' in s else (1, 3, 640, 640)
    print("  解析 dtype=%s shape=%s（原始 desc: %s）" % (dt, shape, s))
    return np.zeros(shape, dtype=dt)


print("=== 1. 加载 kmodel ===")
t0 = time.ticks_ms()
kpu = nn.kpu()
kpu.load_kmodel(KM)
print("加载耗时 %d ms" % (time.ticks_ms() - t0))

print("=== 2. 读 IO 契约 ===")
n_in = kpu.inputs_size()
n_out = kpu.outputs_size()
print("输入个数", n_in)
for i in range(n_in):
    print("  in[%d]" % i, kpu.inputs_desc(i))
print("输出个数", n_out)
for i in range(n_out):
    print("  out[%d]" % i, kpu.outputs_desc(i))

print("=== 3. 全零输入跑一次（只测通路，不看结果）===")
x = build_input(kpu.inputs_desc(0))
kpu.set_input_tensor(0, nn.from_numpy(x))

t0 = time.ticks_ms()
kpu.run()
t1 = time.ticks_ms()
print("首次推理 %d ms" % (t1 - t0))

out = kpu.get_output_tensor(0).to_numpy()
print("输出形状", out.shape, "元素数", out.size)

print("=== 4. 连跑 5 次取稳定帧率 ===")
for _ in range(2):          # 预热，丢掉前两次
    kpu.set_input_tensor(0, nn.from_numpy(x))
    kpu.run()

t0 = time.ticks_ms()
N = 5
for _ in range(N):
    kpu.set_input_tensor(0, nn.from_numpy(x))
    kpu.run()
dt = time.ticks_ms() - t0
print("纯推理 %d 次共 %d ms → 单次 %.1f ms → %.1f fps"
      % (N, dt, dt / N, 1000.0 * N / dt))
print("（这是不含摄像头采集与后处理的上限值）")

print("=== 冒烟完成：kmodel 可加载、可推理 ===")
