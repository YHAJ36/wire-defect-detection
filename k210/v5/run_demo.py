# -*- coding: utf-8 -*-
"""K210 板上运行：摄像头实时检测 + 屏幕显示 + 串口报警。

板端做的事：
  1. 从 /sd 加载 kmodel
  2. 声明三层输出形状
  3. 开摄像头，窗口裁成 320x224（KPU 上限，且必须是 stride 32 的倍数）
  4. 循环：拍照 -> 推理 -> 解码框 -> 画到屏幕上 -> 超阈值发串口报警
  5. 打印每帧耗时、最高置信度、框位置

关于内存：模型工作内存 643 KB，加载后剩约 500 KB；LCD 帧缓冲要 150 KB，
所以必须先 load 模型再 lcd.init()，反过来会 OOM。

用法：
  python run_demo.py                          # 10 帧，/sd/demo.kmodel
  python run_demo.py --model /sd/strike.kmodel --frames 20
  python run_demo.py --no-lcd                 # 不开屏，省 150 KB
"""
import argparse
import sys

sys.path.insert(0, r'D:/2/k210')
from k210 import K210  # noqa: E402

BOARD_CODE = r'''
import sensor, time, gc
import KPU as kpu
from machine import UART
from fpioa_manager import fm

MODEL   = "{model}"
FRAMES  = {frames}
THRESH  = {thresh}      # obj logit 阈值；-0.5 约等于 conf 0.38
USE_LCD = {use_lcd}

# ---- 串口：UART1 IO30(TX) -> STM32，9600 ----
# 原门禁固件里 K210 就是靠这条线发 {{ok}} / {{fail}} 让 STM32 动作（蜂鸣器/舵机）。
try:
    fm.register(30, fm.fpioa.UART1_TX, force=True)
    fm.register(31, fm.fpioa.UART1_RX, force=True)
    uart = UART(UART.UART1, 9600, 8, 1, 0, timeout=1000)
    print("UART OK")
except Exception as e:
    uart = None
    print("UART fail:", e)

# ---- 模型（先加载，再开屏：LCD 要 150KB 帧缓冲） ----
task = kpu.load(MODEL)
kpu.set_outputs(task, 0, 28, 40, 18)
kpu.set_outputs(task, 1, 14, 20, 18)
kpu.set_outputs(task, 2, 7, 10, 18)
gc.collect()
print("LOAD OK  free=", gc.mem_free())

# ---- 屏幕 ----
lcd_on = False
if USE_LCD:
    try:
        import lcd
        lcd.init()
        lcd_on = True
        print("LCD OK  free=", gc.mem_free())
    except Exception as e:
        print("LCD fail:", e)

# ---- 摄像头 ----
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((0, 8, 320, 224))
sensor.set_hmirror(1)
sensor.set_vflip(1)
sensor.run(1)
time.sleep_ms(800)

# 每层: (ny, nx, stride, 3 个 anchor 的 (w,h))
# anchor 值取自导出日志，已乘 stride，单位是输入图像素
HEADS = (
    (28, 40, 8,  ((10.0, 13.0), (16.0, 30.0), (33.0, 23.0))),
    (14, 20, 16, ((30.0, 61.0), (62.0, 45.0), (59.0, 119.0))),
    (7,  10, 32, ((116.0, 90.0), (156.0, 198.0), (373.0, 326.0))),
)
NA = 3
NO = 6          # 每个 anchor: x y w h obj cls

def sig(v):
    return 1.0 / (1.0 + 2.718281828 ** (-v))

alarm_last = -99999
hit = 0

for i in range(FRAMES):
    img = sensor.snapshot()
    t0 = time.ticks_ms()
    kpu.forward(task, img)
    ms = time.ticks_diff(time.ticks_ms(), t0)

    best = -99.0
    bi = (0, 0, 0, 0)       # (head, anchor, row, col)
    for hi in range(3):
        ny = HEADS[hi][0]
        nx = HEADS[hi][1]
        NS = ny * nx
        o = kpu.get_output(task, hi)
        for a in range(NA):
            base = (a * NO + 4) * NS
            for k in range(NS):
                v = o[base + k]
                if v > best:
                    best = v
                    bi = (hi, a, k // nx, k % nx)

    conf = sig(best)
    box = None
    if best > THRESH:
        hi = bi[0]
        a = bi[1]
        row = bi[2]
        col = bi[3]
        stride = HEADS[hi][2]
        anchors = HEADS[hi][3]
        NS = HEADS[hi][0] * HEADS[hi][1]
        o = kpu.get_output(task, hi)
        base = a * NO * NS
        idx = row * HEADS[hi][1] + col
        px = o[base + 0 * NS + idx]
        py = o[base + 1 * NS + idx]
        pw = o[base + 2 * NS + idx]
        ph = o[base + 3 * NS + idx]
        aw = anchors[a][0]
        ah = anchors[a][1]
        cx = (sig(px) * 2.0 - 0.5 + col) * stride
        cy = (sig(py) * 2.0 - 0.5 + row) * stride
        bw = (sig(pw) * 2.0) ** 2 * aw
        bh = (sig(ph) * 2.0) ** 2 * ah
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        box = (x1, y1, int(bw), int(bh))

    # ---- 画到屏幕 ----
    if lcd_on:
        if box:
            img.draw_rectangle(box[0], box[1], box[2], box[3],
                               color=(255, 0, 0), thickness=2)
            img.draw_string(max(box[0], 0), max(box[1] - 16, 0),
                            "%.2f" % conf, color=(255, 0, 0), scale=1.3)
        img.draw_string(4, 4, "%dms c%.2f" % (ms, conf),
                        color=(0, 255, 0), scale=1.3)
        lcd.display(img)

    # ---- 报警 ----
    fired = 0
    if best > THRESH:
        hit += 1
        now = time.ticks_ms()
        if uart and time.ticks_diff(now, alarm_last) > 3000:
            uart.write("{{fail}}")
            alarm_last = now
            fired = 1

    print("f%02d %4dms obj=%6.2f conf=%.3f h%d a%d (y%02d,x%02d) box=%s alarm=%d"
          % (i, ms, best, conf, bi[0], bi[1], bi[2], bi[3], box, fired))

print("SUMMARY %d/%d frames above thresh" % (hit, FRAMES))
kpu.deinit(task)
print("DONE")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/sd/demo.kmodel')
    ap.add_argument('--frames', type=int, default=10)
    ap.add_argument('--thresh', type=float, default=-0.5,
                    help='obj logit 阈值。实测：真雷击 >= 0.0，背景 <= -0.73，取中间 -0.5')
    ap.add_argument('--no-lcd', action='store_true', help='不开屏幕')
    ap.add_argument('--timeout', type=int, default=300)
    a = ap.parse_args()

    code = BOARD_CODE.format(model=a.model, frames=a.frames, thresh=a.thresh,
                             use_lcd='False' if a.no_lcd else 'True')
    d = K210()
    print(d.exec(code, timeout=a.timeout))
    d.close()


if __name__ == '__main__':
    main()
