# K230 板上完整演示：摄像头 → ai2d → KPU → 解码 → 屏幕画框
#
# 与 board_infer_raw.py 的区别：那份用 SD 卡里的预处理数组，无摄像头无屏幕；
# 这份走真实链路，屏幕上能看到画面和框。
#
# 关键：全过程写日志到 /sdcard/demo_log.txt 并 flush。
# 因为 MediaManager / 大块内存分配一旦让 USB 重新枚举，mpremote 的串口会失效，
# 终端输出全部丢失；有文件日志才能事后判断卡在哪一步。

import gc
import os
import sys
import time

import nncase_runtime as nn
import ulab.numpy as np
from media.sensor import *
from media.display import *
from media.media import *

KMODEL = "/sdcard/yolo11s_640_nchw.kmodel"
LOG = "/sdcard/demo_log.txt"
SENSOR_W, SENSOR_H = 640, 480
MODEL_W = MODEL_H = 640
TOP = (MODEL_H - SENSOR_H) // 2          # 80
BOT = MODEL_H - SENSOR_H - TOP           # 80
LEFT = (MODEL_W - SENSOR_W) // 2         # 0
RIGHT = MODEL_W - SENSOR_W - LEFT        # 0

CONF = 0.15        # 量化把分数压到 0.6x，故阈值按 0.25*0.6 折算
IOU_TH = 0.45
MAX_FRAMES = 20    # 0 = 一直跑

NAMES = ("wear", "thunder", "break")
COLORS = ((255, 220, 0), (0, 255, 0), (255, 60, 60))

_f = open(LOG, "w")


def log(msg):
    print(msg)
    _f.write(msg + "\n")
    _f.flush()


def mem(tag):
    gc.collect()
    log("    [mem] %s free=%d" % (tag, gc.mem_free()))


sensor = None
builder = None
try:
    log("### 1 Display.init")
    t0 = time.ticks_ms()
    Display.init(Display.ST7701, width=800, height=480, to_ide=True)
    log("    OK %d ms" % (time.ticks_ms() - t0))
    mem("display")

    log("### 2 MediaManager.init")
    MediaManager.init()
    log("    OK")
    mem("media")

    log("### 3 Sensor")
    sensor = Sensor(id=2)
    sensor.reset()
    sensor.set_framesize(width=SENSOR_W, height=SENSOR_H)
    sensor.set_pixformat(Sensor.RGB565)
    sensor.run()
    log("    OK")
    mem("sensor")

    log("### 4 load kmodel")
    t0 = time.ticks_ms()
    kpu = nn.kpu()
    kpu.load_kmodel(KMODEL)
    log("    OK %d ms  in=%s out=%s" % (time.ticks_ms() - t0,
                                        kpu.inputs_desc(0), kpu.outputs_desc(0)))
    mem("kmodel")

    # 官方 PipeLine 在 load 之后调这个，把 KPU 用剩的内存池还给系统，
    # 否则相机/显示那几块缓冲会抢不到内存
    try:
        nn.shrink_memory_pool()
        log("    shrink_memory_pool OK")
        mem("shrink")
    except Exception as e:
        log("    shrink_memory_pool 不支持: %s" % type(e).__name__)

    log("### 5 ai2d")
    ai2d = nn.ai2d()
    ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
    ai2d.set_pad_param(True, [0, 0, 0, 0, TOP, BOT, LEFT, RIGHT], 0, (114, 114, 114))
    ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    builder = ai2d.build([1, 3, SENSOR_H, SENSOR_W], [1, 3, MODEL_H, MODEL_W])
    log("    builder OK")
    mem("ai2d")

    # 官方写法：AI2D 的输出直接写进 KPU 自己的输入 tensor。
    # 不能用自己 np.zeros 的内存——那不是 nncase 内存池分配的物理缓冲，
    # KPU 读它属于非法地址，硬件层直接异常（上次两次挂死就是这个）。
    kpu_in = kpu.get_input_tensor(0)
    log("### 6 KPU 输入 tensor 就绪")
    mem("kpu_in")

    clock = time.clock()
    n = 0
    t_cap = t_a2d = t_kpu = t_dec = 0
    diag_done = False

    while True:
        os.exitpoint()
        clock.tick()
        n += 1

        t0 = time.ticks_ms()
        img = sensor.snapshot()
        t_cap += time.ticks_ms() - t0

        t0 = time.ticks_ms()
        builder.run(img, kpu_in)      # 官方签名：第二个参数是单个 tensor
        t_a2d += time.ticks_ms() - t0

        t0 = time.ticks_ms()
        kpu.run()                      # 输入已由 ai2d 写入，无需 set_input_tensor
        out = kpu.get_output_tensor(0).to_numpy()
        t_kpu += time.ticks_ms() - t0

        if not diag_done:
            diag_done = True
            got = kpu.get_input_tensor(0).to_numpy().reshape((3, MODEL_H * MODEL_W))
            for ci, nm in enumerate(("R", "G", "B")):
                ssum = 0
                ssum2 = 0
                cnt = 0
                for k in range(0, MODEL_H * MODEL_W, 1024):
                    v = int(got[ci, k])
                    ssum += v
                    ssum2 += v * v
                    cnt += 1
                mean = ssum / cnt
                log("    [首帧] 通道 %s 均值 %.1f 标准差 %.1f"
                    % (nm, mean, (ssum2 / cnt - mean * mean) ** 0.5))

        t0 = time.ticks_ms()
        y = out.reshape((7, 8400))
        sc = np.max(y[4:7], axis=0)
        cand = []
        flags = sc > CONF
        for i in range(8400):
            if flags[i]:
                s = float(sc[i])
                c = 0
                best = float(y[4, i])
                for k in (5, 6):
                    v = float(y[k, i])
                    if v > best:
                        best = v
                        c = k - 4
                cand.append((c, s, float(y[0, i]), float(y[1, i]),
                             float(y[2, i]), float(y[3, i])))
        cand.sort(key=lambda z: -z[1])
        keep = []
        for b in cand:
            dropout = False
            for k in keep:
                if b[0] != k[0]:
                    continue
                ax1 = b[2] - b[4] / 2; ax2 = b[2] + b[4] / 2
                ay1 = b[3] - b[5] / 2; ay2 = b[3] + b[5] / 2
                bx1 = k[2] - k[4] / 2; bx2 = k[2] + k[4] / 2
                by1 = k[3] - k[5] / 2; by2 = k[3] + k[5] / 2
                iw = min(ax2, bx2) - max(ax1, bx1)
                ih = min(ay2, by2) - max(ay1, by1)
                if iw > 0 and ih > 0:
                    inter = iw * ih
                    if inter / (b[4] * b[5] + k[4] * k[5] - inter) > IOU_TH:
                        dropout = True
                        break
            if not dropout:
                keep.append(b)
        t_dec += time.ticks_ms() - t0

        for c, s, cx, cy, w, h in keep:
            x = int(cx - w / 2)
            yy = int(cy - TOP - h / 2)
            if yy < -10 or yy > SENSOR_H:
                continue
            img.draw_rectangle(x, yy, int(w), int(h), color=COLORS[c], thickness=2)
            img.draw_string_advanced(x, max(yy - 18, 0), 16,
                                     "%s %.2f" % (NAMES[c], s), color=COLORS[c])

        img.draw_string_advanced(8, 8, 20, "%.1f fps  %d obj" % (clock.fps(), len(keep)),
                                 color=(0, 255, 0))
        Display.show_image(img, x=(800 - SENSOR_W) // 2, y=0)

        if n % 5 == 0:
            log("  帧 %3d  %.1f fps  框 %d  top1 %.3f  | 采集 %.0f  ai2d %.0f  kpu %.0f  解码 %.0f ms"
                % (n, clock.fps(), len(keep), float(np.max(sc)),
                   t_cap / n, t_a2d / n, t_kpu / n, t_dec / n))

        if MAX_FRAMES and n >= MAX_FRAMES:
            log("### 到 %d 帧，正常结束" % MAX_FRAMES)
            break

except BaseException as e:
    log("!!! 失败: %s: %s" % (type(e).__name__, e))
    try:
        sys.print_exception(e)
    except Exception:
        pass
finally:
    try:
        if builder is not None:
            builder.release()
    except Exception:
        pass
    try:
        if isinstance(sensor, Sensor):
            sensor.stop()
    except Exception:
        pass
    try:
        Display.deinit()
    except Exception:
        pass
    try:
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
        MediaManager.deinit()
    except Exception:
        pass
    log("资源已释放")
    _f.close()
