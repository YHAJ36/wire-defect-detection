# 摄像头 → KPU 完整演示（不用 ai2d）：屏幕出画面 + 检测框 + fps
#
# 相对 cam_infer.py 的两处修正：
# 1) 输入复用同一块 ulab 缓冲与同一个 tensor，不再每帧新建（避免 nncase 内存池上涨）
# 2) 每帧用完显式 del 输出拷贝，并定期 gc.collect（实测泄漏约 287 KB/帧 ≈ 输出张量大小）

import gc
import time

import nncase_runtime as nn
import ulab.numpy as np
from media.display import Display
from media.media import MediaManager
from media.sensor import Sensor

KMODEL = "/sdcard/yolo11s_640_u8r.kmodel"
LOG = "/sdcard/cam_demo_log.txt"
S = 640
CONF = 0.15
IOU_TH = 0.45
MAX_FRAMES = 30

NAMES = ("wear", "thunder", "break")
COLORS = ((255, 220, 0), (0, 255, 0), (255, 60, 60))

_f = open(LOG, "w")


def log(msg):
    print(msg)
    _f.write(msg + "\n")
    _f.flush()


sensor = None
try:
    log("### 1 显示初始化")
    Display.init(Display.ST7701, width=800, height=480, to_ide=False)
    log("    OK  free=%d" % gc.mem_free())

    log("### 2 摄像头 640x640 RGB888")
    MediaManager.init()
    sensor = Sensor(id=2)
    sensor.reset()
    sensor.set_framesize(width=S, height=S)
    sensor.set_pixformat(Sensor.RGB888)
    sensor.run()
    log("    OK  free=%d" % gc.mem_free())

    log("### 3 kmodel")
    kpu = nn.kpu()
    kpu.load_kmodel(KMODEL)
    log("    OK  free=%d" % gc.mem_free())

    # 输入缓冲只建一次
    buf = np.zeros((1, S, S, 3), dtype=np.uint8)
    tin = nn.from_numpy(buf)
    log("    输入 tensor 就绪  free=%d" % gc.mem_free())

    clock = time.clock()
    n = 0
    t_cap = t_cp = t_kpu = t_dec = t_draw = 0
    first_test = True

    while True:
        n += 1
        clock.tick()

        t0 = time.ticks_ms()
        img = sensor.snapshot()
        t_cap += time.ticks_ms() - t0

        t0 = time.ticks_ms()
        src = np.frombuffer(img, dtype=np.uint8).reshape((S, S, 3))
        buf[0] = src
        kpu.set_input_tensor(0, tin)
        t_cp += time.ticks_ms() - t0

        t0 = time.ticks_ms()
        kpu.run()
        out = kpu.get_output_tensor(0).to_numpy()
        t_kpu += time.ticks_ms() - t0

        t0 = time.ticks_ms()
        y = out.reshape((7, 8400))
        sc = np.max(y[4:7], axis=0)
        idxs = []
        flags = sc > CONF
        for i in range(8400):
            if flags[i]:
                idxs.append(i)
        cand = []
        for i in idxs:
            s = float(sc[i])
            c = 0
            bv = float(y[4, i])
            for k in (5, 6):
                if float(y[k, i]) > bv:
                    bv = float(y[k, i])
                    c = k - 4
            cand.append((c, s, float(y[0, i]), float(y[1, i]),
                         float(y[2, i]), float(y[3, i])))
        del out
        cand.sort(key=lambda z: -z[1])
        keep = []
        for b in cand:
            drop = False
            for k in keep:
                if b[0] != k[0]:
                    continue
                ax1 = b[2] - b[4] / 2
                ax2 = b[2] + b[4] / 2
                ay1 = b[3] - b[5] / 2
                ay2 = b[3] + b[5] / 2
                bx1 = k[2] - k[4] / 2
                bx2 = k[2] + k[4] / 2
                by1 = k[3] - k[5] / 2
                by2 = k[3] + k[5] / 2
                iw = min(ax2, bx2) - max(ax1, bx1)
                ih = min(ay2, by2) - max(ay1, by1)
                if iw > 0 and ih > 0:
                    inter = iw * ih
                    if inter / (b[4] * b[5] + k[4] * k[5] - inter) > IOU_TH:
                        drop = True
                        break
            if not drop:
                keep.append(b)
        t_dec += time.ticks_ms() - t0

        t0 = time.ticks_ms()
        # 模型无 letterbox，坐标与传感器 1:1，直接画
        for c, s, cx, cy, w, h in keep:
            img.draw_rectangle(int(cx - w / 2), int(cy - h / 2),
                               int(w), int(h), color=COLORS[c], thickness=3)
            img.draw_string_advanced(int(cx - w / 2), max(int(cy - h / 2) - 20, 0),
                                     18, "%s %.2f" % (NAMES[c], s), color=COLORS[c])
        img.draw_string_advanced(8, 8, 20, "%.1f fps  %d obj  free %dKB"
                                 % (clock.fps(), len(keep), gc.mem_free() // 1024),
                                 color=(0, 255, 0))
        # 640x640 画面居中放进 800x480 屏，纵向超出部分裁掉
        Display.show_image(img, x=80, y=0)
        t_draw += time.ticks_ms() - t0

        if first_test:
            first_test = False
            log("    首帧完成：框 %d  top1 %.3f  free %d"
                % (len(keep), float(np.max(sc)), gc.mem_free()))

        if n % 5 == 0:
            gc.collect()
            log("  帧 %3d %.1ffps 框 %d | 采集 %d 拷贝 %d 推理 %d 解码 %d 绘制 %d ms | free %dKB"
                % (n, clock.fps(), len(keep), t_cap // n, t_cp // n,
                   t_kpu // n, t_dec // n, t_draw // n, gc.mem_free() // 1024))

        if MAX_FRAMES and n >= MAX_FRAMES:
            log("### 跑满 %d 帧，正常结束" % MAX_FRAMES)
            break

    del tin
    gc.collect()
    try:
        nn.shrink_memory_pool()
    except Exception:
        pass

except BaseException as e:
    log("!!! 失败: %s: %s" % (type(e).__name__, e))
finally:
    try:
        if sensor is not None:
            sensor.stop()
    except Exception:
        pass
    try:
        Display.deinit()
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
    log("done")
    _f.close()
