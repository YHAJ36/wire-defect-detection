# 摄像头演示 v2：传感器 640x480（面板装得下），软件补边到 640x640 再推理。
# 补边方式与训练时一致（上下各 80 行填 114），所以框坐标减去 80 就回到屏幕坐标。

import gc
import time

import nncase_runtime as nn
import ulab.numpy as np
from media.display import Display
from media.media import MediaManager
from media.sensor import Sensor

KMODEL = "/sdcard/yolo11s_640_u8r.kmodel"
LOG = "/sdcard/cam_demo2_log.txt"
S = 640
W, H = 640, 480
TOP = (S - H) // 2                 # 80
CONF = 0.15
IOU_TH = 0.45
MAX_FRAMES = 0

NAMES = ("wear", "thunder", "break")
COLORS = ((255, 220, 0), (0, 255, 0), (255, 60, 60))

_f = open(LOG, "w")


def log(msg):
    print(msg)
    _f.write(msg + "\n")
    _f.flush()


sensor = None
try:
    log("### 1 显示")
    Display.init(Display.ST7701, width=800, height=480, to_ide=False)
    log("    OK free=%d" % gc.mem_free())

    log("### 2 摄像头 640x480 RGB888")
    MediaManager.init()
    sensor = Sensor(id=2)
    sensor.reset()
    sensor.set_framesize(width=W, height=H)
    sensor.set_pixformat(Sensor.RGB888)
    sensor.run()
    log("    OK free=%d" % gc.mem_free())

    log("### 3 kmodel")
    kpu = nn.kpu()
    kpu.load_kmodel(KMODEL)
    log("    OK free=%d" % gc.mem_free())

    log("### 4 输入缓冲（含补边）")
    buf = np.zeros((1, S, S, 3), dtype=np.uint8)
    # 上下补边一次性填 114
    padrow = np.ones((1, 1, S, 3), dtype=np.uint8) * 114
    buf[0, 0:TOP] = padrow[0, 0]
    buf[0, TOP + H:S] = padrow[0, 0]
    # 注意：nn.from_numpy 是「拷贝」而非共享内存，所以不能在循环外用一次。
    # 每帧写完 buf 必须重新 from_numpy，否则 KPU 永远读创建那一刻的内容。
    log("    就绪 free=%d" % gc.mem_free())

    # 先验证切片写入是否生效（写一块已知值再读回）
    try:
        buf[0, TOP, 0, 0] = 7
        log("    切片写入测试 buf[0,%d,0,0]=%d" % (TOP, int(buf[0, TOP, 0, 0])))
    except Exception as e:
        log("    切片写入测试失败 %s: %s" % (type(e).__name__, e))

    clock = time.clock()
    n = 0
    t_cap = t_cp = t_kpu = t_dec = t_draw = 0
    prev_top = -1.0
    changed = 0

    while True:
        n += 1
        clock.tick()

        t0 = time.ticks_ms()
        img = sensor.snapshot()
        t_cap += time.ticks_ms() - t0

        t0 = time.ticks_ms()
        src = np.frombuffer(img, dtype=np.uint8).reshape((H, W, 3))
        buf[0, TOP:TOP + H] = src
        tin = nn.from_numpy(buf)
        kpu.set_input_tensor(0, tin)
        t_cp += time.ticks_ms() - t0

        t0 = time.ticks_ms()
        kpu.run()
        out = kpu.get_output_tensor(0).to_numpy()
        t_kpu += time.ticks_ms() - t0

        t0 = time.ticks_ms()
        y = out.reshape((7, 8400))
        sc = np.max(y[4:7], axis=0)
        top = float(np.max(sc))
        if abs(top - prev_top) > 1e-4:
            changed += 1
        prev_top = top

        cand = []
        flags = sc > CONF
        for i in range(8400):
            if flags[i]:
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
        for c, s, cx, cy, w, h in keep:
            x = int(cx - w / 2)
            yy = int(cy - TOP - h / 2)
            if yy < -30 or yy > H:
                continue
            img.draw_rectangle(x, yy, int(w), int(h), color=COLORS[c], thickness=3)
            img.draw_string_advanced(x, max(yy - 20, 0), 18,
                                     "%s %.2f" % (NAMES[c], s), color=COLORS[c])
        img.draw_string_advanced(8, 8, 20, "%.1f fps %d obj %dKB"
                                 % (clock.fps(), len(keep), gc.mem_free() // 1024),
                                 color=(0, 255, 0))
        Display.show_image(img, x=80, y=0)
        t_draw += time.ticks_ms() - t0

        if n % 5 == 0:
            gc.collect()
            log("  帧 %3d %.1ffps 框 %d top1 %.3f | 采集 %d 拷贝 %d 推理 %d 解码 %d 绘制 %d ms | free %dKB"
                % (n, clock.fps(), len(keep), top, t_cap // n, t_cp // n,
                   t_kpu // n, t_dec // n, t_draw // n, gc.mem_free() // 1024))

        if MAX_FRAMES and n >= MAX_FRAMES:
            log("### 跑满 %d 帧，top1 变动次数 %d/%d" % (MAX_FRAMES, changed, n))
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
