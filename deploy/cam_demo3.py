# 摄像头演示 v3：nano 模型 + 快速解码 + 时域确认
#
# 相对 v2 的改动
#   1. 换 yolo11n（推理约为 s 的 1/3）
#   2. 解码不再遍历 8400 个 anchor，改用 argmax 取 top-K（54 ms -> 个位数）
#   3. 类名按 data.yaml 修正：0=break 1=thunderbolt 2=wear（v2 的顺序是错的）
#   4. 时域确认：连续 2 帧命中同一位置才画框，压掉闪一下的误检
#   5. 补边拷贝改走 memoryview，byte 级 memcpy

import gc
import time

import nncase_runtime as nn
import ulab.numpy as np
from media.display import Display
from media.media import MediaManager
from media.sensor import Sensor

KMODEL = "/sdcard/yolo11n_640_u8r.kmodel"
LOG = "/sdcard/cam_demo3_log.txt"
S = 640
W, H = 640, 480
TOP = (S - H) // 2                 # 80
# 分数标定（2026-11-02 实测，见 calib_scores.py）：8 张图 640 个 anchor 配对，
# 板端分 / PC 分 = 0.927（中位），PC>0.5 时 0.969，PC>0.7 时 0.979，
# |Δ| 中位 0.033、P90 0.059 —— nano 这个 kmodel 几乎不压分数。
# 注意：0.605× 是 s 版（11 MB）的数字，不能套到 nano 上（A.9.84 已更正）。
SCORE_GAIN = 0.927
CONF = 0.28                        # 等效 PC 约 0.30
IOU_TH = 0.45
TOPK = 8
HITS_NEED = 2                      # 连续命中几帧才画
MAX_FRAMES = 0

NAMES = ("break", "thunder", "wear")
COLORS = ((255, 60, 60), (0, 255, 0), (255, 220, 0))

_f = open(LOG, "w")


def log(msg):
    print(msg)
    _f.write(msg + "\n")
    _f.flush()


def iou(a, b):
    ax1 = a[2] - a[4] / 2
    ax2 = a[2] + a[4] / 2
    ay1 = a[3] - a[5] / 2
    ay2 = a[3] + a[5] / 2
    bx1 = b[2] - b[4] / 2
    bx2 = b[2] + b[4] / 2
    by1 = b[3] - b[5] / 2
    by2 = b[3] + b[5] / 2
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[4] * a[5] + b[4] * b[5] - inter)


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

    log("### 3 kmodel (nano)")
    kpu = nn.kpu()
    kpu.load_kmodel(KMODEL)
    log("    OK free=%d" % gc.mem_free())

    log("### 4 输入缓冲 bytearray + memoryview")
    buf = bytearray(S * S * 3)
    mv = memoryview(buf)
    pad = b"\x72" * (TOP * S * 3)          # 0x72 = 114，与训练 letterbox 一致
    mv[0:len(pad)] = pad
    mv[(TOP + H) * S * 3:] = pad
    log("    就绪 free=%d" % gc.mem_free())

    clock = time.clock()
    n = 0
    t_cap = t_cp = t_kpu = t_dec = t_draw = 0
    prev = []                              # [(c,cx,cy,w,h,hits)]
    n_conf = 0

    while True:
        n += 1
        clock.tick()

        t0 = time.ticks_ms()
        img = sensor.snapshot()
        t_cap += time.ticks_ms() - t0

        t0 = time.ticks_ms()
        mv[TOP * S * 3:(TOP + H) * S * 3] = memoryview(img)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((1, S, S, 3))
        tin = nn.from_numpy(arr)
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
        cand = []
        for _ in range(TOPK):
            i = int(np.argmax(sc))
            v = float(sc[i])
            if v <= CONF:
                break
            s0 = float(y[4, i])
            s1 = float(y[5, i])
            s2 = float(y[6, i])
            c = 0
            if s1 > s0:
                c = 1
                s0 = s1
            if s2 > s0:
                c = 2
            cand.append((c, v, float(y[0, i]), float(y[1, i]),
                         float(y[2, i]), float(y[3, i])))
            sc[i] = -1.0
        del out
        cand.sort(key=lambda z: -z[1])

        keep = []
        for b in cand:
            dup = False
            for k in keep:
                if b[0] == k[0] and iou(b, k) > IOU_TH:
                    dup = True
                    break
            if not dup:
                keep.append(b)

        # 时域确认：跟上一帧同类别、同位置（IoU>0.3）配对的，命中数 +1
        track = []
        used = [False] * len(prev)
        for b in keep:
            hits = 1
            best = 0.3
            bi = -1
            for pi, p in enumerate(prev):
                if p[0] != b[0]:
                    continue
                v = iou(b, p)
                if v > best:
                    best = v
                    hits = p[6] + 1
                    bi = pi
            if bi >= 0:
                used[bi] = True
            track.append((b[0], b[1], b[2], b[3], b[4], b[5], hits))
        # 残留：上一帧有、这一帧没配上的保留 1 帧再消失，避免闪烁
        for pi, p in enumerate(prev):
            if (not used[pi]) and p[6] > 1:
                track.append((p[0], p[1], p[2], p[3], p[4], p[5], p[6] - 1))
        prev = track
        t_dec += time.ticks_ms() - t0

        t0 = time.ticks_ms()
        drawn = 0
        for c, scv, cx, cy, w, h, hits in track:
            if hits < HITS_NEED:
                continue
            x = int(cx - w / 2)
            yy = int(cy - TOP - h / 2)
            if yy < -30 or yy > H:
                continue
            img.draw_rectangle(x, yy, int(w), int(h), color=COLORS[c], thickness=3)
            img.draw_string_advanced(x, max(yy - 20, 0), 18,
                                     "%s %d%% x%d" % (NAMES[c], int(scv * 100), hits),
                                     color=COLORS[c])
            drawn += 1
        n_conf = drawn
        img.draw_string_advanced(8, 8, 20, "%.1f fps %d obj %dKB"
                                 % (clock.fps(), n_conf, gc.mem_free() // 1024),
                                 color=(0, 255, 0))
        Display.show_image(img, x=80, y=0)
        t_draw += time.ticks_ms() - t0

        if n % 10 == 0:
            gc.collect()
            log("  帧 %3d %.1ffps 框 %d top1 %.3f(PC≈%.3f) | 采集 %d 拷贝 %d 推理 %d 解码 %d 绘制 %d ms | free %dKB"
                % (n, clock.fps(), n_conf, top, min(top / SCORE_GAIN, 1.0),
                   t_cap // n, t_cp // n, t_kpu // n, t_dec // n,
                   t_draw // n, gc.mem_free() // 1024))

        if MAX_FRAMES and n >= MAX_FRAMES:
            break

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
