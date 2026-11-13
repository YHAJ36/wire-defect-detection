# main.py —— 脱机上电自动运行的演示程序
#
# 用你自己转换出来的 kmodel（float32 输入那个），摄像头取画面，屏上实时画框。
#
# 三种跑法，从安全到"高级"：
#   1. 在线跑（脚本留在 PC，板子不存）：
#        python -m mpremote connect COM8 run D:/2/manual/main_demo.py
#   2. 传上去手动跑（板上有一份，但不开机自启）：
#        python -m mpremote connect COM8 fs cp D:/2/manual/main_demo.py :/sdcard/demo.py
#   3. 脱机自启（上电就跑）：把本文件传到 /sdcard/main.py
#        python -m mpremote connect COM8 fs cp D:/2/manual/main_demo.py :/sdcard/main.py
#      然后拔线、按复位，屏幕应该自己亮起来出画面。
#
# 板子的开机顺序（官方机制，不是猜的）：
#   上电 -> 执行 /sdcard/boot.py（没有就跳过）-> 执行 /sdcard/main.py
#   所以"改名 main.py 就能脱机"是对的，但有两个前提：
#     · 必须放在 /sdcard 根目录，不能放子目录
#     · 它调用的 kmodel 也必须在板上，路径要对
#
# 脱机时你看不到串口输出，所以：
#   · 屏幕上直接显示 fps / 目标数 / 分数 / 剩余内存
#   · 同时写一份 /sdcard/main_log.txt，出问题拔下卡插电脑就能看
#   · 想中途停下：接上串口，在 REPL 里按 Ctrl-C
#
# 安全：不刷固件、不写 flash，只写 /sdcard。Ctrl-C 随时能中断。

import gc
import time

import nncase_runtime as nn
import ulab.numpy as np
from media.display import Display
from media.media import MediaManager
from media.sensor import Sensor

KMODEL = "/sdcard/yolo11n_640_manual.kmodel"
LOG = "/sdcard/main_log.txt"

S = 640                 # 模型输入边长
W, H = 640, 480         # 摄像头出图尺寸（面板只有 480 高，不能直接给 640×640）
TOP = (S - H) // 2      # 上下各补 80 行灰边，凑成 640×640

CONF = 0.30             # 阈值：这个 kmodel 分数几乎不压（0.99×），可以直接用正常值
IOU_TH = 0.45
TOPK = 8
HITS_NEED = 2           # 连续 2 帧都命中的才画，压掉闪一下的误检
MAX_FRAMES = 0          # 0 = 一直跑；调试时可设成 40 自动停

NAMES = ("break", "thunderbolt", "wear")
COLORS = ((255, 60, 60), (0, 255, 0), (255, 220, 0))

# ulab 各固件对 float32 的叫法不统一：有的是 np.float32，有的只有 np.float
FT = getattr(np, "float32", None) or getattr(np, "float")

_f = open(LOG, "w")


def log(msg):
    print(msg)                 # 在线时看串口
    _f.write(msg + "\n")       # 脱机时看这个文件
    _f.flush()


def iou(a, b):
    """两个框的交并比。a/b 都是 (类别, 分数, cx, cy, w, h...)，取后 4 位。"""
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


def frame_to_input(mv, buf, fa):
    """把摄像头那帧（uint8，通道在后）搬进模型的输入（float32，通道在前）。

    这是 float 版 kmodel 比 uint8 版多付的代价，实测约 150 ms/帧。
    三件事：除 255 变小树、通道顺序从 HWC 换成 CHW、最外面套一层 batch。
    """
    u = np.frombuffer(buf, dtype=np.uint8).reshape((S, S, 3))
    for c in range(3):
        fa[0, c] = np.array(u[:, :, c], dtype=FT) / 255.0


sensor = None
try:
    log("### 1 显示 800x480")
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

    log("### 4 缓冲区")
    buf = bytearray(S * S * 3)                  # 补好边的 uint8 整帧
    mv = memoryview(buf)
    pad = b"\x72" * (TOP * S * 3)               # 0x72 = 114，和训练时 letterbox 的灰一致
    mv[0:len(pad)] = pad
    mv[(TOP + H) * S * 3:] = pad
    fa = np.zeros((1, 3, S, S), dtype=FT)       # 模型的输入，循环外只建一次
    log("    就绪 free=%d" % gc.mem_free())

    clock = time.clock()
    n = 0
    t_cap = t_conv = t_kpu = t_dec = 0
    prev = []

    while True:
        n += 1
        clock.tick()

        t0 = time.ticks_ms()
        img = sensor.snapshot()
        t_cap += time.ticks_ms() - t0

        # 把 480 行的画面拷到 640 块的正中间，上下留灰边
        mv[TOP * S * 3:(TOP + H) * S * 3] = memoryview(img)
        t0 = time.ticks_ms()
        frame_to_input(mv, buf, fa)
        kpu.set_input_tensor(0, nn.from_numpy(fa))
        t_conv += time.ticks_ms() - t0

        t1 = time.ticks_ms()
        kpu.run()
        out = kpu.get_output_tensor(0).to_numpy()
        t_kpu += time.ticks_ms() - t1

        t1 = time.ticks_ms()
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

        # 同类别的重叠框只留一个
        keep = []
        for b in cand:
            dup = False
            for k in keep:
                if b[0] == k[0] and iou(b, k) > IOU_TH:
                    dup = True
                    break
            if not dup:
                keep.append(b)

        # 时域确认：和上一帧位置接近的，命中数 +1
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
        for pi, p in enumerate(prev):
            if (not used[pi]) and p[6] > 1:
                track.append((p[0], p[1], p[2], p[3], p[4], p[5], p[6] - 1))
        prev = track
        t_dec += time.ticks_ms() - t1

        # 画框：框坐标是 640 空间的，减掉 TOP 才是屏幕上的位置
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
                                     "%s %d%%" % (NAMES[c], int(scv * 100)),
                                     color=COLORS[c])
            drawn += 1

        img.draw_string_advanced(8, 8, 20,
                                 "%.1f fps  %d obj  top %.2f  %dKB"
                                 % (clock.fps(), drawn, top, gc.mem_free() // 1024),
                                 color=(0, 255, 0))
        Display.show_image(img, x=80, y=0)

        if n % 20 == 0:
            gc.collect()
            log("  帧 %3d %.1ffps 框 %d top %.3f | 采集 %d 转格式 %d 推理 %d 解码 %d ms | free %dKB"
                % (n, clock.fps(), drawn, top, t_cap // n, t_conv // n,
                   t_kpu // n, t_dec // n, gc.mem_free() // 1024))

        if MAX_FRAMES and n >= MAX_FRAMES:
            break

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
        Display.deinit()          # 关屏，不然会一直停在最后一帧
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
    log("done")
    _f.close()
