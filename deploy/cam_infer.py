# 不经过 ai2d 的摄像头推理：传感器直接出 640x640 RGB888，
# 用 np.frombuffer 取帧内存，喂给 NHWC 输入的 kmodel。
#
# 为什么能绕开 ai2d：传感器本身就支持 640x640，出来的缓冲
# 1,228,800 字节 = 1*640*640*3，和模型输入逐字节同构，不需要缩放/补边。

import gc
import time

import nncase_runtime as nn
import ulab.numpy as np
from media.media import MediaManager
from media.sensor import Sensor

KMODEL = "/sdcard/yolo11s_640_u8r.kmodel"      # NHWC uint8 输入
LOG = "/sdcard/cam_infer_log.txt"
S = 640
CONF = 0.10

_f = open(LOG, "w")


def log(msg):
    print(msg)
    _f.write(msg + "\n")
    _f.flush()


try:
    log("### 1 MediaManager + sensor")
    MediaManager.init()
    sensor = Sensor(id=2)
    sensor.reset()
    sensor.set_framesize(width=S, height=S)
    sensor.set_pixformat(Sensor.RGB888)
    sensor.run()
    log("    OK")

    log("### 2 载入 kmodel（NHWC）")
    t0 = time.ticks_ms()
    kpu = nn.kpu()
    kpu.load_kmodel(KMODEL)
    log("    OK %d ms  in=%s" % (time.ticks_ms() - t0, kpu.inputs_desc(0)))

    log("### 3 预热一帧")
    img = sensor.snapshot()
    arr = np.frombuffer(img, dtype=np.uint8)
    log("    帧长度 %d" % len(arr))
    t = nn.from_numpy(arr.reshape((1, S, S, 3)))
    kpu.set_input_tensor(0, t)
    kpu.run()
    out = kpu.get_output_tensor(0).to_numpy()
    log("    输出 %s  free=%d" % (str(out.shape), gc.mem_free()))

    log("### 4 连跑 10 帧")
    t_cap = 0
    t_cp = 0
    t_kpu = 0
    t_dec = 0
    n = 0
    for i in range(10):
        t0 = time.ticks_ms()
        img = sensor.snapshot()
        t_cap += time.ticks_ms() - t0

        t0 = time.ticks_ms()
        arr = np.frombuffer(img, dtype=np.uint8)
        t = nn.from_numpy(arr.reshape((1, S, S, 3)))
        kpu.set_input_tensor(0, t)
        t_cp += time.ticks_ms() - t0

        t0 = time.ticks_ms()
        kpu.run()
        out = kpu.get_output_tensor(0).to_numpy()
        t_kpu += time.ticks_ms() - t0

        t0 = time.ticks_ms()
        y = out.reshape((7, 8400))
        sc = np.max(y[4:7], axis=0)
        best = float(np.max(sc))
        idx = int(np.argmax(sc))
        cls = 0
        bv = float(y[4, idx])
        for k in (5, 6):
            if float(y[k, idx]) > bv:
                bv = float(y[k, idx])
                cls = k - 4
        hits = 0
        flags = sc > CONF
        for j in range(8400):
            if flags[j]:
                hits += 1
        t_dec += time.ticks_ms() - t0

        n += 1
        log("    帧 %2d top1 %.3f cls %d  cx %.0f cy %.0f  >%.2f 的 anchor %d  free %d"
            % (i, best, cls, float(y[0, idx]), float(y[1, idx]), CONF, hits,
               gc.mem_free()))

    log("### 5 平均耗时（10 帧）")
    log("    采集 %d ms  from_numpy %d ms  推理 %d ms  解码 %d ms"
        % (t_cap // n, t_cp // n, t_kpu // n, t_dec // n))
    log("=== 摄像头推理链路跑通 ===")

except BaseException as e:
    log("!!! 失败: %s: %s" % (type(e).__name__, e))
finally:
    try:
        sensor.stop()
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
    log("done")
    _f.close()
