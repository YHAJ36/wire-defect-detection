# 探针：只验证「摄像头帧 → ai2d → NCHW uint8 [1,3,640,640]」这一段能不能通。
# 不做推理、不画框，只为把 API 不明处一次性暴露出来。
# 每步单独 try，任何一步失败都继续往下，方便一次运行看清全部兼容性。

import os
import sys
import time

import nncase_runtime as nn
import ulab.numpy as np
from media.sensor import *
from media.display import *
from media.media import *

SENSOR_W, SENSOR_H = 640, 480
MODEL_W, MODEL_H = 640, 640
TOP = (MODEL_H - SENSOR_H) // 2      # 80
BOT = MODEL_H - SENSOR_H - TOP       # 80
LEFT = (MODEL_W - SENSOR_W) // 2     # 0
RIGHT = MODEL_W - SENSOR_W - LEFT    # 0

sensor = None
try:
    print("### 1 Display.init")
    t0 = time.ticks_ms()
    Display.init(Display.ST7701, width=800, height=480, to_ide=True)
    print("    OK %d ms" % (time.ticks_ms() - t0))

    print("### 2 MediaManager.init")
    MediaManager.init()
    print("    OK")

    print("### 3 Sensor")
    sensor = Sensor(id=2)
    sensor.reset()
    sensor.set_framesize(width=SENSOR_W, height=SENSOR_H)
    sensor.set_pixformat(Sensor.RGB565)
    sensor.run()
    print("    OK  pixformat=%s" % sensor.get_pixformat())

    print("### 4 取一帧")
    t0 = time.ticks_ms()
    img = sensor.snapshot()
    print("    OK %d ms  size=%s format=%s" % (time.ticks_ms() - t0, img.size(), img.format()))

    print("### 5 ai2d 建对象")
    ai2d = nn.ai2d()
    print("    OK  attrs=%s" % [a for a in dir(ai2d) if not a.startswith("_")])

    print("### 6 set_dtype")
    ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
    print("    OK  NCHW/NCHW uint8/uint8")

    print("### 7 set_pad_param  (8 值: 0,0,0,0,top,bot,left,right)")
    ai2d.set_pad_param(True, [0, 0, 0, 0, TOP, BOT, LEFT, RIGHT], 0, (114, 114, 114))
    print("    OK")

    print("### 8 set_resize_param")
    ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    print("    OK  interp_method=%s interp_mode=%s"
          % (nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel))

    print("### 9 build")
    ai2d.build([1, 3, SENSOR_H, SENSOR_W], [1, 3, MODEL_H, MODEL_W])
    print("    OK")

    print("### 10 分配输出缓冲并 run")
    out = np.zeros((1, 3, MODEL_H, MODEL_W), dtype=np.uint8)
    t = nn.from_numpy(out)
    t0 = time.ticks_ms()
    r = ai2d.run(img, [t])
    print("    OK %d ms  ret=%s" % (time.ticks_ms() - t0, r))

    print("### 11 输出统计（应非全 0，且落在 0..255）")
    s = 0
    mn = 255
    mx = 0
    for i in range(0, out.size, 997):
        v = int(out.reshape(-1)[i])
        s += v
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    print("    抽样 %d 点  值域 [%d, %d]  均值 %.1f" % (out.size // 997 + 1, mn, mx, s / (out.size // 997 + 1)))

    print("### 12 放到屏幕上（应看到 640x640 带灰边的画面）")
    big = img.resize(640, 640) if hasattr(img, "resize") else None
    Display.show_image(img, x=0, y=0)
    print("    show_image OK")

    print("### 13 连续 30 帧 ai2d 耗时")
    t0 = time.ticks_ms()
    for i in range(30):
        img = sensor.snapshot()
        ai2d.run(img, [t])
    el = time.ticks_ms() - t0
    print("    30 帧 %.0f ms  -> %.1f fps（采集+转换）" % (el, 30000.0 / el))

    print("=== 探针全部通过 ===")

except BaseException as e:
    print("!!! 失败: %s: %s" % (type(e).__name__, e))
    sys.print_exception(e)
finally:
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
    print("资源已释放")
