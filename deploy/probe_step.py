import os, sys, time
import nncase_runtime as nn
import ulab.numpy as np
from media.sensor import *
from media.display import *
from media.media import *

LOG = "/sdcard/step_log.txt"
_f = open(LOG, "w")
def L(m):
    _f.write(m + "\n"); _f.flush()

def bail(tag, e):
    L("%s EXC %s: %s" % (tag, type(e).__name__, e))

sensor = None
builder = None
try:
    L("A display"); Display.init(Display.ST7701, width=800, height=480, to_ide=True)
    L("A media"); MediaManager.init()
    L("B sensor create")
    sensor = Sensor(id=2); sensor.reset()
    sensor.set_framesize(width=640, height=480)
    L("B pixfmt RGB565")
    sensor.set_pixformat(Sensor.RGB565)
    L("B run")
    sensor.run()
    L("B ok pixformat=%s" % sensor.get_pixformat())

    L("C snapshot ...")
    t0 = time.ticks_ms()
    img = sensor.snapshot()
    L("C snapshot ok %d ms size=%s" % (time.ticks_ms() - t0, img.size()))

    L("D kpu load")
    kpu = nn.kpu(); kpu.load_kmodel("/sdcard/yolo11s_640_nchw.kmodel")
    L("D ok")

    L("E ai2d build")
    a = nn.ai2d()
    a.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
    a.set_pad_param(True, [0,0,0,0,80,80,0,0], 0, (114,114,114))
    a.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    b = a.build([1,3,480,640], [1,3,640,640])
    L("E build ok")

    L("F alloc out")
    buf = np.zeros((1,3,640,640), dtype=np.uint8)
    t = nn.from_numpy(buf)
    L("F ok")

    L("G builder.run(img,[t]) ...")
    b.run(img, [t])
    L("G ok")

    L("H kpu.set_input_tensor")
    kpu.set_input_tensor(0, t)
    L("H ok")
    L("I kpu.run ...")
    kpu.run()
    L("I ok")
    out = kpu.get_output_tensor(0).to_numpy()
    L("J out shape=%s" % str(out.shape))
    y = out.reshape((7,8400))
    L("K top1=%.3f" % float(np.max(y[4:7])))
    L("=== 全部通过 ===")
except BaseException as e:
    bail("MAIN", e)
finally:
    L("finally")
    try:
        if builder is not None: builder.release()
    except Exception: pass
    try:
        if isinstance(sensor, Sensor): sensor.stop()
    except Exception: pass
    try: Display.deinit()
    except Exception: pass
    try:
        MediaManager.deinit()
    except Exception: pass
    L("done")
    _f.close()
