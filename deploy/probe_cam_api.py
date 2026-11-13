# 零风险 API 探测：只问「有什么方法」，不做任何大运算。
# 目的：确认能否绕开 ai2d——直接把摄像头帧变成 numpy 喂给 KPU。
# 前提：需要一份 NHWC 输入的 kmodel（/sdcard/yolo11s_640_u8r.kmodel）。

import gc

import ulab.numpy as np

LOG = "/sdcard/cam_api_log.txt"
_f = open(LOG, "w")


def log(msg):
    print(msg)
    _f.write(msg + "\n")
    _f.flush()


def names(o):
    out = []
    for a in dir(o):
        if a.startswith("_"):
            continue
        out.append(a)
    return out


try:
    import image
    log("### image.Image 方法")
    log("    " + " ".join(names(image.Image)))

    for m in ("to_numpy", "bytearray", "tobytes", "to_bytes", "resize",
              "copy", "crop", "draw_image", "compress", "to_rgb888",
              "to_rgb565", "get_pixel", "set_pixel", "width", "height",
              "format", "size"):
        log("    hasattr %-12s %s" % (m, hasattr(image.Image, m)))

    log("### image 模块常量")
    log("    " + " ".join([a for a in names(image) if a.isupper()]))

    from media.sensor import Sensor
    log("### Sensor 大写常量（像素格式/尺寸）")
    log("    " + " ".join([a for a in names(Sensor) if a.isupper()]))

    log("### 试 RGB888 摄像头帧")
    from media.media import MediaManager
    MediaManager.init()
    sensor = Sensor(id=2)
    sensor.reset()
    sensor.set_framesize(width=640, height=480)
    sensor.set_pixformat(Sensor.RGB888)
    sensor.run()
    log("    RGB888 OK, pixformat=%s" % sensor.get_pixformat())
    img = sensor.snapshot()
    log("    snapshot OK size=%s len=%d" % (str(img.size()), len(img)))

    log("### 试 to_numpy")
    try:
        arr = img.to_numpy()
        log("    to_numpy OK shape=%s dtype=%s" % (str(arr.shape), str(arr.dtype)))
        log("    前 8 值 %s" % str([int(arr[0, 0, i]) for i in range(3)]))
    except Exception as e:
        log("    to_numpy 失败 %s: %s" % (type(e).__name__, e))

    log("### 试 from_numpy 直喂（NHWC kmodel）")
    try:
        arr = img.to_numpy()
        t = nn_t = None
        import nncase_runtime as nn
        kpu = nn.kpu()
        kpu.load_kmodel("/sdcard/yolo11s_640_u8r.kmodel")
        flat = np.frombuffer(arr, dtype=np.uint8)
        log("    flat len=%d（640*480*3=%d）" % (len(flat), 640 * 480 * 3))
        t = nn.from_numpy(flat.reshape((1, 480, 640, 3)))
        kpu.set_input_tensor(0, t)
        kpu.run()
        out = kpu.get_output_tensor(0).to_numpy()
        log("    kpu.run OK out=%s" % str(out.shape))
        y = out.reshape((7, 8400))
        sc = np.max(y[4:7], axis=0)
        log("    top1=%.4f" % float(np.max(sc)))
    except Exception as e:
        log("    直喂失败 %s: %s" % (type(e).__name__, e))

    try:
        sensor.stop()
    except Exception:
        pass
    MediaManager.deinit()
    log("=== 探测结束 ===")
except BaseException as e:
    log("!!! 失败: %s: %s" % (type(e).__name__, e))
finally:
    log("done")
    _f.close()
