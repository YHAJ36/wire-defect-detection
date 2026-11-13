# 找「把摄像头帧变成 ulab 数组」的通道，以及能否让传感器直接出 640x640。
# 全程不碰 ai2d、不做大运算，最坏只是抛 Python 异常。

import gc

import ulab.numpy as np

LOG = "/sdcard/pix_log.txt"
_f = open(LOG, "w")


def log(msg):
    print(msg)
    _f.write(msg + "\n")
    _f.flush()


try:
    from media.media import MediaManager
    from media.sensor import Sensor

    MediaManager.init()
    sensor = Sensor(id=2)
    sensor.reset()

    log("### 1 传感器能否直接出 640x640（省掉缩放与补边）")
    ok = False
    try:
        sensor.set_framesize(width=640, height=640)
        ok = True
        log("    set_framesize(640,640) 接受")
    except Exception as e:
        log("    set_framesize(640,640) 拒绝 %s: %s" % (type(e).__name__, e))
    if ok:
        try:
            sensor.set_pixformat(Sensor.RGB888)
            sensor.run()
            img = sensor.snapshot()
            log("    RGB888 640x640 → size=%d（期望 1228800）" % img.size())
        except Exception as e:
            log("    RGB888 640x640 失败 %s: %s" % (type(e).__name__, e))
    else:
        sensor.set_framesize(width=640, height=480)
        sensor.set_pixformat(Sensor.RGB888)
        sensor.run()
        img = sensor.snapshot()
        log("    退回 640x480，size=%d" % img.size())

    log("### 2 帧对象有哪些方法")
    log("    " + " ".join([a for a in dir(img) if not a.startswith("_")]))

    log("### 3 能否用 frombuffer 直接读帧（缓冲区协议）")
    try:
        flat = np.frombuffer(img, dtype=np.uint8)
        log("    frombuffer OK len=%d 首值=%s" %
            (len(flat), str([int(flat[i]) for i in range(4)])))
    except Exception as e:
        log("    frombuffer 失败 %s: %s" % (type(e).__name__, e))

    log("### 4 其它可能的取像素通道")
    for m in ("to_numpy", "bytearray", "to_bytes", "tobytes", "get_bytes",
              "compress", "save", "to_rgb888"):
        if hasattr(img, m):
            log("    存在 %s" % m)
    try:
        v = img.get_pixel(320, 240)
        log("    get_pixel(320,240) = %s" % str(v))
    except Exception as e:
        log("    get_pixel 失败 %s: %s" % (type(e).__name__, e))
    try:
        log("    len(img)=%d  img[0]=%s" % (len(img), type(img[0])))
    except Exception as e:
        log("    img[0] 失败 %s: %s" % (type(e).__name__, e))

    try:
        sensor.stop()
    except Exception:
        pass
    MediaManager.deinit()
    log("=== 结束 ===")
except BaseException as e:
    log("!!! 失败: %s: %s" % (type(e).__name__, e))
finally:
    log("done")
    _f.close()
