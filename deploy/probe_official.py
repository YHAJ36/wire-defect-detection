# 按官方签名重测 ai2d，分两级，风险递增，每级先写日志再执行。
#
# 官方文档（nncase_runtime 使用说明）原文写法：
#     kpu_input = kpu.get_input_tensor(0)
#     ai2d_builder.run(ai2d_input, kpu_input)
# 即第二个参数是「单个 tensor」，且应当是 KPU 自己的输入 tensor，
# 不是自己 np.zeros 出来的普通内存。
#
# 之前 board_demo.py 写的是 builder.run(img, [inp_t])：传了 list，
# 且输出用了自己分配的内存。两处都与官方不符，因此先做隔离测试。
#
# 级别 A：ai2d 输入也用静态 tensor（不碰摄像头、不碰屏幕）—— 只验签名
# 级别 B：ai2d 输入换成 sensor.snapshot()（仍不碰屏幕）
#
# 结果全部写入 /sdcard/probe_log.txt，flush 到磁盘，USB 掉了也能读回来。

import gc
import os
import time

import nncase_runtime as nn
import ulab.numpy as np

KMODEL = "/sdcard/yolo11s_640_nchw.kmodel"
LOG = "/sdcard/probe_log.txt"
SENSOR_W, SENSOR_H = 640, 480
MODEL_W = MODEL_H = 640

_f = open(LOG, "w")


def log(msg):
    print(msg)
    _f.write(msg + "\n")
    _f.flush()


def mem(tag):
    gc.collect()
    log("    [mem] %-10s free=%d" % (tag, gc.mem_free()))


kpu = None
ai2d = None
builder = None
sensor = None
try:
    log("### A1 加载 kmodel")
    t0 = time.ticks_ms()
    kpu = nn.kpu()
    kpu.load_kmodel(KMODEL)
    log("    OK %d ms" % (time.ticks_ms() - t0))
    log("    in desc : %s" % str(kpu.inputs_desc(0)))
    mem("kmodel")

    log("### A2 取 KPU 输入 tensor（官方写法，让 nncase 分配）")
    kpu_in = kpu.get_input_tensor(0)
    log("    OK type=%s" % type(kpu_in))
    try:
        log("    shape=%s" % str(kpu_in.shape))
    except Exception as e:
        log("    shape 读取失败 %s" % type(e).__name__)
    mem("kpu_in")

    log("### A3 配置 ai2d")
    ai2d = nn.ai2d()
    ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT,
                   np.uint8, np.uint8)
    ai2d.set_pad_param(True, [0, 0, 0, 0, 80, 80, 0, 0], 0, (114, 114, 114))
    ai2d.set_resize_param(True, nn.interp_method.tf_bilinear,
                          nn.interp_mode.half_pixel)
    builder = ai2d.build([1, 3, SENSOR_H, SENSOR_W], [1, 3, MODEL_H, MODEL_W])
    log("    builder OK")
    mem("ai2d")

    log("### A4 静态输入 + 官方签名 run(in, kpu_in)")
    static_in = nn.from_numpy(
        np.zeros((1, 3, SENSOR_H, SENSOR_W), dtype=np.uint8))
    t0 = time.ticks_ms()
    builder.run(static_in, kpu_in)
    log("    A4 OK %d ms" % (time.ticks_ms() - t0))

    log("### A5 kpu.run")
    t0 = time.ticks_ms()
    kpu.run()
    out = kpu.get_output_tensor(0).to_numpy()
    log("    A5 OK %d ms  out shape=%s" % (time.ticks_ms() - t0, str(out.shape)))
    y = out.reshape((7, 8400))
    sc = np.max(y[4:7], axis=0)
    log("    top1=%.4f（全零输入，只验不崩，不看数值）" % float(np.max(sc)))
    mem("after run")

    log("### B1 启动摄像头")
    from media.sensor import Sensor
    from media.media import MediaManager
    MediaManager.init()
    sensor = Sensor(id=2)
    sensor.reset()
    sensor.set_framesize(width=SENSOR_W, height=SENSOR_H)
    sensor.set_pixformat(Sensor.RGB565)
    sensor.run()
    log("    OK pixformat=%s" % sensor.get_pixformat())
    mem("sensor")

    log("### B2 snapshot")
    img = sensor.snapshot()
    log("    OK size=%s" % str(img.size()))

    log("### B3 builder.run(摄像头帧, kpu_in)")
    t0 = time.ticks_ms()
    builder.run(img, kpu_in)
    log("    B3 OK %d ms" % (time.ticks_ms() - t0))

    log("### B4 kpu.run")
    t0 = time.ticks_ms()
    kpu.run()
    out = kpu.get_output_tensor(0).to_numpy()
    log("    B4 OK %d ms" % (time.ticks_ms() - t0))
    y = out.reshape((7, 8400))
    sc = np.max(y[4:7], axis=0)
    hits = 0
    flags = sc > 0.15
    for i in range(8400):
        if flags[i]:
            hits += 1
    log("    top1=%.4f  >0.15 的 anchor=%d" % (float(np.max(sc)), hits))
    log("=== 全部通过 ===")

except BaseException as e:
    log("!!! 失败: %s: %s" % (type(e).__name__, e))
finally:
    log("finally")
    try:
        if sensor is not None:
            sensor.stop()
    except Exception:
        pass
    try:
        if builder is not None:
            builder.release()
    except Exception:
        pass
    try:
        if ai2d is not None:
            ai2d.release()
    except Exception:
        pass
    try:
        if kpu is not None:
            kpu.release()
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
    log("done")
    _f.close()
