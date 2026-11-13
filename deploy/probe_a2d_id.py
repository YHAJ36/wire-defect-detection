# ai2d 分阶段隔离：从最小可用（恒等变换）开始，一次只加一个功能。
# 崩在哪一阶段，日志就停在哪一行——板子会自己重启，USB 会重新枚举，日志在 SD 卡上不丢。
#
# S1：ai2d 恒等（640x640 -> 640x640，不 pad 不 resize，只验「能不能跑起来」
# S2：只加 pad（480x640 -> 640x640，上下各 80）
# S3：pad + resize（480x640 -> 640x640 完整链路）

import gc
import time

import nncase_runtime as nn
import ulab.numpy as np

KMODEL = "/sdcard/yolo11s_640_nchw.kmodel"
LOG = "/sdcard/a2d_id_log.txt"

_f = open(LOG, "w")


def log(msg):
    print(msg)
    _f.write(msg + "\n")
    _f.flush()


def stage(name, src_hw, pad, resize):
    """返回 True = 该阶段跑通"""
    log("### %s  build %s -> 640x640  pad=%s resize=%s"
        % (name, list(src_hw), pad, resize))
    a = nn.ai2d()
    a.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT,
                np.uint8, np.uint8)
    if pad is not None:
        a.set_pad_param(True, pad, 0, (114, 114, 114))
        log("    set_pad_param OK")
    if resize:
        a.set_resize_param(True, nn.interp_method.tf_bilinear,
                           nn.interp_mode.half_pixel)
        log("    set_resize_param OK")
    b = a.build([1, 3, src_hw[0], src_hw[1]], [1, 3, 640, 640])
    log("    build OK")
    src = nn.from_numpy(np.zeros((1, 3, src_hw[0], src_hw[1]), dtype=np.uint8))
    log("    from_numpy OK")
    kpu_in = kpu.get_input_tensor(0)
    log("    即将 builder.run（崩就崩在这句）")
    t0 = time.ticks_ms()
    b.run(src, kpu_in)
    log("    run OK %d ms" % (time.ticks_ms() - t0))
    t0 = time.ticks_ms()
    kpu.run()
    log("    kpu.run OK %d ms" % (time.ticks_ms() - t0))
    b.release()
    a.release()
    gc.collect()
    log("    %s 通过" % name)
    return True


kpu = None
try:
    log("### 载入 kmodel")
    kpu = nn.kpu()
    kpu.load_kmodel(KMODEL)
    log("    OK  free=%d" % gc.mem_free())

    stage("S1 恒等", (640, 640), None, False)
    stage("S2 只 pad", (480, 640), [0, 0, 0, 0, 80, 80, 0, 0], False)
    stage("S3 pad+resize", (480, 640), [0, 0, 0, 0, 80, 80, 0, 0], True)
    log("=== 三段全部通过 ===")
except BaseException as e:
    log("!!! 失败: %s: %s" % (type(e).__name__, e))
finally:
    log("done")
    _f.close()
