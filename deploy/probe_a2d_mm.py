# 最后一次 ai2d 归因测试：先 MediaManager.init()，再跑恒等变换。
#
# 依据：ai2d 是 K230 的硬件模块，其时钟/内存池由媒体子系统管理。
# 前面所有失败用例里，要么没 init，要么 init 了但混着 pad+resize+list 多个变量。
# 这一份只保留两个变量：MediaManager.init() + 恒等 run。

import gc
import time

import nncase_runtime as nn
import ulab.numpy as np

KMODEL = "/sdcard/yolo11s_640_nchw.kmodel"
LOG = "/sdcard/a2d_mm_log.txt"

_f = open(LOG, "w")


def log(msg):
    print(msg)
    _f.write(msg + "\n")
    _f.flush()


try:
    log("### M1 MediaManager.init")
    from media.media import MediaManager
    MediaManager.init()
    log("    OK free=%d" % gc.mem_free())

    log("### M2 载入 kmodel")
    kpu = nn.kpu()
    kpu.load_kmodel(KMODEL)
    log("    OK free=%d" % gc.mem_free())

    log("### M3 ai2d 恒等 build")
    a = nn.ai2d()
    a.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT,
                np.uint8, np.uint8)
    b = a.build([1, 3, 640, 640], [1, 3, 640, 640])
    log("    build OK")

    src = nn.from_numpy(np.zeros((1, 3, 640, 640), dtype=np.uint8))
    log("    from_numpy OK")

    kpu_in = kpu.get_input_tensor(0)
    log("    即将 run（崩就在这句）")
    t0 = time.ticks_ms()
    b.run(src, kpu_in)
    log("    run OK %d ms" % (time.ticks_ms() - t0))

    t0 = time.ticks_ms()
    kpu.run()
    log("    kpu.run OK %d ms" % (time.ticks_ms() - t0))
    out = kpu.get_output_tensor(0).to_numpy()
    log("    out shape=%s" % str(out.shape))
    log("=== ai2d 在 MediaManager 下可用 ===")
except BaseException as e:
    log("!!! 失败: %s: %s" % (type(e).__name__, e))
finally:
    try:
        from media.media import MediaManager
        MediaManager.deinit()
    except Exception:
        pass
    log("done")
    _f.close()
