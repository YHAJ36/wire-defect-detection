# 量三种 kmodel 的纯推理耗时，隔离「图内预处理」的开销
import gc
import time

import nncase_runtime as nn
import ulab.numpy as np

CASES = [
    ("s_u8r_NHWC+cast/div/tr", "/sdcard/yolo11s_640_u8r.kmodel", (1, 640, 640, 3)),
    ("s_nchw_U8_NCHW(no tr)", "/sdcard/yolo11s_640_nchw.kmodel", (1, 3, 640, 640)),
    ("n_u8r_NHWC+cast/div/tr", "/sdcard/yolo11n_640_u8r.kmodel", (1, 640, 640, 3)),
]

for name, path, shape in CASES:
    k = None
    try:
        k = nn.kpu()
        t0 = time.ticks_ms()
        k.load_kmodel(path)
        t_load = time.ticks_ms() - t0

        a = np.zeros(shape, dtype=np.uint8)
        t = nn.from_numpy(a)
        k.set_input_tensor(0, t)
        k.run()                      # 预热
        n = 5
        t0 = time.ticks_ms()
        for _ in range(n):
            k.set_input_tensor(0, t)
            k.run()
        dt = (time.ticks_ms() - t0) // n
        print("%-24s 加载 %4d ms  推理 %4d ms  free %dKB"
              % (name, t_load, dt, gc.mem_free() // 1024))
    except BaseException as e:
        print("%-24s FAIL %s: %s" % (name, type(e).__name__, e))
    finally:
        try:
            if k is not None:
                k.release()
        except Exception:
            pass
        del k
        gc.collect()
        try:
            nn.shrink_memory_pool()
        except Exception:
            pass

print("done")
