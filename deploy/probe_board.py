# 板子体检：内存、SD 卡写速、可用传输通道
import os
import gc
import time
import sys

print("=== 系统 ===")
try:
    print("uname:", os.uname())
except Exception as e:
    print("uname 失败", e)
print("mem_free: %.2f MB" % (gc.mem_free() / 1048576))
print("mem_alloc: %.2f MB" % (gc.mem_alloc() / 1048576))

print("=== 文件系统 ===")
for p in ("/", "/sdcard", "/flash"):
    try:
        v = os.statvfs(p)
        print("%-9s total %.1f MB  free %.1f MB" %
              (p, v[2] * v[0] / 1048576, v[3] * v[0] / 1048576))
    except Exception as e:
        print("%-9s 不可用: %s" % (p, e))

print("=== SD 卡本地写速（256 KB，4096 字节/次）===")
try:
    t0 = time.ticks_ms()
    f = open("/sdcard/_speedtest.bin", "wb")
    blk = b"\x00" * 4096
    for _ in range(64):
        f.write(blk)
    f.close()
    dt = time.ticks_ms() - t0
    print("  写入 256 KB 用 %d ms -> %.1f KB/s" % (dt, 256 * 1000 / max(dt, 1)))
    t0 = time.ticks_ms()
    f = open("/sdcard/_speedtest.bin", "rb")
    n = 0
    while True:
        b = f.read(4096)
        if not b:
            break
        n += len(b)
    f.close()
    dt = time.ticks_ms() - t0
    print("  读回 %d 字节用 %d ms -> %.1f KB/s" % (n, dt, (n / 1024) * 1000 / max(dt, 1)))
    os.remove("/sdcard/_speedtest.bin")
except Exception as e:
    print("  写速测试失败:", e)
    sys.print_exception(e)

print("=== 传输通道 ===")
try:
    import network
    print("network 模块:", [x for x in dir(network) if x[0].isupper()])
except Exception as e:
    print("network 不可用:", e)

try:
    import machine
    cand = [x for x in dir(machine) if any(k in x.upper() for k in ("USB", "MSC", "SD"))]
    print("machine 相关:", cand)
except Exception as e:
    print("machine 不可用:", e)

print("=== 已有 kmodel 状态 ===")
for n in ("yolo11s_640_u8r.kmodel",):
    p = "/sdcard/" + n
    try:
        print("  %s : %d 字节" % (n, os.stat(p)[6]))
    except Exception as e:
        print("  %s : 不存在 (%s)" % (n, e))

print("=== 完成 ===")
