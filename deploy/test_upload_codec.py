# -*- coding: utf-8 -*-
r"""对比三种传输编码的实际速率，选出全量上传用哪种。

已实测（chunk_size 扫描）：mpremote 默认 repr(bytes) 编码下，
chunk 256 -> 7.1 KB/s，16384 -> 29.1 KB/s。提升到 29 就上不去了，
说明瓶颈不是往返次数，而是 repr 把每个不可打印字节膨胀成 4 字节（\xNN）。

本脚本对比：
  A. repr(bytes)          —— mpremote 原生，膨胀 ~4x
  B. base64 + a2b_base64  —— 膨胀 1.33x，板上多一次解码
并试 115200 / 921600 两档波特率（USB CDC 上波特率通常不生效，但 K230 的
CDC 实现有可能真的按波特率限流，值得花 20 秒排除）。
"""
import base64
import os
import time
from pathlib import Path

from mpremote.transport_serial import SerialTransport

PORT = "COM8"
SRC = Path("D:/2/deploy/_speed_probe.bin")
DST = "/sdcard/_speed_probe.bin"
SIZE = 64 * 1024


def write_repr(t, data, chunk):
    t.exec("f=open('%s','wb')\nw=f.write" % DST)
    for i in range(0, len(data), chunk):
        t.exec("w(" + repr(data[i:i + chunk]) + ")")
    t.exec("f.close()")


def write_b64(t, data, chunk):
    t.exec("import ubinascii\nf=open('%s','wb')\nw=f.write" % DST)
    for i in range(0, len(data), chunk):
        s = base64.b64encode(data[i:i + chunk]).decode()
        t.exec("w(ubinascii.a2b_base64(b'%s'))" % s)
    t.exec("f.close()")


def bench(t, name, fn, chunk, data):
    t0 = time.time()
    try:
        fn(t, data, chunk)
    except Exception as e:
        return "%s 失败 %s: %s" % (name, type(e).__name__, str(e)[:120])
    dt = time.time() - t0
    st = t.fs_stat(DST).st_size
    kbps = len(data) / 1024 / dt
    flag = "" if st == len(data) else "  <<< 大小不符 %d" % st
    return "%-16s %6.2f s  %7.1f KB/s  全量11MB %5.0f s%s" % (
        name, dt, kbps, 11.08 * 1024 / kbps, flag)


def main():
    if not SRC.exists() or SRC.stat().st_size != SIZE:
        SRC.write_bytes(os.urandom(SIZE))
    data = SRC.read_bytes()

    for baud in (115200, 921600):
        print("=== baudrate %d ===" % baud)
        t = SerialTransport(PORT, baudrate=baud)
        t.enter_raw_repl()
        try:
            print(bench(t, "repr c=16384", write_repr, 16384, data))
            print(bench(t, "b64  c=8192", write_b64, 8192, data))
            print(bench(t, "b64  c=16384", write_b64, 16384, data))
            print(bench(t, "b64  c=32768", write_b64, 32768, data))
        except Exception as e:
            print("  该波特率下出错:", type(e).__name__, str(e)[:150])
            try:
                t.exit_raw_repl()
            except Exception:
                pass
            try:
                t.enter_raw_repl()
            except Exception:
                pass
        finally:
            try:
                t.exec("import os\nos.remove('%s')" % DST)
            except Exception:
                pass
            try:
                t.exit_raw_repl()
                t.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
