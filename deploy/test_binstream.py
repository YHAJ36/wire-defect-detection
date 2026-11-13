# -*- coding: utf-8 -*-
r"""绕开文本编码，直接把二进制灌进板子 stdin —— 测真实串口吞吐。

原理：
  exec_raw_no_follow(code) 把代码交给板子执行后**不等待输出**；
  板子的代码阻塞在 sys.stdin 上按精确字节数读；
  此时 PC 侧连续 write 原始二进制，板上边收边写 SD 卡。
  全程零编码膨胀（base64 是 1.33x，mpremote 的 repr 是 ~4x）。

先测 64 KB。成功且明显快于 base64（57.8 KB/s）才用于全量。
失败则复位 raw REPL，退回 base64 方案。
"""
import os
import time
from pathlib import Path

from mpremote.transport_serial import SerialTransport

PORT = "COM8"
SRC = Path("D:/2/deploy/_speed_probe.bin")
DST = "/sdcard/_speed_probe.bin"


def make_code(dst, total):
    return (
        "import sys\n"
        "f=open('%s','wb')\n"
        "try:\n"
        " rd=sys.stdin.buffer.read\n"
        "except AttributeError:\n"
        " rd=sys.stdin.read\n"
        "n=%d\n"
        "while n:\n"
        " b=rd(min(4096,n))\n"
        " if not b: break\n"
        " f.write(b)\n"
        " n-=len(b)\n"
        "f.close()\n"
        "print('got',%d-n)\n" % (dst, total, total)
    ).encode()


def binupload(t, dst, data, block=64 * 1024, verbose=True):
    total = len(data)
    t.use_raw_paste = False
    t.exec_raw_no_follow(make_code(dst, total))
    t0 = time.time()
    mv = memoryview(data)
    sent = 0
    for i in range(0, total, block):
        piece = bytes(mv[i:i + block])
        t.serial.write(piece)
        t.serial.flush()
        sent += len(piece)
    dt_write = time.time() - t0
    resp = t.read_until(1, b"\x04", timeout=30, timeout_overall=120)
    dt_total = time.time() - t0
    if verbose:
        print("  写入 %d 字节耗时 %.2f s" % (sent, dt_write))
        print("  等到响应耗时 %.2f s，尾部: %r" % (dt_total, resp[-120:]))
        print("  有效速率 %.1f KB/s" % (total / 1024 / dt_total))
    return dt_total, resp


def main():
    if not SRC.exists() or SRC.stat().st_size != 64 * 1024:
        SRC.write_bytes(os.urandom(64 * 1024))
    data = SRC.read_bytes()

    t = SerialTransport(PORT)
    t.enter_raw_repl()
    try:
        dt, resp = binupload(t, DST, data)
        st = t.fs_stat(DST).st_size
        print("  板端文件 %d 字节（源 %d）%s" % (st, len(data), "OK" if st == len(data) else "不符"))
        print("  全量 11.08 MB 预计 %.0f s" % (11.08 * 1024 / (len(data) / 1024 / dt)))
    except Exception as e:
        print("二进制流失败：%s: %s" % (type(e).__name__, str(e)[:200]))
        try:
            t.serial.write(b"\r\x03")
            time.sleep(0.3)
            t.serial.read(t.serial.inWaiting())
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
