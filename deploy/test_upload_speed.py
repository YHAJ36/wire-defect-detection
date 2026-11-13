# -*- coding: utf-8 -*-
r"""测 mpremote 传输速率随 chunk_size 的变化，找出不炸板子的最大块。

背景：mpremote 的 fs_writefile 默认 chunk_size=256，且用 repr(bytes) 传二进制
（不可打印字节膨胀成 \xNN，平均 4 倍），每块一次完整 raw REPL 往返。
实测 11 MB 传 10 分钟只进去 2.69 MB ≈ 4.5 KB/s，纯属往返次数太多。

这里用 64 KB 测试块分别试 256 / 1024 / 4096 / 8192 / 16384，
用真实速率决定全量传输用多大块。失败（板子 raw REPL 吃不下长行）就停。
"""
import os
import time
from pathlib import Path

from mpremote.transport_serial import SerialTransport

PORT = "COM8"
SRC = Path("D:/2/deploy/_speed_probe.bin")
DST = "/sdcard/_speed_probe.bin"
SIZE = 64 * 1024
CHUNKS = [256, 1024, 4096, 8192, 16384]


def main():
    if not SRC.exists() or SRC.stat().st_size != SIZE:
        SRC.write_bytes(os.urandom(SIZE))
    data = SRC.read_bytes()

    t = SerialTransport(PORT)
    t.enter_raw_repl()
    ok = []
    try:
        for c in CHUNKS:
            t0 = time.time()
            try:
                t.fs_writefile(DST, data, chunk_size=c)
            except Exception as e:
                print("chunk=%-6d 失败：%s: %s" % (c, type(e).__name__, str(e)[:200]))
                # 失败后 raw REPL 可能处于半截状态，重新进一次
                try:
                    t.exit_raw_repl()
                except Exception:
                    pass
                t.enter_raw_repl()
                continue
            dt = time.time() - t0
            kbps = len(data) / 1024 / dt
            print("chunk=%-6d %6.2f s  -> %8.1f KB/s   全量 11MB 预计 %.0f s"
                  % (c, dt, kbps, 11.08 * 1024 / kbps))
            ok.append((c, kbps))
        # 校验最大块写进去的内容是否完整
        st = t.fs_stat(DST)
        print("板端文件大小 %d 字节（源 %d）" % (st.st_size, SIZE))
    finally:
        try:
            t.exec("import os\nos.remove('%s')" % DST)
        except Exception:
            pass
        t.exit_raw_repl()
        try:
            t.close()
        except Exception:
            pass

    if ok:
        best = max(ok, key=lambda x: x[1])
        print("最佳：chunk=%d，%.1f KB/s" % best)
        print("全量 11.08 MB 预计 %.0f s" % (11.08 * 1024 / best[1]))


if __name__ == "__main__":
    main()
