# -*- coding: utf-8 -*-
"""把本地文件传到 K210 板子上（串口 raw REPL）。

为什么要专门写这个：MaixPy 老固件的 FAT 驱动对「频繁 open/append/close」不可靠，
1.9 MB 文件写到 90% 会丢目录项。所以用 write_file_batch —— 一次会话连写多块，
把开关次数降一个数量级。

用法：
  python upload_kmodel.py <本地文件> <板上路径> [每次连写块数]
例：
  python upload_kmodel.py D:/2/k210/v5/deploy_demo.kmodel /sd/demo.kmodel
  python upload_kmodel.py D:/2/k210/v5/strike.kmodel /sd/strike.kmodel 8
"""
import os
import sys
import time

sys.path.insert(0, r'D:/2/k210')
from k210 import K210  # noqa: E402


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    src, dst = sys.argv[1], sys.argv[2]
    per = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    data = open(src, 'rb').read()
    print(f'[src ] {src}')
    print(f'[size] {len(data)} B ({len(data)/1024:.0f} KB)')
    print(f'[dst ] {dst}')

    d = K210()
    t0 = time.time()
    d.write_file_batch(dst, data, chunk=8192, per=per)
    dt = time.time() - t0
    print(f'[up  ] {dt:.0f}s  {len(data)/1024/max(dt,0.1):.1f} KB/s')

    # 校验：板上文件大小必须和本地一致
    got = d.exec(f"import os\nprint(os.stat('{dst}')[6])").strip()
    print(f'[chk ] 板上大小 {got}  {"OK" if got == str(len(data)) else "不一致！"}')
    d.close()
    return 0 if got == str(len(data)) else 2


if __name__ == '__main__':
    sys.exit(main())
