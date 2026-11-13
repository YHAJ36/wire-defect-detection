# -*- coding: utf-8 -*-
"""板子插上之后，一条命令把该推的推上去、该跑的直接跑。

做的事（每一步都可重复执行，文件已存在会覆盖）：
  1. 扫串口，找到 K230 的 CDC 口（没插就报错退出，不会瞎连）
  2. 推 yolo11s_640_u8r.kmodel  -> /sdcard/
  3. 推 board_case/testcase.bin -> /sdcard/
  4. 跑 board_infer_raw.py，把板子打印的东西原样贴出来

用法
    D:/yolo-env/Scripts/python.exe D:/2/deploy/push_and_run.py
    D:/yolo-env/Scripts/python.exe D:/2/deploy/push_and_run.py --port COM9
    D:/yolo-env/Scripts/python.exe D:/2/deploy/push_and_run.py --smoke   # 只跑加载冒烟

安全：全程只做文件写入 /sdcard 和 MicroPython 脚本执行，
不碰固件、不调用 K230BurningTool、不写 flash。
"""
import argparse
import subprocess
import sys
from pathlib import Path

PY = r'C:\Users\LR\AppData\Local\Programs\Python\Python312\python.exe'
DEPLOY = Path('D:/2/deploy')
KM = DEPLOY / 'yolo11s_640_u8r.kmodel'
BIN = DEPLOY / 'board_case' / 'testcase.bin'


def ports():
    try:
        import serial.tools.list_ports as lp
    except ImportError:
        raise SystemExit('缺 pyserial：%s -m pip install pyserial' % PY)
    return [p.device for p in lp.comports()]


def mpremote(port, *args, timeout=600):
    cmd = [PY, '-m', 'mpremote', 'connect', port] + list(args)
    print('  $ ' + ' '.join(cmd[3:]))
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=timeout, encoding='utf-8', errors='replace')
    if r.stdout:
        print(r.stdout.rstrip())
    if r.returncode != 0:
        if r.stderr:
            print('[stderr] ' + r.stderr.rstrip()[-1500:])
        raise SystemExit('mpremote 失败，返回码 %d' % r.returncode)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default=None)
    ap.add_argument('--smoke', action='store_true', help='只跑 board_smoke.py（全零输入，不用推 bin）')
    ap.add_argument('--skip-copy', action='store_true', help='文件已在板上，不重复推')
    args = ap.parse_args()

    port = args.port
    if not port:
        p = ports()
        if not p:
            raise SystemExit('没扫到任何串口——板子没插，或者 USB 线是纯充电线')
        print('扫到串口:', p)
        port = p[0]
        if len(p) > 1:
            print('注意：有多个串口，默认取第一个 %s，可用 --port 指定' % port)

    for f in (KM, BIN):
        if not f.exists():
            raise SystemExit('缺文件: %s（先跑 compile_kmodel.py / make_board_testcase.py）' % f)

    if not args.skip_copy:
        print('[1] 推 kmodel（11 MB，走 USB CDC，可能要十几秒到一两分钟）')
        mpremote(port, 'fs', 'cp', str(KM), ':/sdcard/yolo11s_640_u8r.kmodel')
    else:
        print('[1] 跳过推 kmodel（--skip-copy）')

    if args.smoke:
        print('[*] 运行 board_smoke.py（全零输入：只验加载 + 推理 + fps）')
        mpremote(port, 'run', str(DEPLOY / 'board_smoke.py'))
        return

    # 默认走全链：先冒烟证明「能加载能跑」，再上真实输入对数值。
    # 冒烟失败就没必要推 1.2 MB 的 bin 了，直接停。
    print('[2] 冒烟：board_smoke.py')
    mpremote(port, 'run', str(DEPLOY / 'board_smoke.py'))

    if not args.skip_copy:
        print('[3] 推测试输入（1.2 MB）')
        mpremote(port, 'fs', 'cp', str(BIN), ':/sdcard/testcase.bin')

    print('[4] 真实输入：board_infer_raw.py')
    mpremote(port, 'run', str(DEPLOY / 'board_infer_raw.py'))


if __name__ == '__main__':
    main()
