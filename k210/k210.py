"""K210 (MaixPy) 串口工具 —— raw REPL 方式收发代码。

为什么不用 mpremote:
    MaixPy 的固件是老版 MicroPython，文件系统协议和 mpremote 不兼容。
    这里直接用 pyserial + MicroPython 的 raw REPL 协议:
        0x01 进入 raw REPL -> 发代码 -> 0x04 执行 -> 读输出 -> 0x02 退出

用法:
    from k210 import K210
    d = K210()                      # 默认 COM9
    print(d.exec("import os;print(os.listdir('/'))"))
    print(d.read_file('/flash/boot.py'))
    d.close()
"""
import time

import serial


class K210:
    def __init__(self, port='COM9', baud=115200):
        self.s = serial.Serial(port, baud, timeout=0.2)
        time.sleep(0.3)
        # 抢回控制权: 板子重启后会跑 /flash/_boot.py（初始化 sensor/lcd），
        # 单次 Ctrl-C 常常打不断，多发几次并给足间隔
        for _ in range(6):
            self.s.write(b'\x03')
            time.sleep(0.35)
        self.s.reset_input_buffer()
        self._sync()

    # ---------- 底层 ----------
    def _sync(self, t=6.0):
        """确保回到 normal REPL 的 >>> 提示符"""
        self.s.write(b'\x02')          # 退出 raw
        time.sleep(0.2)
        buf = b''
        t0 = time.time()
        while time.time() - t0 < t:
            buf += self.s.read(256)
            if buf.rstrip().endswith(b'>>>'):
                time.sleep(0.15)
                self.s.read(400)
                return True
        return False

    def _read_until(self, marker, t=20.0):
        buf = b''
        t0 = time.time()
        while time.time() - t0 < t:
            chunk = self.s.read(512)
            if chunk:
                buf += chunk
                if marker in buf:
                    break
        return buf

    # ---------- 对外 ----------
    def exec(self, code, timeout=25.0):
        """在板上执行一段代码, 返回 stdout 文本。支持多行。"""
        if not code.endswith('\n'):
            code += '\n'
        self.s.reset_input_buffer()
        self.s.write(b'\x01')                       # 进入 raw REPL
        if not self._read_until(b'>', 5.0):
            raise RuntimeError('raw REPL 无响应')
        self.s.reset_input_buffer()
        self.s.write(code.encode('utf-8'))
        time.sleep(0.05)
        self.s.write(b'\x04')                       # 执行
        out = self._read_until(b'\x04>', timeout)
        self.s.write(b'\x02')                       # 退出 raw
        time.sleep(0.15)
        self.s.reset_input_buffer()
        txt = out.decode('utf-8', 'replace')
        if txt.startswith('OK'):
            txt = txt[2:]
        # 去掉结尾的执行标记
        idx = txt.rfind('\x04>')
        if idx >= 0:
            txt = txt[:idx]
        return txt.replace('\x04', '').replace('\x02', '').strip()

    def read_file(self, path, chunk=1024):
        """分块读文件, 返回 bytes。大文件也不会把 REPL 撑爆。"""
        size = self.exec(
            "import os\n"
            "try: print(os.stat(%r)[6])\n"
            "except Exception as e: print('ERR', e)" % path)
        try:
            n = int(size.strip().splitlines()[-1])
        except Exception:
            raise RuntimeError('读不到 %s: %s' % (path, size))

        data = bytearray()
        while len(data) < n:
            part = self.exec(
                "f=open(%r,'rb')\n"
                "f.seek(%d)\n"
                "d=f.read(%d)\n"
                "f.close()\n"
                "import binascii\n"
                "print(binascii.hexlify(d).decode())" % (path, len(data), chunk),
                timeout=30.0)
            hexline = ''.join(part.split())
            if not hexline:
                break
            data += bytes.fromhex(hexline)
        return bytes(data[:n])

    def write_file(self, path, data, chunk=512):
        """分块写文件。用 base64 而不是 hex，体积只膨胀 1/3，比 hex 快一倍。

        每块都重新 open('ab')，避免依赖板上长期存在的文件句柄——
        raw REPL 每次 exec 是独立上下文，变量不保留。
        """
        import base64
        self.exec("f=open(%r,'wb')\nf.close()\nprint('ok')" % path)
        for off in range(0, len(data), chunk):
            part = data[off:off + chunk]
            b64 = base64.b64encode(part).decode()
            self.exec(
                "import binascii\n"
                "f=open(%r,'ab')\n"
                "f.write(binascii.a2b_base64(%r))\n"
                "f.close()\n"
                "print('ok')" % (path, b64))
        size = self.exec("import os;print(os.stat(%r)[6])" % path)
        got = int(size.strip().splitlines()[-1])
        if got != len(data):
            raise RuntimeError('写入不完整: %d / %d' % (got, len(data)))
        return got

    def write_file_batch(self, path, data, chunk=8192, per=8, verbose=True):
        """批量写入：一次 exec 里连续写 per 块，把 open/close 次数降一个数量级。

        为什么需要它：MaixPy 老固件的 FAT 驱动在 232 次 open('ab')/close 之后就
        把目录项弄丢了（最后 os.stat 报 EINVAL，文件直接不存在）。
        每次 exec 独立上下文、句柄留不住，所以只能"一次开、连写多块、再关"。
        """
        import base64
        self.exec("f=open(%r,'wb')\nf.close()\nprint('ok')" % path)
        off = 0
        nb = 0
        while off < len(data):
            parts = []
            for _ in range(per):
                if off >= len(data):
                    break
                parts.append(data[off:off + chunk])
                off += len(parts[-1])
            code = "import binascii\nf=open(%r,'ab')\n" % path
            for p in parts:
                code += "f.write(binascii.a2b_base64(%r))\n" % base64.b64encode(p).decode()
            code += "f.close()\nprint('ok')\n"
            out = self.exec(code, timeout=120.0)
            nb += 1
            if 'ok' not in out:
                raise RuntimeError('第 %d 批写入异常: %s' % (nb, out[:200]))
            if verbose:
                print(f'  [{off/len(data)*100:5.1f}%] {off}/{len(data)}', flush=True)
        size = self.exec("import os;print(os.stat(%r)[6])" % path)
        got = int(size.strip().splitlines()[-1])
        if got != len(data):
            raise RuntimeError('写入不完整: %d / %d' % (got, len(data)))
        return got

    def listdir(self, path='/'):
        return self.exec("import os;print(os.listdir(%r))" % path)

    def close(self):
        try:
            self._sync()
            self.s.close()
        except Exception:
            pass


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    d = K210()
    print('--- 根目录 ---')
    print(d.listdir('/'))
    print('--- flash ---')
    print(d.listdir('/flash'))
    d.close()
