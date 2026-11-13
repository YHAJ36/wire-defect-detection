# -*- coding: utf-8 -*-
r"""用 base64 分块把文件推到板子 /sdcard，带进度、断点容错与 SHA256 校验。

为什么不用 mpremote fs cp：它默认 chunk_size=256 且用 repr(bytes) 编码
（不可打印字节变成 \xNN，平均膨胀 4 倍），实测只有 4.5 KB/s，11 MB 要 40 分钟。
base64 只膨胀 1.33 倍，chunk 16384 实测 57.8 KB/s，11 MB 约 196 秒。

为什么不走 stdin 二进制直传：K230 的 raw REPL 在 exec 后不把 stdin 透传给
板上 sys.stdin，实测写入后 30 秒无任何响应、REPL 失联，需 Ctrl-C 恢复。已放弃。

用法
    python upload_kmodel.py                      # 传 kmodel + testcase.bin
    python upload_kmodel.py --only manual        # 传手动编译出来的那个（D:/2/manual）
    python upload_kmodel.py --file "D:/x.kmodel::/sdcard/x.kmodel"     # 任意文件
    python upload_kmodel.py --only manual --dry-run                    # 只看不传
"""
import argparse
import base64
import hashlib
import time
from pathlib import Path

from mpremote.transport_serial import SerialTransport

PORT = "COM8"
CHUNK = 16384


def sha256_local(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


SHA_CODE = (
    "import uhashlib,ubinascii\n"
    "h=uhashlib.sha256()\n"
    "f=open(%r,'rb')\n"
    "while True:\n"
    " b=f.read(4096)\n"
    " if not b: break\n"
    " h.update(b)\n"
    "f.close()\n"
    "print(ubinascii.hexlify(h.digest()).decode())\n"
)


def upload(t, src, dst, chunk=CHUNK, label=""):
    data = Path(src).read_bytes()
    n = len(data)
    t.exec("import ubinascii\nf=open(%r,'wb')\nw=f.write" % dst)
    t0 = time.time()
    nchunk = (n + chunk - 1) // chunk
    for k, i in enumerate(range(0, n, chunk)):
        payload = base64.b64encode(data[i:i + chunk]).decode()
        cmd = "w(ubinascii.a2b_base64(b'%s'))" % payload
        for attempt in (1, 2):
            try:
                t.exec(cmd)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print("  第 %d 块重试（%s）" % (k, type(e).__name__), flush=True)
                time.sleep(0.3)
        if k % 32 == 0 or k == nchunk - 1:
            done = min(i + chunk, n)
            el = max(time.time() - t0, 1e-6)
            print("  %s %5.1f%%  %.2f/%.2f MB  %.1f KB/s  剩余 %.0f s"
                  % (label, 100.0 * done / n, done / 1048576, n / 1048576,
                     done / 1024 / el, (n - done) / 1024 / (done / 1024 / el)),
                  flush=True)
    t.exec("f.close()")

    size = t.fs_stat(dst).st_size
    local = sha256_local(src)
    remote = t.exec(SHA_CODE % dst).decode().strip().splitlines()[-1].strip()
    ok = (size == n and remote == local)
    print("  %s 大小 %d/%d  SHA256 %s  -> %s"
          % (label, size, n, "一致" if remote == local else "不一致(%s vs %s)" % (remote[:12], local[:12]),
             "通过" if ok else "失败"), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["kmodel", "nchw", "nano", "bin", "manual"],
                    default=None, help="传预设项；manual = 手动编译出来的那个")
    ap.add_argument("--file", action="append", default=[], metavar="源:目标",
                    help="任意文件，格式 源:目标（可写多次）。目标省略则放到 /sdcard/ 下同名")
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--dry-run", action="store_true", help="只打印要传什么，不连板子")
    args = ap.parse_args()

    jobs = [
        ("kmodel", Path("D:/2/deploy/yolo11s_640_u8r.kmodel"), "/sdcard/yolo11s_640_u8r.kmodel"),
        ("nchw", Path("D:/2/deploy/yolo11s_640_nchw.kmodel"), "/sdcard/yolo11s_640_nchw.kmodel"),
        ("nano", Path("D:/2/deploy/yolo11n_640_u8r.kmodel"), "/sdcard/yolo11n_640_u8r.kmodel"),
        ("bin", Path("D:/2/deploy/board_case/testcase.bin"), "/sdcard/testcase.bin"),
        # 手动转换流程（D:/2/manual）产出的模型，输入是 float32 NCHW
        ("manual", Path("D:/2/manual/yolo11n_640_manual.kmodel"),
         "/sdcard/yolo11n_640_manual.kmodel"),
    ]

    # --file 优先：它明确说了要传什么，就只传它
    if args.file:
        jobs = []
        for spec in args.file:
            # 分隔符用 :: 而不是 :，因为 Windows 路径本身带冒号（D:），
            # 按单个冒号切会把 "D:/2/x.kmodel" 切成 "D" 和 "/2/x.kmodel"
            if "::" in spec:
                src, dst = spec.split("::", 1)
            else:
                src, dst = spec, ""
            src = Path(src.strip().strip('"'))
            dst = dst.strip() or ("/sdcard/" + src.name)
            jobs.append((src.name, src, dst))

    if args.only and not args.file:
        jobs = [j for j in jobs if j[0] == args.only]

    if not jobs:
        raise SystemExit("没有要传的文件，检查 --only / --file")

    if args.dry_run:
        print("只演练，不连板子：")
        for name, src, dst in jobs:
            state = "存在 %.2f MB" % (src.stat().st_size / 1048576) if src.exists() else "缺失！"
            print("  %-28s -> %-40s %s" % (src, dst, state))
        return

    t = SerialTransport(args.port)

    t.enter_raw_repl()
    all_ok = True
    try:
        # 先把上次残留的 0 字节文件删掉，避免混淆
        for _, _, dst in jobs:
            try:
                t.exec("import os\nos.remove(%r)" % dst)
            except Exception:
                pass
        for name, src, dst in jobs:
            if not src.exists():
                print("缺文件 %s，跳过" % src)
                all_ok = False
                continue
            print("[%s] %s -> %s (%.2f MB)" % (name, src.name, dst, src.stat().st_size / 1048576))
            all_ok &= upload(t, src, dst, label=name)
    finally:
        try:
            t.exit_raw_repl()
            t.close()
        except Exception:
            pass
    print("结果：%s" % ("全部通过" if all_ok else "有失败项"))


if __name__ == "__main__":
    main()
