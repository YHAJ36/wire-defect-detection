# -*- coding: utf-8 -*-
"""ONNX → kmodel（K230，nncase 2.11.0 / target=k230，INT8 PTQ）

几个必须对齐的点（错了不会报错，只会精度崩）：
  1. 校准数据的预处理必须与板上推理**逐像素一致**：letterbox 到 640×640、
     灰边 114、BGR→RGB、/255、HWC→CHW、加 batch 维。
  2. ONNX 输入已是 float32 [0,1]，因此 compile_options.preprocess=False，
     板上直接喂 float32（nncase 会在 graph 内插量化）。
  3. 校准图取自 train 集（分布最接近 test），不用 val/test，避免泄漏进校准。
  4. mean/std 不在这里做——归一化已经写进预处理。

用法
    set DOTNET_ROLL_FORWARD=LatestMajor
    D:/yolo-env/Scripts/python.exe D:/2/deploy/compile_kmodel.py \
        --onnx D:/2/deploy/yolo11s_640_single.onnx \
        --out  D:/2/deploy/yolo11s_640.kmodel \
        --calib 50
"""
import argparse
import io
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault('DOTNET_ROLL_FORWARD', 'LatestMajor')
import nncase  # noqa: E402


def letterbox(im, new=640, color=(114, 114, 114)):
    """与 ultralytics 推理预处理一致：等比缩放 + 居中灰边。"""
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new - nh) // 2
    bottom = new - nh - top
    left = (new - nw) // 2
    right = new - nw - left
    return cv2.copyMakeBorder(im, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)


def build_calib(img_dir, count, size=640, uint8=False, layout='nhwc'):
    files = sorted([p for p in Path(img_dir).iterdir()
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not files:
        raise SystemExit(f'校准图目录为空：{img_dir}')
    step = max(1, len(files) // count)
    picked = files[::step][:count]
    data = []
    for p in picked:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = letterbox(im, size)
        if uint8:
            # uint8 RGB，与板上 ai2d 输出逐像素一致。布局必须跟 kmodel 输入
            # 声明的一致，否则量化的 scale 会按错的排布统计，全盘偏。
            rgb = im[:, :, ::-1]
            if layout == 'nchw':
                data.append(np.ascontiguousarray(rgb.transpose(2, 0, 1))[None])
            else:
                data.append(np.ascontiguousarray(rgb)[None])
            continue
        im = im[:, :, ::-1].astype(np.float32) / 255.0   # BGR->RGB
        im = np.ascontiguousarray(im.transpose(2, 0, 1)[None])
        data.append(im)
    print(f'校准图 {len(data)} 张（从 {len(files)} 张里等距抽，布局 {layout}）')
    return data


class Sink(io.RawIOBase):
    """gencode 要求 RawIOBase，BytesIO 不是。"""

    def __init__(self):
        self.buf = bytearray()

    def writable(self):
        return True

    def write(self, b):
        self.buf += b
        return len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', default='D:/2/deploy/yolo11s_640_single.onnx')
    ap.add_argument('--out', default='D:/2/deploy/yolo11s_640.kmodel')
    ap.add_argument('--calib', type=int, default=50)
    ap.add_argument('--calib-dir', default='D:/2/dataset_noleak/images/train')
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--samples', type=int, default=0,
                    help='实际喂入量化的样本数，0=全部校准图')
    ap.add_argument('--uint8', action='store_true',
                    help='uint8 NHWC 输入 + 图内归一化（板上零 Python 逐像素开销）。'
                         '不加此开关则保持 float32 NCHW、preprocess=False。')
    ap.add_argument('--uint8b', action='store_true',
                    help='uint8 的变体 B：用 input_range=[0,1] 代替显式 mean/std，'
                         '绕开 nncase FoldNopBinary 的崩溃。须同时给 --uint8。')
    ap.add_argument('--u8raw', action='store_true',
                    help='uint8 输入、preprocess=False，配 yolo11s_640_u8*.onnx '
                         '（归一化已写进图里）。校准数据自动切成 uint8。')
    ap.add_argument('--layout', choices=['nhwc', 'nchw'], default='nhwc',
                    help='uint8 输入的物理布局。nchw 配官方 libs/AI2D.py'
                         '（set_ai2d_dtype 默认 NCHW_FMT）；nhwc 配自写 ai2d。')
    ap.add_argument('--quant', default='uint8', help='激活量化类型 uint8/int8')
    ap.add_argument('--w-quant', default='uint8', help='权重量化类型 uint8/int8')
    ap.add_argument('--calib-method', default='Kld',
                    help='Kld（默认）或 NoClip。Kld 会主动裁掉长尾，'
                         '实测把类别分数压顶在 0.544，高分框全被削平。')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f'缺 ONNX：{onnx_path}')
    print(f'ONNX: {onnx_path} {onnx_path.stat().st_size/1e6:.2f} MB')

    cali = build_calib(args.calib_dir, args.calib, args.size,
                       args.uint8 or args.u8raw, args.layout)

    o = nncase.CompileOptions()
    o.target = 'k230'
    o.input_shape = [1, 3, args.size, args.size]
    o.output_layout = 'NCHW'
    o.dump_ir = False
    o.dump_asm = False
    if args.u8raw:
        # 最干净的一条路：归一化已经写进 ONNX 图里（make_uint8_onnx.py），
        # 所以这里不需要 nncase 的 preprocess，也就不碰那个会崩的 pass。
        # 板上把 ai2d 出来的 uint8 直接喂进来即可。
        # input_shape 一律按 NCHW 逻辑写，物理排布由 input_layout 决定。
        o.input_type = 'uint8'
        o.input_layout = 'NHWC' if args.layout == 'nhwc' else 'NCHW'
        o.preprocess = False
        o.swapRB = False
    elif args.uint8:
        # 板上直接喂 ai2d 出来的 uint8 RGB888（NHWC），归一化交给图内的
        # Normalize 算子。校准数据也要跟着换成 uint8 NHWC，否则量化的
        # scale 会按 float 分布算，全错。
        o.input_type = 'uint8'
        o.input_layout = 'NHWC'
        o.preprocess = True
        o.swapRB = False
        if args.uint8b:
            # 变体 B：不写 mean/std，改用 input_range=[0,1] 让 nncase 自己
            # 生成 uint8→[0,1] 的映射。变体 A（显式 mean=[0,0,0]、
            # std=[255,255,255]）会让 nncase 在 FoldNopBinary 折叠
            # "减 0 / 乘 1" 时报 Only The Expr Have CheckedType Can Get It's Shape。
            o.input_range = [0.0, 1.0]
        else:
            o.mean = [0.0, 0.0, 0.0]
            o.std = [255.0, 255.0, 255.0]
            # 不设 input_range 会抛 System.FormatException（'' 解析成 float 失败）——
            # nncase 内部把 input_range 当逗号串传，空串直接炸
            o.input_range = [0.0, 255.0]
    else:
        o.input_type = 'float32'
        o.input_layout = 'NCHW'
        o.preprocess = False
        o.swapRB = False

    c = nncase.Compiler(o)
    c.import_onnx(onnx_path.read_bytes(), nncase.ImportOptions())
    print('import_onnx OK')

    q = nncase.PTQTensorOptions()
    q.samples_count = args.samples if args.samples else len(cali)
    q.calibrate_method = args.calib_method
    q.quant_type = args.quant
    q.w_quant_type = args.w_quant
    q.cali_data = [nncase.RuntimeTensor.from_numpy(a) for a in cali]
    c.use_ptq(q)
    print(f'use_ptq OK  samples_count={q.samples_count}')

    t0 = time.time()
    c.compile()
    print(f'compile OK  {time.time()-t0:.1f}s')

    sink = io.BytesIO()
    c.gencode(sink)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(sink.getvalue())
    print(f'kmodel: {out}  {out.stat().st_size/1e6:.2f} MB')


if __name__ == '__main__':
    main()
