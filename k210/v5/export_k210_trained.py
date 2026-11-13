# -*- coding: utf-8 -*-
"""把 train_k210.py 训出来的权重导出成 K210 能吃的 ONNX。

和 export_v5_raw.py 的区别
--------------------------------
train_k210.py 在训练时就已经把结构改成 K210 兼容了（ReLU / 去池化 SPPF / 固定上采样），
所以这里**不能再做一遍 swap**（会把 ReLU 换成 ReLU，无害，但 Upsample 已经不存在了）。
本脚本只做两件事：
  1) 把 Detect.forward 换成 raw（不做 sigmoid / split / pow 解码）
  2) 导出 ONNX 并做算子体检

坑：best.pt 是用 pickle 存整个模型对象的，里面 SPPFK / Up2 是在 train_k210.py
    被当作 __main__ 执行时定义的，所以反序列化需要 __main__ 里有同名类。
    本文件顶层定义这两个类，就是为了这个。

用法：
    python export_k210_trained.py --weights runs/v5n125_break_320/weights/best.pt
"""
import os
import sys
import argparse
from collections import Counter

import torch
import torch.nn as nn

Y5 = r'D:/2/code/baselines/yolov5'
if Y5 not in sys.path:
    sys.path.insert(0, Y5)
os.chdir(Y5)

from models.common import SPPF            # noqa: E402
from models.yolo import Detect            # noqa: E402
from models.experimental import attempt_load  # noqa: E402


# ---- 必须与 train_k210.py 里的定义完全一致，供 pickle 反序列化 ----
class SPPFK(SPPF):
    def forward(self, x):
        y = [self.cv1(x)]
        y.extend([y[-1]] * 3)
        return self.cv2(torch.cat(y, 1))


class Up2(nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(
            x, scale_factor=2, mode='nearest', recompute_scale_factor=False)


def raw_forward(self, x):
    """只吐 3 个尺度的原始卷积输出：(1, na*(5+nc), ny, nx)"""
    return tuple(self.m[i](x[i]) for i in range(self.nl))


ALLOWED = set("""Abs Add AveragePool Ceil Concat Constant Conv ConvTranspose Cos Div Exp Floor
    LpNormalization LeakyRelu Log MatMul MaxPool Max Min Mul Neg Pad ReduceMax ReduceMean
    ReduceMin ReduceSum Relu Reshape Resize Sin Slice Softmax SpaceToDepth Sqrt Squeeze Sub
    Transpose""".split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=r'D:/2/k210/v5/runs/v5n125_break_320/weights/best.pt')
    ap.add_argument('--imgsz', default='224x320', help='HxW')
    ap.add_argument('--out', default=r'D:/2/k210/v5/v5n125_break_k210.onnx')
    ap.add_argument('--opset', type=int, default=12)
    a = ap.parse_args()

    H, W = (int(v) for v in a.imgsz.lower().split('x'))
    assert H % 32 == 0 and W % 32 == 0
    assert W <= 320 and H <= 240, 'K210 KPU 特征图上限 320x240'

    model = attempt_load(a.weights, device='cpu', inplace=True, fuse=True)
    model = model.float().eval()
    head = model.model[-1]
    print(f'[load] {a.weights}')
    print(f'[head] nc={head.nc} na={head.na} nl={head.nl} stride={head.stride.tolist()}')
    print(f'[head] anchors={[(head.anchors[i] * head.stride[i]).tolist() for i in range(head.nl)]}')

    # 结构体检：确认激活是 ReLU（训练时改的），不是 SiLU
    n_relu = n_silu = 0
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            n_relu += 1
        elif m.__class__.__name__ == 'SiLU':
            n_silu += 1
    print(f'[chk ] ReLU {n_relu} 处 / SiLU {n_silu} 处  ' +
          ('OK' if n_relu > 0 and n_silu == 0 else '⚠ 结构不对！'))

    n_param = sum(p.numel() for p in model.parameters())
    print(f'[chk ] params {n_param/1e6:.3f} M')

    Detect.forward = raw_forward
    dummy = torch.zeros(1, 3, H, W)
    with torch.no_grad():
        outs = model(dummy)
    for i, o in enumerate(outs):
        print(f'[out ] p{i} {tuple(o.shape)}')

    torch.onnx.export(
        model, (dummy,), a.out,
        opset_version=a.opset,
        input_names=['images'],
        output_names=[f'p{i}' for i in range(len(outs))],
        do_constant_folding=True,
        dynamo=False,
    )
    print(f'[onnx] -> {a.out}  {os.path.getsize(a.out)/1e6:.2f} MB')

    import onnx
    g = onnx.load(a.out).graph
    c = Counter(n.op_type for n in g.node)
    print('[ops ]', dict(c))
    bad = {k: v for k, v in c.items() if k not in ALLOWED}
    print('[ops ] 不支持:', bad if bad else '无 ★')


if __name__ == '__main__':
    main()
