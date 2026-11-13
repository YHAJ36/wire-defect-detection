# -*- coding: utf-8 -*-
"""
K210 专用 YOLOv5 ONNX 导出器
============================
为什么不能用官方 export.py：
  nncase 0.2（K210 唯一能出 kmodel V4 的工具链）只认这 41 个 ONNX 算子：
    Abs Add AveragePool Ceil Concat Constant Conv ConvTranspose Cos Div Exp Floor
    LpNormalization LeakyRelu Log MatMul MaxPool Max Min Mul Neg Pad ReduceMax
    ReduceMean ReduceMin ReduceSum Relu Reshape Resize Sin Slice Softmax
    SpaceToDepth Sqrt Squeeze Sub Transpose
  官方导出会引入三个它不认的算子：
    1) SiLU 激活  -> 拆成 Sigmoid + Mul          （Sigmoid 没有）
    2) Detect 头  -> x.sigmoid().split(...)      （Sigmoid + Split 都没有）
    3) wh=(w*2)**2*anchor                        （Pow 没有）
  所以这里做两件事：
    A. 把全网的 SiLU 换成 LeakyReLU(0.1)          —— 只用它认识的激活
    B. 把 Detect 头砍成"纯卷积输出"，不做 sigmoid/解码
       anchor 解码、sigmoid、NMS 全部留给板子上的 Python 做
  输出 3 个尺度，形状 (1, na*no, ny, nx)，na=3，no=5+nc
"""
import sys, os, argparse
import torch
import torch.nn as nn

Y5 = r'D:/2/code/baselines/yolov5'      # 本地 yolov5 源码
sys.path.insert(0, Y5)
os.chdir(Y5)                            # yolov5 内部有相对路径引用

from models.yolo import Detect          # noqa: E402
from models.experimental import attempt_load  # noqa: E402


def swap_silu(module, act='relu'):
    """递归把 nn.SiLU 替换成 K210 能吃的激活，返回替换个数。
    relu  : nn.ReLU            —— nncase 0.2 实测可用（KPU 原生带 ReLU）
    leaky : nn.LeakyReLU(0.1)  —— 实测报 Shapes must be same，不可用，留作对照"""
    n = 0
    make = (lambda: nn.ReLU(inplace=True)) if act == 'relu' else (lambda: nn.LeakyReLU(0.1, inplace=True))
    for name, child in module.named_children():
        if child.__class__.__name__ == 'SiLU':
            setattr(module, name, make())
            n += 1
        else:
            n += swap_silu(child, act)
    return n


class Up2(nn.Module):
    """固定 2 倍最近邻上采样。
    用 recompute_scale_factor=False 让 ONNX 导出直接写常量 scales=[1,1,2,2]，
    否则 torch 会生成 Shape->Cast->Slice->Concat->Unsqueeze 一整套动态算尺寸的子图，
    而 nncase 0.2 不认 Shape / Cast / Unsqueeze 这三个算子。"""

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x, scale_factor=2, mode='nearest', recompute_scale_factor=False)


def swap_upsample(module):
    n = 0
    for name, child in module.named_children():
        if child.__class__.__name__ == 'Upsample':
            new = Up2()
            # yolov5 的 _forward_once 依赖 m.f / m.i / m.type / m.np 这四个属性
            new.f, new.i, new.type = child.f, child.i, child.type
            new.np = sum(p.numel() for p in new.parameters())
            setattr(module, name, new)
            n += 1
        else:
            n += swap_upsample(child)
    return n


def patch_maxpool(module):
    """SPPF 里串联的 MaxPool 会让 nncase 0.2 在 Import graph 阶段段错误(139)，
    5x5 和 3x3 都崩 —— 它的 ONNX MaxPool 导入路径是坏的。
    所以直接把 SPPF.forward 改成"不复用池化"的版本：
        cv1 -> [y, y, y, y] -> concat -> cv2
    等价于去掉了多尺度池化，感受野变小，重训时网络自己补偿。
    nn.Module.forward 在实例上被覆盖，不影响保存/加载权重。"""
    import types

    def no_pool_forward(self, x):
        y = [self.cv1(x)]
        y.extend([y[-1]] * 3)                  # 用复制代替 3 次 maxpool 串联
        return self.cv2(torch.cat(y, 1))

    n = 0
    for m in module.modules():
        if m.__class__.__name__ == 'SPPF':
            m.forward = types.MethodType(no_pool_forward, m)
            n += 1
    return n


def raw_forward(self, x):
    """替换 Detect.forward：只吐 3 个尺度的原始卷积输出，不做任何解码。"""
    return tuple(self.m[i](x[i]) for i in range(self.nl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=r'D:/2/k210/v5/yolov5n.pt')
    ap.add_argument('--imgsz', default='224x320', help='HxW，必须被 32 整除')
    ap.add_argument('--out', default=r'D:/2/k210/v5/yolov5n_224x320_raw.onnx')
    ap.add_argument('--act', default='relu', choices=['relu','leaky'])
    ap.add_argument('--opset', type=int, default=12, help='ONNX opset，nncase 0.2 对 12 可能崩，可退 11')
    a = ap.parse_args()

    H, W = (int(v) for v in a.imgsz.lower().split('x'))
    assert H % 32 == 0 and W % 32 == 0, '尺寸必须能被 32 整除'
    assert W <= 320 and H <= 240, 'K210 KPU 特征图上限 320x240'

    # ---- 1. 载模型 ----
    model = attempt_load(a.weights, device='cpu', inplace=True, fuse=True)
    model = model.float().eval()
    nc = model.model[-1].nc
    print(f'[load] {a.weights}  nc={nc}  na={model.model[-1].na}')

    # ---- 2. SiLU -> LeakyReLU ----
    cnt = swap_silu(model, a.act)
    print(f'[act ] SiLU -> {a.act} 共替换 {cnt} 处')
    print(f'[up  ] Upsample -> 固定 scale 共替换 {swap_upsample(model)} 处')
    print(f'[pool] SPPF 去池化 共替换 {patch_maxpool(model)} 处')

    # ---- 3. Detect 头换成 raw ----
    Detect.forward = raw_forward

    dummy = torch.zeros(1, 3, H, W)
    with torch.no_grad():
        outs = model(dummy)
    for i, o in enumerate(outs):
        print(f'[head] out{i} {tuple(o.shape)}   (1, {model.model[-1].na}x{model.model[-1].no}, '
              f'{o.shape[2]}, {o.shape[3]})')

    # ---- 4. 导出 ----
    torch.onnx.export(
        model, (dummy,), a.out,
        opset_version=a.opset,
        input_names=['images'],
        output_names=[f'p{i}' for i in range(len(outs))],
        do_constant_folding=True,
        dynamo=False,   # 走旧 TorchScript exporter：只依赖 onnx，不需要 onnxscript
    )
    print(f'[onnx] -> {a.out}  ({os.path.getsize(a.out)/1e6:.2f} MB)')

    # ---- 5. 算子体检 ----
    import onnx
    m = onnx.load(a.out)
    from collections import Counter
    c = Counter(n.op_type for n in m.graph.node)
    allowed = set("""Abs Add AveragePool Ceil Concat Constant Conv ConvTranspose Cos Div Exp
        Floor LpNormalization LeakyRelu Log MatMul MaxPool Max Min Mul Neg Pad ReduceMax
        ReduceMean ReduceMin ReduceSum Relu Reshape Resize Sin Slice Softmax SpaceToDepth
        Sqrt Squeeze Sub Transpose""".split())
    print('[ops ]', dict(c))
    bad = {k: v for k, v in c.items() if k not in allowed}
    print('[ops ] 不支持:', bad if bad else '无 ★')


if __name__ == '__main__':
    main()
