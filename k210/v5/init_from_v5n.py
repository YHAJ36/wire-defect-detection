# -*- coding: utf-8 -*-
"""用 yolov5n 官方预训练权重给 width 0.125 的小模型做"通道切片"初始化。

原理
----
width 0.25 -> 0.125 时，每一层的通道数正好减半（64->32? 不，是 16->8 这种，
因为两层都过 make_divisible(≥8)，实测处处严格 2:1）。
所以可以把 [Cout, Cin, k, k] 的卷积核取 [:, :, :, :] 的左上角子块，
BN 的 4 个参数取前一半通道 —— 这不是数学等价的裁剪，但作为初始化，
能继承预训练学到的低层滤波器，比随机初始化收敛快很多。

Detect 头不继承（nc 从 80 变 1，形状对不上），让它随机初始化。

用法：
    python init_from_v5n.py            # 生成 v5n125_init.pt
然后训练时 --weights D:/2/k210/v5/v5n125_init.pt
"""
import os
import sys

import torch
import torch.nn as nn
import yaml

Y5 = r'D:/2/code/baselines/yolov5'
sys.path.insert(0, Y5)
os.chdir(Y5)

from models.common import SPPF, SPPFK, Up2  # noqa: E402
from models.yolo import Model           # noqa: E402
from models.experimental import attempt_load  # noqa: E402


def build_dst():
    cfg = yaml.safe_load(open(r'D:/2/k210/v5/models/yolov5n125.yaml', encoding='utf-8'))
    m = Model(cfg).float()
    # 和训练时一致的结构改造
    def sw(module):
        for nm, c in module.named_children():
            if c.__class__.__name__ == 'SiLU':
                setattr(module, nm, nn.ReLU(inplace=True))
            else:
                sw(c)
    sw(m)
    for mod in m.modules():
        if mod.__class__ is SPPF:
            mod.__class__ = SPPFK
    for nm, c in m.named_children():
        pass
    # Upsample 保留标准 nn.Upsample 也能训，但为了和导出完全一致，换 Up2
    def swup(module):
        for nm, c in module.named_children():
            if c.__class__.__name__ == 'Upsample':
                new = Up2()
                new.f, new.i, new.type, new.np = c.f, c.i, c.type, 0
                setattr(module, nm, new)
            else:
                swup(c)
    swup(m)
    return m


def main():
    src = attempt_load(r'D:/2/k210/v5/yolov5n.pt', device='cpu', inplace=False, fuse=False).float()
    dst = build_dst()

    sp = src.state_dict()
    dp = dst.state_dict()
    hit_exact = hit_slice = miss = 0
    for k, v in dp.items():
        if k.startswith('model.24.'):
            # Detect 头：源是 COCO 80 类（3*85=255 通道），目标是 1 类（3*6=18），
            # 切片继承过来的是 COCO 的类别先验，对断股无意义，让它自己初始化
            continue
        if k not in sp:
            miss += 1
            print(f'  [miss] {k} 目标 {tuple(v.shape)} 源中不存在')
            continue
        s = sp[k]
        if s.shape == v.shape:
            v.copy_(s)
            hit_exact += 1
            continue
        # 逐维取前缀（要求目标每一维都不大于源）
        if all(a <= b for a, b in zip(v.shape, s.shape)):
            sl = tuple(slice(0, a) for a in v.shape)
            v.copy_(s[sl])
            hit_slice += 1
        else:
            miss += 1
            print(f'  [skip] {k} 目标 {tuple(v.shape)} > 源 {tuple(s.shape)}')

    print(f'\n[init] 完全一致 {hit_exact} | 切片继承 {hit_slice} | 跳过 {miss}')
    print(f'[init] 目标参数量 {sum(p.numel() for p in dst.parameters())/1e6:.3f} M')

    out = r'D:/2/k210/v5/v5n125_init.pt'
    torch.save({'model': dst.half(), 'epoch': -1}, out)
    print(f'[save] {out}  ({os.path.getsize(out)/1e6:.2f} MB)')


if __name__ == '__main__':
    main()
