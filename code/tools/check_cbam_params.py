# -*- coding: utf-8 -*-
"""CBAM 注入参数量核验工具。

用途一：理论核验（秒级，不训练）——构建 YOLO11s 并按各 mode 注入，打印参数量与增量。
    cd /d/2/code/yolov11 && D:/yolo-env/Scripts/python.exe ../tools/check_cbam_params.py

用途二：训练产物核验——给定 run 名（首个视为 baseline 基准），读取 best.pt 数参数量。
    D:/yolo-env/Scripts/python.exe ../tools/check_cbam_params.py \
        smoke_v11_640_adamw smoke_v11_640_backbone_adamw \
        smoke_v11_640_neck_adamw smoke_v11_640_both_adamw

注意：backbone / neck_sppf / both 注入的 CBAM 模块数不同（4 / 1 / 5），
      参数量增量本就不相同，不能用一个固定阈值衡量三组。
"""
import sys
import torch.nn as nn

sys.path.insert(0, 'D:/2/code/yolov11')  # 使 `import cbam` 可解析

from ultralytics import YOLO  # noqa: E402
from cbam import CBAM  # noqa: E402

WEIGHTS = 'D:/1/models/yolo11s.pt'
BACKBONE = [2, 4, 6, 8]
SPPF = 9


def get_channels(m):
    for name in ('cv2', 'cv3', 'cv4'):
        conv = getattr(m, name, None)
        if conv is None:
            continue
        if isinstance(conv, nn.Conv2d):
            return conv.out_channels
        inner = getattr(conv, 'conv', None)  # ultralytics Conv 包装了 nn.Conv2d
        if isinstance(inner, nn.Conv2d):
            return inner.out_channels
    for sub in reversed(list(m.modules())):
        if isinstance(sub, nn.Conv2d):
            return sub.out_channels
    raise ValueError('no conv')


def inject(model, mode):
    seq = model.model
    if mode == 'baseline':
        return []
    if mode == 'backbone':
        idx = BACKBONE
    elif mode == 'neck_sppf':
        idx = [SPPF]
    else:
        idx = BACKBONE + [SPPF]
    for i in idx:
        ch = get_channels(seq[i])
        cbam = CBAM(ch)
        w = nn.Sequential(seq[i], cbam)
        w.f = seq[i].f
        w.i = seq[i].i
        if hasattr(seq[i], 'type'):
            w.type = seq[i].type
        seq[i] = w
    return idx


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    if len(sys.argv) > 1:
        print(f'{"run":<36} {"params":>12} {"delta":>10}')
        base = None
        for n in sys.argv[1:]:
            mm = YOLO(f'D:/2/runs/{n}/weights/best.pt').model
            p = n_params(mm)
            if base is None:
                base = p
            print(f'{n:<36} {p:>12,} {p - base:>+10,}')
        return

    print('=== 理论核验：构建 YOLO11s + 注入（不训练）===')
    base = n_params(YOLO(WEIGHTS).model)
    print(f'{"mode":<12} {"params":>12} {"delta":>10}  injected')
    print(f'{"baseline":<12} {base:>12,} {0:>+10,}  -')
    for mode in ('backbone', 'neck_sppf', 'both'):
        mm = YOLO(WEIGHTS).model
        idx = inject(mm, mode)
        p = n_params(mm)
        print(f'{mode:<12} {p:>12,} {p - base:>+10,}  {idx}')


if __name__ == '__main__':
    main()
