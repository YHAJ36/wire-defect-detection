# -*- coding: utf-8 -*-
"""生成"未训练（@0 轮）"的 CBAM 权重，供 sa_input_probe 测量。

为什么要它：@0 轮模型没有 runs/<name> 目录，`load_model(run)` 找不到 best.pt。
但 0/5/60 三列的增长率对比**必须用同一把尺子**——`check_cbam_fixed.py` 当初也测过 @0，
可它是在**含 padding** 的图上算的，且只记带符号的 z 均值；
`sa_input_probe` 是在**去 padding** 的图上算的。两者相减会把工具偏差混进"增长率"里。

所以这里存两份 @0 权重，交给 `sa_input_probe` 测，保证与 @5/@60 同口径。

构造方式与训练起点**逐项一致**（复刻 ultralytics 的 trainer 做法）：
    从 yolo11s.pt 自带 yaml 建 DetectionModel(nc=3) → load 预训练权重 → 注入 CBAM
seed 固定，两臂的 CBAM 初值相同（同样的 nn.Conv2d(2,1,7,7) → 同一 kaiming 抽样）。

产物（刻意放在 logs/_probe0/ 而不是 runs/，避免污染 run 列表）：
    logs/_probe0/orig0.pt   原版 cbam.py 注入
    logs/_probe0/fixed0.pt  cbam_fixed.py 注入
"""
import sys
from pathlib import Path

for _d in ('D:/2/code/yolov11', 'D:/2/code/tools'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from cbam import CBAM
from cbam_fixed import CBAM as CBAM_FIX

WEIGHTS = 'D:/1/models/yolo11s.pt'
OUT = 'D:/2/logs/_probe0'
IDX = [2, 4, 6, 8, 9]          # mode=both：backbone [2,4,6,8] + SPPF [9]
NC = 3


def get_channels(m):
    for name in ('cv2', 'cv3', 'cv4'):
        conv = getattr(m, name, None)
        if conv is None:
            continue
        if isinstance(conv, nn.Conv2d):
            return conv.out_channels
        inner = getattr(conv, 'conv', None)
        if isinstance(inner, nn.Conv2d):
            return inner.out_channels
    for sub in reversed(list(m.modules())):
        if isinstance(sub, nn.Conv2d):
            return sub.out_channels
    raise ValueError('no conv')


def build(Cls, seed=0):
    cfg = YOLO(WEIGHTS).model.yaml
    torch.manual_seed(seed)
    m = DetectionModel(cfg, nc=NC, verbose=False)
    from ultralytics.nn.tasks import attempt_load_one_weight
    m.load(attempt_load_one_weight(WEIGHTS)[0], verbose=False)
    for i in IDX:
        ch = get_channels(m.model[i])
        cbam = Cls(ch)
        w = nn.Sequential(m.model[i], cbam)
        w.f, w.i = m.model[i].f, m.model[i].i
        if hasattr(m.model[i], 'type'):
            w.type = m.model[i].type
        m.model[i] = w
    m.eval()
    return m


def main():
    Path(OUT).mkdir(parents=True, exist_ok=True)
    for tag, Cls in (('orig0', CBAM), ('fixed0', CBAM_FIX)):
        m = build(Cls)
        n = sum(p.numel() for p in m.parameters())
        p = f'{OUT}/{tag}.pt'
        torch.save({'model': m}, p)
        print(f'{tag}: 参数量 {n:,}  ->  {p}')


if __name__ == '__main__':
    main()
