# -*- coding: utf-8 -*-
"""从缓存重新解码，扫 NMS 阈值，确认脚本与官方口径的差异来源。

背景：`eval_subset_map.py` 在 test 203 张上算 ONNX mAP50 = 0.6950，
而 ultralytics 同一份权重同一份数据是 0.7189，差 2.4 pt。
怀疑是 NMS 阈值不同（脚本 0.45，官方默认 0.7）。

缓存 `map_cache.npz` 里存的是**模型原始输出**（o_ 前缀=ONNX，k_ 前缀=kmodel），
所以换阈值只需重新 decode + 算 AP，不用再跑一遍推理（省 1 小时）。

用法
    set DOTNET_ROLL_FORWARD=LatestMajor
    D:/yolo-env/Scripts/python.exe D:/2/deploy/rescore_cache.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'D:/2/deploy')
import eval_subset_map as E          # 复用 load_img / load_gt / decode / ap50

IMG = Path('D:/2/dataset_noleak/images/test')
LBL = Path('D:/2/dataset_noleak/labels/test')
CACHE = 'D:/2/deploy/map_cache.npz'

z = np.load(CACHE)
names = sorted(k[2:] for k in z.files if k.startswith('o_'))
print(f'缓存 {len(names)} 张\n')

# 图只读一次，geo/imsz/GT 与阈值无关
pre = []
for name in names:
    p = IMG / name
    canvas, geo, imsz = E.load_img(p)
    if canvas is None:
        continue
    pre.append((name, geo, imsz, E.load_gt(LBL / (p.stem + '.txt'), imsz)))

print(f'{"NMS":>6}{"ONNX":>10}{"kmodel":>10}{"Δ":>9}')
for thr in (0.45, 0.5, 0.6, 0.65, 0.7, 0.75):
    po, pk, gts = [], [], []
    for name, geo, imsz, gt in pre:
        po.append(E.decode(z['o_' + name], geo, imsz, iou_thr=thr))
        pk.append(E.decode(z['k_' + name], geo, imsz, iou_thr=thr))
        gts.append(gt)
    _, mo, _ = E.ap50(po, gts)
    _, mk, _ = E.ap50(pk, gts)
    print(f'{thr:>6.2f}{mo:>10.4f}{mk:>10.4f}{mk - mo:>+9.4f}')

print('\n参照：ultralytics 官方 test 口径 mAP50 = 0.7189（iou=0.7, conf=0.001）')
