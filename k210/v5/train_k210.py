# -*- coding: utf-8 -*-
"""K210 可部署版 YOLOv5n 训练（width 0.125 / ReLU / 去池化 SPPF / 320 输入 / 单类 break）

为什么结构要和标准 yolov5n 不同
--------------------------------
K210 的 KPU 只有 2 MB，且 nncase 0.2 的 ONNX 导入器只认 41 个算子：
  - SiLU 会拆成 Sigmoid+Mul，Sigmoid 不在表里  -> 全网换 ReLU
  - SPPF 里的 MaxPool 让 importer 段错误(139)   -> SPPF 去池化（SPPFK）
  - nn.Upsample 会带进 Shape/Cast/Unsqueeze      -> 固定 scale=2（Up2）
这些改造必须在**训练时**就生效，否则训出来的权重和板上结构对不上。

SPPFK / Up2 定义在 models/common.py 里（不是这里），因为最佳权重是用 pickle
存整个模型对象的，类必须落在可导入的模块路径下，否则换个脚本就加载失败。

实现方式：monkey-patch models.yolo.Model.__init__，构建完立刻改造。
只在本进程内生效，不影响项目里其他训练脚本。
"""
import os
import sys

ROOT = r'D:/2/code/baselines/yolov5'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import torch.nn as nn

import models.yolo as Y
from models.common import SPPF, SPPFK, Up2


def _swap_act(module):
    """递归把 nn.SiLU 换成 nn.ReLU（nncase 0.2 唯一能吃的激活）。"""
    n = 0
    for name, child in module.named_children():
        if child.__class__.__name__ == 'SiLU':
            setattr(module, name, nn.ReLU(inplace=True))
            n += 1
        else:
            n += _swap_act(child)
    return n


def _swap_up(module):
    n = 0
    for name, child in module.named_children():
        if child.__class__.__name__ == 'Upsample':
            new = Up2()
            new.f, new.i, new.type = child.f, child.i, child.type
            new.np = 0
            setattr(module, name, new)
            n += 1
        else:
            n += _swap_up(child)
    return n


def _swap_sppf(module):
    n = 0
    for m in module.modules():
        if m.__class__ is SPPF:
            m.__class__ = SPPFK     # 换类不换 forward，deepcopy(EMA) 才安全
            n += 1
    return n


_calls = {'n': 0}


def _patched_init(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
    _orig_init(self, cfg, ch, nc, anchors)
    a = _swap_act(self)
    u = _swap_up(self)
    s = _swap_sppf(self)
    _calls['n'] += 1
    if _calls['n'] == 1:
        print(f'[k210] 结构改造：SiLU->ReLU {a} 处 | Upsample 固定 {u} 处 | SPPF 去池化 {s} 处')


_orig_init = Y.Model.__init__
Y.Model.__init__ = _patched_init


# ---------------- 训练参数 ----------------

BASE_ARGV = [
    'train.py',
    '--data', 'D:/2/k210/v5/dataset_break/data.yaml',
    '--cfg', 'D:/2/k210/v5/models/yolov5n125.yaml',
    '--epochs', '150',
    '--imgsz', '320',
    '--batch-size', '16',
    '--weights', 'D:/2/k210/v5/v5n125_init.pt',   # 官方 v5n 通道切片初始化
    '--project', 'D:/2/k210/v5/runs',
    '--name', 'v5n125_break_320',
    '--workers', '0',                # Windows spawn 下多 worker 会爆页面文件
    '--seed', '42',
    '--optimizer', 'SGD',
]

sys.argv = BASE_ARGV + sys.argv[1:]

from train import parse_opt, main  # noqa: E402
import train as _train             # noqa: E402

# 关 AMP：本项目断股目标中位仅 34x41 px，320 输入后 w/h 比值极端，
# CIoU 里的 torch.atan 在 fp16 下溢出 -> 第 2 轮 val loss 直接变 nan。
# 实测 fp32 前向/反向梯度最大 26.6、无 inf，故只关半精度，不改结构。
_train.check_amp = lambda model: False


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
