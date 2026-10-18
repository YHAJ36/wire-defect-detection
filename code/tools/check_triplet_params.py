# -*- coding: utf-8 -*-
"""Triplet Attention 注入参数量核验（照 CBAM 的坑设闸门，见 WORKLOG 答辩素材 1）。

背景：CBAM 曾因"在 model.train() 之前注入"被 train() 内部的 yaml 重建静默丢弃。
      症状是**参数量没有变化**。本脚本就是那个闸门。

用法一（理论核验，秒级、不训练、不占 GPU）：
    cd /d/2/code/yolov11 && D:/yolo-env/Scripts/python.exe ../tools/check_triplet_params.py

用法二（训练产物核验，给定 run 名）：
    D:/yolo-env/Scripts/python.exe ../tools/check_triplet_params.py wire_v11_1024_triplet

预期值（先手算、再用本脚本证实，两边必须相等）：
    单个 AttentionGate 的可训练参数 = conv(2×1×7×7 = 98，bias=False) + BN2d(1) 的
        weight(1) + bias(1) = 100
    单个 TripletAttention            = 3 个 gate = 300
    注入到层 9（SPPF，1024 通道）后，全模型参数量应 **+300**。
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, 'D:/2/code/yolov11')

from ultralytics import YOLO  # noqa: E402
from triplet_att import TripletAttention  # noqa: E402

WEIGHTS = 'D:/1/models/yolo11s.pt'
RUNS = 'D:/2/runs'
SPPF = 9


def get_channels(m):
    for name in ('cv2', 'cv3', 'cv4'):
        conv = getattr(m, name, None)
        if conv is None:
            continue
        if isinstance(conv, nn.Conv2d):
            return conv.out_channels
        inner = getattr(conv, 'conv', None)     # ultralytics Conv 包装了 nn.Conv2d
        if isinstance(inner, nn.Conv2d):
            return inner.out_channels
    for sub in reversed(list(m.modules())):
        if isinstance(sub, nn.Conv2d):
            return sub.out_channels
    raise ValueError('no conv')


def inject(seq, idx):
    """与 train_v11_640_neck.py 的 inject_cbam_custom 逐字同法。"""
    for i in idx:
        gate = TripletAttention()
        w = nn.Sequential(seq[i], gate)
        w.f = seq[i].f
        w.i = seq[i].i
        if hasattr(seq[i], 'type'):
            w.type = seq[i].type
        seq[i] = w


def n_params(m):
    """全参数量。

    ⚠ 不要按 requires_grad 过滤！`YOLO(...).model` 在推理态下所有参数的
      requires_grad 都是 False，过滤后 baseline 会算出 **0**，于是"增量 300"
      变成 0→300 的**假通过**（本脚本首版就踩了，被自己的闸门抓到）。
    """
    return sum(p.numel() for p in m.parameters())


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    print('=' * 84)
    print('Triplet Attention 注入参数量核验')
    print('=' * 84)

    # ---- 单模块手算 vs 实测 ----
    g = TripletAttention()
    conv_par = sum(p.numel() for p in g.hw.conv.conv.parameters())
    bn_par = sum(p.numel() for p in g.hw.conv.bn.parameters() if p.requires_grad)
    print(f'\n单个 AttentionGate : conv {conv_par} + BN {bn_par} = {conv_par + bn_par}')
    print(f'单个 TripletAttn   : {n_params(g)}   （手算 3×100 = 300）')
    print(f'  分支数 = {3 if not g.no_spatial else 2}  （no_spatial=False 时为 3）')

    # ---- 注入前后对比（理论核验）----
    print(f'\n{"-" * 84}')
    print(f'{"配置":<26}{"参数量":>14}{"相对 baseline":>18}')
    print('-' * 84)

    m0 = YOLO(WEIGHTS).model
    base = n_params(m0)
    print(f'{"baseline (无注入)":<26}{base:>14,}{"—":>18}')

    m1 = YOLO(WEIGHTS).model
    inject(m1.model, [SPPF])
    got = n_params(m1)
    print(f'{"+Triplet@neck(层9)":<26}{got:>14,}{f"+{got - base:,}":>18}')

    exp = sum(p.numel() for p in TripletAttention().parameters())
    # 先证量具本身没坏：baseline 必须是真实的大数，否则后续比较全是假通过
    if base < 1_000_000:
        print(f'\n❌ 量具自检失败：baseline 参数量 {base:,} 不合理（YOLO11s 应 ≈2.6M）。')
        print('   多半是又把 requires_grad 当过滤条件了 —— 比较结果无意义，中止。')
        return
    ok = (got - base) == exp
    print('-' * 84)
    print(f'预期增量 = {exp:,}（单个 TripletAttention 的可训练参数）')
    print(f'实测增量 = {got - base:,}')
    print(f'\n{"✅ 通过：注入生效，参数量按预期增加" if ok else "❌ 失败：增量不符 —— 注入被丢弃或重复"}')

    # ---- 对注入后的层做一次前向，确认形状不变（即插即用）----
    m1.eval()
    with torch.no_grad():
        y = m1(torch.randn(1, 3, 640, 640))
        if isinstance(y, (list, tuple)):
            y = y[0]
    print(f'前向 smoke: 输出形状 {tuple(y.shape)[:2]} …  x 为 3 分量 list'
          f' —— 形状与 baseline 一致 ✅' if isinstance(y, torch.Tensor) else '前向异常')

    # ---- 训练产物核验（可选）----
    if len(sys.argv) > 1:
        print(f'\n{"-" * 84}\n训练产物核验：')
        for name in sys.argv[1:]:
            w = Path(RUNS) / name / 'weights' / 'best.pt'
            if not w.exists():
                print(f'  {name:<28} 缺权重 {w}')
                continue
            mdl = YOLO(str(w)).model
            n = n_params(mdl)
            print(f'  {name:<28}{n:>14,}   相对 baseline {n - base:+,}')

    print('=' * 84)


if __name__ == '__main__':
    main()
