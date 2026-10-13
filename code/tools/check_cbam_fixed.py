# -*- coding: utf-8 -*-
"""CBAM 修复版 —— 训练前的**未训练前向闸门**。

目的：在花 1.2 小时训练之前，先用未训练的模型判断 cbam_fixed 到底有没有
把「SA 读出头退化」这件事修好。若未训练状态下 SA 输出仍然近常数，说明
这两处改动（中心化 + bias）没解决根本问题，训练也白跑。

判据（硬门槛，用户 2026-10-11 决策 2 变体 c）：
    SA 输出 std > 0.05  且  框内外比 > 1.1   → PASS，放行训练
    否则                                    → FAIL，改一版再测（最多 2 次尝试）

对照（已训练的原版 wire_v11_640_both，8 张测试图，来自 analyze_cbam_degradation.py）：
    层 2/4/6/8/9 输出 std = 0.0159 / 0.0823 / 0.0390 / 0.0307 / 0.0306
    框内外比      = 1.012 / 1.085 / 1.028 / 0.981 / 0.995
    z 均值(DC)    = +4.94 / +3.15 / +4.41 / −0.23 / −0.08
即：训练了 60 轮之后仍然是这个数，所以未训练状态下的读数是有参考价值的
（读出头是结构性缺陷，不靠训练自愈）。

三个模型配对构造（同一份预训练权重、同一随机种子、同样 nc=3）：
    baseline  不注入
    orig      cbam.CBAM        （原版：bias=False、无中心化）
    fixed     cbam_fixed.CBAM  （修复版：bias=True、空间中心化）

运行：
    D:/yolo-env/Scripts/python.exe D:/2/code/tools/check_cbam_fixed.py
"""
import argparse
import sys
from pathlib import Path

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12', 'D:/2/code/tools'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight

from cbam import CBAM
from cbam_fixed import CBAM as CBAM_FIX

# 复用 viz_cbam_attention 的预处理与度量，保证与已有分析同口径
from viz_cbam_attention import (IMAGES, LABELS, letterbox, read_gt, list_test_images,
                                map_to_original, contrast_ratio, locate_cbam, fmt_mm)

WEIGHTS = 'D:/1/models/yolo11s.pt'
OUT = 'D:/2/logs/cbam_fixed_gate.txt'

NC = 3
IMGSZ = 640
N_IMG = 8

# 注入配置：both = backbone [2,4,6,8] + SPPF [9]，与原版 wire_v11_640_both 一致
IDX = [2, 4, 6, 8, 9]

# 参数量验收（nc=3 口径）
P_BASE = 9428953
P_ORIG = 9513411          # baseline + both 注入 84,458
P_FIXED = 9513416         # 再多 5 个 SA conv bias

# 闸门阈值
TH_STD = 0.05
TH_RATIO = 1.1

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# 预训练 checkpoint 只读一次，三个模型共用（配对比较的前提）
_CKPT = [None]


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


def build(mode, seed=0):
    """mode: 'baseline' | 'orig' | 'fixed'。

    复刻 ultralytics 训练时的模型构造：从 checkpoint 自带 yaml 建 DetectionModel(nc=3)，
    再 load 预训练权重（打不上的头自动跳过）。这正是 epoch 0 的模型状态。
    """
    cfg = YOLO(WEIGHTS).model.yaml
    # BaseModel.load 只吃 checkpoint dict / nn.Module，不吃路径字符串
    # （ultralytics 内部由 attempt_load_one_weight 负责把路径读成 ckpt）。
    if _CKPT[0] is None:
        _CKPT[0] = attempt_load_one_weight(WEIGHTS)[0]
    torch.manual_seed(seed)
    m = DetectionModel(cfg, nc=NC, verbose=False)
    m.load(_CKPT[0], verbose=False)

    if mode != 'baseline':
        Cls = CBAM if mode == 'orig' else CBAM_FIX
        seq = m.model
        for i in IDX:
            ch = get_channels(seq[i])
            cbam = Cls(ch)
            w = nn.Sequential(seq[i], cbam)
            w.f, w.i = seq[i].f, seq[i].i
            if hasattr(seq[i], 'type'):
                w.type = seq[i].type
            seq[i] = w
    m.eval()
    return m


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def probe(m, img_names):
    """跑 N_IMG 张图，逐层记录 z（sigmoid 前）与 SA 输出的空间统计与框内外比。

    返回 {层号: {'z_mean':[...], 'z_std':[...], 'sa_std':[...], 'ratio':[...], 'parent':.., 'ch':..}}
    """
    items = locate_cbam(m)
    agg = {}
    for i, parent, cbam in items:
        agg[i] = {'parent': parent, 'fmap': None, 'z_mean': [], 'z_std': [],
                  'sa_std': [], 'sa_mean': [], 'ratio': [], 'grad': []}

    for name in img_names:
        img_bgr = read_img(name)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        boxes = read_gt(Path(LABELS) / f'{Path(name).stem}.txt', w, h)

        lb, _r, crop = letterbox(img_bgr, IMGSZ)
        canvas_hw = lb.shape[:2]
        x = torch.from_numpy(
            np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1))
        ).float().unsqueeze(0) / 255.0

        store, handles = {}, []
        for i, parent, cbam in items:
            def mk(i):
                # conv 输出即 sigmoid 前的 z（修复版的中心化发生在 conv 之前）
                def h_conv(_m, _inp, out):
                    store.setdefault(i, {})['z'] = out.detach()[0, 0].clone()
                def h_sa(_m, _inp, out):
                    store.setdefault(i, {})['sa'] = out.detach()[0, 0].clone()
                return h_conv, h_sa
            hc, hs = mk(i)
            handles.append(cbam.sa.conv.register_forward_hook(hc))
            handles.append(cbam.sa.register_forward_hook(hs))

        try:
            with torch.no_grad():
                m(x)
        finally:
            # 变量名用 hd 而非 h —— h 是本函数的图像高度，被覆盖过一次
            # （cv2.resize 报 "dsize item 1 has wrong type"，就是 h 变成了 hook 句柄）。
            for hd in handles:
                hd.remove()

        for i, _parent, _cbam in items:
            d = store.get(i, {})
            if 'z' not in d or 'sa' not in d:
                continue
            z = d['z'].numpy()
            sa = d['sa'].numpy()
            sa_orig = map_to_original(sa, canvas_hw, crop, (h, w))
            r = contrast_ratio(sa_orig, boxes)
            a = agg[i]
            if a['fmap'] is None:
                a['fmap'] = f'{z.shape[0]}x{z.shape[1]}'
            a['z_mean'].append(float(z.mean()))
            a['z_std'].append(float(z.std()))
            a['sa_mean'].append(float(sa.mean()))
            a['sa_std'].append(float(sa.std()))
            # σ'(z) = SA·(1−SA)：sigmoid 在**工作点**上的梯度增益。
            # 这才是"梯度还流不流得动"的直接读数，且未训练就能测。
            a['grad'].append(float((sa * (1.0 - sa)).mean()))
            if r is not None:
                a['ratio'].append(float(r))
    return agg


def read_img(name):
    import cv2
    return cv2.imread(str(Path(IMAGES) / name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=OUT)
    p.add_argument('--n-img', type=int, default=N_IMG)
    args = p.parse_args()

    lines = []

    def say(s=''):
        print(s)
        lines.append(s)

    say('=' * 78)
    say('CBAM 修复版 —— 未训练前向闸门')
    say('=' * 78)

    # ---------- 1. 参数量验收 ----------
    say()
    say('【1】参数量验收（nc=3 口径）')
    mb = build('baseline')
    mo = build('orig', seed=0)
    mf = build('fixed', seed=0)
    for tag, m, exp in (('baseline', mb, P_BASE), ('orig  (cbam.py)', mo, P_ORIG),
                        ('fixed (cbam_fixed.py)', mf, P_FIXED)):
        n = n_params(m)
        say(f'  {tag:22s} {n:>10,}   期望 {exp:>10,}   {"OK" if n == exp else "!! 不符"}')

    nb, no, nf = locate_cbam(mb), locate_cbam(mo), locate_cbam(mf)
    say(f'  locate_cbam 识别到的 CBAM 层数：baseline {len(nb)} / orig {len(no)} / fixed {len(nf)}')
    say(f'  → 鸭子类型判据{"有效，修复版被正确识别" if len(nf) == 5 else "失效！修复版没被识别出来"}')

    # ---------- 2. 未训练前向 ----------
    imgs = list_test_images(prefer_class=0)[:args.n_img]
    say()
    say(f'【2】未训练前向（{len(imgs)} 张测试图，imgsz={IMGSZ}）')

    res = {}
    for tag, m in (('orig', mo), ('fixed', mf)):
        res[tag] = probe(m, imgs)

    say()
    say('层 | 特征图 | 口径 | z均值(DC) | z std | SA输出std | 框内外比 | σ\'(z)梯度增益')
    say('-' * 92)
    for i in IDX:
        for tag in ('orig', 'fixed'):
            a = res[tag].get(i)
            if a is None:
                continue
            say(f'{i:>2} | {a["fmap"] or "":>7} | {tag:5s} | '
                f'{fmt_mm(a["z_mean"], 3):>16} | {fmt_mm(a["z_std"], 3):>12} | '
                f'{fmt_mm(a["sa_std"], 4):>16} | {fmt_mm(a["ratio"], 3):>16} | '
                f'{fmt_mm(a["grad"], 4):>16}')
        say('-' * 92)

    # ---------- 3. 闸门判定 ----------
    fa = res['fixed']
    std_ok = all(np.mean(fa[i]['sa_std']) > TH_STD for i in IDX if i in fa)
    ratio_ok = all(np.mean(fa[i]['ratio']) > TH_RATIO for i in IDX if i in fa)

    say()
    say('【3】闸门判定（修复版，全 5 层都要过）')
    say(f'  判据：SA 输出 std > {TH_STD}  且  框内外比 > {TH_RATIO}')
    for i in IDX:
        if i not in fa:
            continue
        s = float(np.mean(fa[i]['sa_std']))
        r = float(np.mean(fa[i]['ratio']))
        say(f'    层 {i}: std {s:.4f} {"OK" if s > TH_STD else "不过"}   '
            f'比 {r:.3f} {"OK" if r > TH_RATIO else "不过"}')
    say()
    say(f'  std  全层通过：{"是" if std_ok else "否"}')
    say(f'  比值 全层通过：{"是" if ratio_ok else "否"}')
    if std_ok and ratio_ok:
        say('  闸门结果：PASS —— 放行训练（D:/2/code/yolov11/train_v11_640_cbamfix.py）')
    else:
        say('  闸门结果：FAIL —— 不放行训练。按决策 2 自行调整（最多 2 次尝试：')
        say('            调大 kernel，或换非饱和激活 tanh/ReLU）后重测。')
    say('=' * 78)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {args.out}')


if __name__ == '__main__':
    main()
