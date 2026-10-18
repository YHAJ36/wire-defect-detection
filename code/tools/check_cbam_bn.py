# -*- coding: utf-8 -*-
"""CBAM+BN 的**未训练前向闸门** —— 在花 1.2h 训练之前，先证"读出头 BN 确实消了 DC"。

为什么要专门写一版（与 check_cbam_fixed.py 的差别）
    ① **必须在 train() 态测**。BN 在推理态用 running 统计；未训练的 BN
       running_mean=0 / running_var=1 ⇒ `(z−0)/√(1+ε)` **近似恒等**，
       在 eval 态根本测不出 BN 的作用（这会让闸门给出**假 FAIL**）。
       cbam_fixed 的两处改动与 BN 无关，故旧闸门在 eval 态测没有这个问题；
       本文件把两种态都测出来对照，正是为了把这个坑钉死。
    ② 快照位置不同：cbam_fixed 抓 `sa.conv` 输出（= sigmoid 前 z，因为它的中心化在
       conv 之前）；本文件对 cbam_bn 抓 `sa.bn` 输出（= 真正的 sigmoid 前 z）。

三项核验
    【1】参数量：增量 == 注入模块的参数量之和（BN 每层 +2）。
    【2】train 态前向：`sigmoid` 前 z 的**批次均值 ≈ 0**（消 DC），
         SA 输出 std 显著高于原版 CBAM（钉尺度、不塌成常数）。
    【3】eval 态对照：证明未训练 BN 在 eval 态 ≈ 恒等（所以闸门必须 train 态）。

对照（已训练的原版 wire_v11_640_both，8 张测试图，来自 analyze_cbam_degradation.py）：
    层 2/4/6/8/9 输出 std = 0.0159 / 0.0823 / 0.0390 / 0.0307 / 0.0306
    框内外比      = 1.012 / 1.085 / 1.028 / 0.981 / 0.995
    z 均值(DC)    = +4.94 / +3.15 / +4.41 / −0.23 / −0.08
即训练 60 轮后仍是这个读数 —— 读出头是结构性缺陷，不靠训练自愈。

    ⚠ 本闸门只作"结构正确、未训练就不退化"的 sanity，**不作训练放行硬门槛**
      （WORKLOG 已记：旧闸门对"训练中才长出来的病"无效）。

运行：
    D:/yolo-env/Scripts/python.exe D:/2/code/tools/check_cbam_bn.py
"""
import argparse
import sys
from pathlib import Path

for _d in ('D:/2/code/yolov11', 'D:/2/code/v12', 'D:/2/code/yolov12', 'D:/2/code/tools'):
    if _d not in sys.path:
        sys.path.insert(0, _d)
sys.path.insert(0, 'D:/2/code/yolov11')

import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight

from cbam import CBAM as CBAM_ORIG
from cbam_bn import CBAM as CBAM_BN

# 复用 viz_cbam_attention 的预处理与度量，与既有分析同口径
from viz_cbam_attention import (IMAGES, LABELS, letterbox, read_gt, list_test_images,
                                map_to_original, contrast_ratio, locate_cbam, fmt_mm)

WEIGHTS = 'D:/1/models/yolo11s.pt'
OUT = 'D:/2/logs/cbam_bn_gate.txt'

NC = 3
IMGSZ = 640
N_IMG = 8

# 按**训练配置**核验参数量：neck_sppf = 只注入 SPPF(层 9)；
# 但**诊断 probe** 仍注入 5 层，以便与原版 CBAM 逐层对照（训练只用层 9）。
TRAIN_IDX = [9]
PROBE_IDX = [2, 4, 6, 8, 9]

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

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


def build(mode, idx, seed=0, eval_mode=True):
    """mode: 'baseline' | 'orig'(cbam.py) | 'bn'(cbam_bn.py)。

    复刻 ultralytics 训练时的模型构造：从 checkpoint 自带 yaml 建 DetectionModel(nc=3)，
    再 load 预训练权重。这正是 epoch 0 的模型状态。
    """
    cfg = YOLO(WEIGHTS).model.yaml
    if _CKPT[0] is None:
        _CKPT[0] = attempt_load_one_weight(WEIGHTS)[0]
    torch.manual_seed(seed)
    m = DetectionModel(cfg, nc=NC, verbose=False)
    m.load(_CKPT[0], verbose=False)

    if mode != 'baseline':
        Cls = CBAM_ORIG if mode == 'orig' else CBAM_BN
        seq = m.model
        for i in idx:
            ch = get_channels(seq[i])
            attn = Cls(ch)
            w = nn.Sequential(seq[i], attn)
            w.f, w.i = seq[i].f, seq[i].i
            if hasattr(seq[i], 'type'):
                w.type = seq[i].type
            seq[i] = w
    (m.train() if not eval_mode else m.eval())
    return m


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def pre_sigmoid(sa):
    """sigmoid 之前的那个模块：cbam_bn 有 bn，原版没有（退化为 conv）。"""
    return getattr(sa, 'bn', None) if getattr(sa, 'bn', None) is not None else sa.conv


def probe(m, img_names):
    """逐层记录 sigmoid 前置 z、SA 输出、框内外比。train 态下 BN 才生效。"""
    items = locate_cbam(m)
    agg = {i: {'parent': p, 'fmap': None, 'z_mean': [], 'z_std': [],
               'sa_std': [], 'sa_mean': [], 'ratio': [], 'grad': []}
           for i, p, _c in items}

    for name in img_names:
        import cv2
        img_bgr = cv2.imread(str(Path(IMAGES) / name))
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
        for i, _p, cbam in items:
            def mk(i, cbam):
                def h_pre(_m, _inp, out):
                    store.setdefault(i, {})['z'] = out.detach()[0, 0].clone()
                def h_sa(_m, _inp, out):
                    store.setdefault(i, {})['sa'] = out.detach()[0, 0].clone()
                return h_pre, h_sa
            hp, hs = mk(i, cbam)
            handles.append(pre_sigmoid(cbam.sa).register_forward_hook(hp))
            handles.append(cbam.sa.register_forward_hook(hs))

        try:
            with torch.no_grad():
                m(x)
        finally:
            for hd in handles:
                hd.remove()

        for i, _p, _c in items:
            d = store.get(i, {})
            if 'z' not in d or 'sa' not in d:
                continue
            z, sa = d['z'].numpy(), d['sa'].numpy()
            r = contrast_ratio(map_to_original(sa, canvas_hw, crop, (h, w)), boxes)
            a = agg[i]
            if a['fmap'] is None:
                a['fmap'] = f'{z.shape[0]}x{z.shape[1]}'
            a['z_mean'].append(float(z.mean()))
            a['z_std'].append(float(z.std()))
            a['sa_mean'].append(float(sa.mean()))
            a['sa_std'].append(float(sa.std()))
            # σ'(z) = SA·(1−SA)：sigmoid 在工作点上的梯度增益（梯度还流不流得动）
            a['grad'].append(float((sa * (1.0 - sa)).mean()))
            if r is not None:
                a['ratio'].append(float(r))
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=OUT)
    p.add_argument('--n-img', type=int, default=N_IMG)
    args = p.parse_args()

    lines = []

    def say(s=''):
        print(s)
        lines.append(s)

    say('=' * 96)
    say('CBAM+BN —— 未训练前向闸门')
    say('=' * 96)

    # ---------- 1. 参数量验收（按训练配置 neck_sppf）----------
    say()
    say('【1】参数量验收（nc=3；训练配置 = 只注入 SPPF 层 9）')
    mb = build('baseline', TRAIN_IDX)
    mo_t = build('orig', TRAIN_IDX, seed=0)
    mn_t = build('bn', TRAIN_IDX, seed=0)
    nb, no, nn_ = n_params(mb), n_params(mo_t), n_params(mn_t)

    # 期望增量：用**独立新建**的模块实例算，不写死魔数
    cfg = YOLO(WEIGHTS).model.yaml
    ch9 = get_channels(YOLO(WEIGHTS).model.model[9])
    exp_o = sum(q.numel() for q in CBAM_ORIG(ch9).parameters())
    exp_n = sum(q.numel() for q in CBAM_BN(ch9).parameters())
    say(f'  层 9 通道数 = {ch9}')
    say(f'  {"配置":<28}{"参数量":>12}{"增量":>10}{"期望":>10}{"判定":>8}')
    say('  ' + '-' * 70)
    for tag, n, exp in (('baseline（无注入）', nb, 0),
                        ('+CBAM 原版@层9', no, exp_o),
                        ('+CBAM+BN @层9', nn_, exp_n)):
        d = n - nb
        say(f'  {tag:<28}{n:>12,}{d:>+10,}{exp:>10,}'
            f'{("OK" if d == exp else "不符"):>8}')
    say(f'  ⇒ BN 的代价 = {exp_n - exp_o} 参数（每层 1 个 BN2d 的 weight+bias）')

    say()
    say(f'  locate_cbam 识别层数（训练配置）：baseline {len(locate_cbam(mb))} / '
        f'orig {len(locate_cbam(mo_t))} / bn {len(locate_cbam(mn_t))}')
    say(f'  → 鸭子类型判据{"有效" if len(locate_cbam(mn_t)) == 1 else "失效！"}')

    # ---------- 2. train 态前向（BN 生效）----------
    imgs = list_test_images(prefer_class=0)[:args.n_img]
    say()
    say(f'【2】未训练前向 —— **train 态**（BN 用 batch 统计，{len(imgs)} 张测试图，'
        f'imgsz={IMGSZ}，诊断注入 {PROBE_IDX} 层）')
    say('     ⚠ batch=1：BN 实际只用**空间**统计（H×W）做归一 —— 对"消空间 DC"而言足够。')
    mo = build('orig', PROBE_IDX, seed=0, eval_mode=False)
    mn = build('bn', PROBE_IDX, seed=0, eval_mode=False)
    res = {'orig': probe(mo, imgs), 'bn': probe(mn, imgs)}

    say()
    say('层 | 特征图 | 口径 | sigmoid前 z 均值(DC) | z std | SA输出std | 框内外比 | σ\'(z)')
    say('-' * 96)
    for i in PROBE_IDX:
        for tag in ('orig', 'bn'):
            a = res[tag].get(i)
            if a is None:
                continue
            say(f'{i:>2} | {a["fmap"] or "":>7} | {tag:4s} | '
                f'{fmt_mm(a["z_mean"], 3):>20} | {fmt_mm(a["z_std"], 3):>11} | '
                f'{fmt_mm(a["sa_std"], 4):>15} | {fmt_mm(a["ratio"], 3):>15} | '
                f'{fmt_mm(a["grad"], 4):>15}')
        say('-' * 96)

    # ---------- 3. eval 态对照（证明闸门必须 train 态）----------
    say()
    say('【3】eval 态对照（未训练 BN 的 running_mean=0/running_var=1 ≈ 恒等）')
    mn_e = build('bn', PROBE_IDX, seed=0, eval_mode=True)
    res_e = probe(mn_e, imgs)
    say(f'  {"层":>3} | {"train 态 z 均值":>22} | {"eval 态 z 均值":>22} | 说明')
    say('  ' + '-' * 78)
    for i in PROBE_IDX:
        a_t, a_e = res['bn'].get(i), res_e.get(i)
        if a_t is None or a_e is None:
            continue
        zt, ze = float(np.mean(a_t['z_mean'])), float(np.mean(a_e['z_mean']))
        say(f'  {i:>3} | {zt:>22.4f} | {ze:>22.4f} | '
            f'{"train 态被压到 0" if abs(zt) < 1e-6 else "train 态未压到 0!"}')

    # ---------- 4. 判定 ----------
    say()
    say('【4】判定')
    dev = []
    for i in PROBE_IDX:
        a = res['bn'].get(i)
        if a is None:
            continue
        dev.append(abs(float(np.mean(a['z_mean']))))
    z_ok = max(dev) < 1e-6 if dev else False
    std_bn = float(np.mean([np.mean(res['bn'][i]['sa_std']) for i in PROBE_IDX
                            if i in res['bn']]))
    std_o = float(np.mean([np.mean(res['orig'][i]['sa_std']) for i in PROBE_IDX
                           if i in res['orig']]))
    say(f'  train 态 z 均值全层 ≈ 0（消 DC）      ：{"PASS" if z_ok else "FAIL"}'
        f'   最大偏离 {max(dev) if dev else float("nan"):.2e}')
    say(f'  SA 输出 std：CBAM+BN {std_bn:.4f}  vs  CBAM {std_o:.4f}'
        f'   （理论 std(sigmoid(N(0,1)))=0.2082）')
    say(f'  尺度钉住（BN 后不塌成常数）           ：'
        f'{"PASS" if std_bn > 0.15 else "FAIL"}')
    say()
    say('  说明：本闸门只证"结构正确 + 未训练就不退化"，**不放行/否决训练**。')
    say('       最终是否有效，仍是训练后的 test mAP 与噪声地板之比。')
    say('=' * 96)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {args.out}')


if __name__ == '__main__':
    main()
