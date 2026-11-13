# -*- coding: utf-8 -*-
"""用「三种子共识」直接测量同帧标注噪声，并据此给出 mAP50 天花板。

为什么不再用抖动模拟去拟合 0.477
    上一版（ceiling_analysis.py）用「相邻帧同目标 IoU 中位 0.477」标定抖动强度，
    推出 sigma*=0.2525 -> mAP50 天花板 0.5881，**低于实测 0.7467**。
    完美模型不可能输给真实模型，所以这个标定链条错了。错在：
        0.477 是**跨帧**量，包含目标真实移动、视角变化、遮挡；
        它不是「同一帧的两次标注」的差异。
    换句话说：0.477 里有多少是标注噪声、多少是真实变化，用跨帧数据分不开。

本脚本换一个不依赖跨帧的做法
    三个独立训练的种子（s42/s43/s44，只差初始化与数据顺序）对**同一帧同一目标**
    的预测，其共识位置是「目标真实位置」的一个近似无偏估计。
    那么：
        三种子两两 IoU 的中位  = 模型自身的定位不确定性      记为 A
        共识位置与 GT 的 IoU   = 标注相对真实位置的偏离      记为 B
    A 与 B 的大小关系直接回答「瓶颈在模型还是在标注」：
        A >> B  模型很一致，但都离 GT 远  -> 标注是漂移的那一方（可归因）
        A ~= B  模型自己就不确定          -> 模型能力问题（不可归因于标注）
    并用 B 的分布标定抖动强度，算出 mAP50 天花板。

        注意 B 会**高估**标注噪声（模型误差不是严格零均值独立），
        所以据此算出的天花板是**下界**，偏保守 —— 方向是我们想要的。

另外给出上界交叉检验
    若把 0.477 当作纯标注噪声（上一版做法），天花板只有 58.8% < 74.67%（实测）。
    这个矛盾说明：**0.477 里能算作标注噪声的部分必然远小于 0.477**。
    这一条本身就是对「标注太烂所以加模块没用」这个说法的一个定量反驳。

用法
    D:/yolo-env/Scripts/python.exe code/tools/eval/iou_diagnosis.py
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
from ultralytics.utils.metrics import ap_per_class

ROOT = Path('D:/2')
DS = ROOT / 'dataset_noleak'
CACHE = ROOT / 'exp_upscale/logs/wbf_cache_noleak_orig'
OUT = ROOT / 'exp_upscale/logs/iou_diagnosis.tsv'

SEEDS = ('s42', 's43', 's44')
TMPL = 'nf_v11_noleak_{s}.npz'
CLASS_NAMES = {0: 'break', 1: 'thunder', 2: 'wear'}
MIN_IOU = 0.3       # 预测与 GT 的最低关联门槛（低于此视为未检出，不参与共识）


def read_gt(stem):
    p = DS / 'labels/test' / f'{stem}.txt'
    if not p.exists():
        return np.zeros((0,), np.int64), np.zeros((0, 4), np.float32)
    rows = [[float(x) for x in ln.split()[:5]]
            for ln in p.read_text().splitlines() if len(ln.split()) >= 5]
    if not rows:
        return np.zeros((0,), np.int64), np.zeros((0, 4), np.float32)
    a = np.asarray(rows, np.float32)
    cx, cy, w, h = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    return (a[:, 0].astype(np.int64),
            np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1))


def load_pred(path):
    z = np.load(path, allow_pickle=True)
    return {str(k): z[k] for k in z['__keys__']}


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    u = aa[:, None] + ab[None, :] - inter
    o = np.zeros_like(inter)
    m = u > 1e-12
    o[m] = inter[m] / u[m]
    return o


def iou_one(a, b):
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[0] * wh[1]
    aa = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    ab = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    u = aa + ab - inter
    return float(inter / u) if u > 1e-12 else 0.0


def pack(v):
    if len(v) == 0:
        return None
    v = np.asarray(v, float)
    return dict(n=len(v), p10=float(np.percentile(v, 10)),
                median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                hit50=float((v >= 0.5).mean()), mean=float(v.mean()))


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    preds = {}
    for s in SEEDS:
        p = CACHE / TMPL.format(s=s)
        if not p.exists():
            raise SystemExit(f'缺缓存 {p}')
        preds[s] = load_pred(p)
    stems = sorted(os.path.splitext(x)[0] for x in os.listdir(DS / 'images/test'))
    print(f'test 图 {len(stems)}   种子 {SEEDS}')

    # 每个 GT 记录：A=三种子两两 IoU 均值，B=共识框与 GT 的 IoU
    recs = []      # dict(cls, A, B, nseed, area)
    per_seed_gt_iou = {s: [] for s in SEEDS}
    for st in stems:
        gcls, gbox = read_gt(st)
        if len(gbox) == 0:
            continue
        picks = {}      # s -> {gt_idx: pred_box}
        for s in SEEDS:
            pr = preds[s].get(st)
            if pr is None or len(pr) == 0:
                picks[s] = {}
                continue
            M = iou_matrix(pr[:, :4], gbox)
            got = {}
            for gi in range(len(gbox)):
                cand = np.where((pr[:, 5].astype(int) == gcls[gi]) & (M[:, gi] >= MIN_IOU))[0]
                if len(cand) == 0:
                    continue
                j = cand[np.argmax(M[cand, gi])]
                got[gi] = pr[j, :4]
                per_seed_gt_iou[s].append(float(M[j, gi]))
            picks[s] = got
        for gi in range(len(gbox)):
            if not all(gi in picks[s] for s in SEEDS):
                continue
            boxes = [picks[s][gi] for s in SEEDS]
            A = float(np.mean([iou_one(boxes[i], boxes[j])
                               for i in range(3) for j in range(i + 1, 3)]))
            cons = np.mean(boxes, 0)             # 共识框 = 三种子均值
            B = iou_one(cons, gbox[gi])
            w = gbox[gi, 2] - gbox[gi, 0]
            h = gbox[gi, 3] - gbox[gi, 1]
            recs.append(dict(cls=int(gcls[gi]), A=A, B=B, area=float(w * h)))
    print(f'三种子都检出并能关联的 GT：{len(recs)} / '
          f'{sum(len(read_gt(s)[0]) for s in stems)}')

    # ---- 主表：A vs B
    print('\n' + '=' * 72)
    print('模型共识一致性 A  vs  共识-标注偏离 B（同一帧，同一目标）')
    print('=' * 72)
    rows = []
    for key in ['ALL', 'break', 'thunder', 'wear']:
        sub = recs if key == 'ALL' else [r for r in recs if CLASS_NAMES[r['cls']] == key]
        if not sub:
            continue
        A = pack([r['A'] for r in sub])
        B = pack([r['B'] for r in sub])
        gap = A['median'] - B['median']
        print(f'  {key:<8} n={A["n"]:<4}  '
              f'A(模型间) 中位={A["median"]:.3f}  '
              f'B(共识-标注) 中位={B["median"]:.3f}  '
              f'差 A−B={gap:+.3f}')
        print(f'  {"":8}      A: p10={A["p10"]:.3f} p90={A["p90"]:.3f} '
              f'≥0.5={A["hit50"]*100:.0f}%   |   '
              f'B: p10={B["p10"]:.3f} p90={B["p90"]:.3f} ≥0.5={B["hit50"]*100:.0f}%')
        rows.append([key, A['n'], A['median'], A['p10'], A['p90'], A['hit50'],
                     B['median'], B['p10'], B['p90'], B['hit50'], gap])

    # ---- 按目标面积分层（大框 vs 小框，标注噪声的相对影响不同）
    print('\n  按 GT 面积分层（归一化面积，中位切分）：')
    areas = np.array([r['area'] for r in recs])
    cut = float(np.median(areas))
    for name, sel in [('小框', [r for r in recs if r['area'] <= cut]),
                      ('大框', [r for r in recs if r['area'] > cut])]:
        if not sel:
            continue
        A = pack([r['A'] for r in sel])
        B = pack([r['B'] for r in sel])
        print(f'    {name} (n={A["n"]:<3}) A 中位={A["median"]:.3f}  '
              f'B 中位={B["median"]:.3f}  差={A["median"]-B["median"]:+.3f}')
        rows.append([f'area_{name}', A['n'], A['median'], A['p10'], A['p90'],
                     A['hit50'], B['median'], B['p10'], B['p90'], B['hit50'],
                     A['median'] - B['median']])

    # ---- 天花板：完美模型 = 共识位置
    print('\n' + '=' * 72)
    print('共识位置与标注的偏离（这是 B 的分布，不是 mAP 天花板）')
    print('=' * 72)
    B_all = pack([r['B'] for r in recs])
    print(f'  若「完美模型」输出共识位置，它与 GT 的 IoU 中位 = {B_all["median"]:.3f}')
    print(f'  → 其中 IoU≥0.5 的比例 = {B_all["hit50"]*100:.1f}%（这是可被 mAP50 记入的上限份额）')
    # ---- 覆盖率：三种子都检出的 GT 占比（这是 recall 的结构性上限份额）
    n_gt = sum(len(read_gt(s)[0]) for s in stems)
    n_cov = len(recs)
    print('\n' + '=' * 72)
    print('覆盖率（三种子同时检出并可关联的 GT 占比）')
    print('=' * 72)
    print(f'  {n_cov}/{n_gt} = {n_cov/n_gt*100:.1f}%   '
          f'→ 未覆盖的 {n_gt-n_cov} 个目标必然是漏检来源')

    # ---- 低共识目标画像：A<0.5 的那批，模型自己就不一致
    low = [r for r in recs if r['A'] < 0.5]
    print(f'\n  低共识目标（A<0.5）：{len(low)}/{len(recs)} = {len(low)/len(recs)*100:.1f}%')
    if low:
        for k in ('break', 'thunder', 'wear'):
            n_k = sum(1 for r in recs if CLASS_NAMES[r['cls']] == k)
            n_low = sum(1 for r in low if CLASS_NAMES[r['cls']] == k)
            if n_k:
                print(f'    {k:<8} {n_low}/{n_k} = {n_low/n_k*100:.1f}%')
        ar = np.array([r['area'] for r in recs])
        al = np.array([r['area'] for r in low])
        print(f'    面积：全体中位 {np.median(ar):.5f}  vs  低共识组中位 '
              f'{np.median(al):.5f}（{"更小" if np.median(al) < np.median(ar) else "更大"}）')

    # ---- 说明：这里**不给** mAP 天花板
    print('\n' + '=' * 72)
    print('关于「mAP50 天花板」——本脚本刻意不给这个数，原因如下')
    print('=' * 72)
    print('  1. 用「三种子共识框」当预测时，只覆盖已检出的 GT，漏检被排除在外，')
    print('     recall 虚高，算出的 mAP 不是天花板而是有选择偏差的中间量。')
    print('  2. 用「抖动版 GT」当预测时，等于假设模型能 100% 检出，只受抖动影响，')
    print('     这会把上限抬到 0.93 量级，同样不可信。')
    print('  3. 因此本轮的可靠产出是 A/B 对比，而不是一个 mAP 天花板数字。')
    print('     A 只依赖模型自身预测，不受标注正确性影响 —— 这是它能说硬话的原因。')

    # ---- 落盘
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['group', 'n', 'A_median', 'A_p10', 'A_p90', 'A_hit50',
                    'B_median', 'B_p10', 'B_p90', 'B_hit50', 'A_minus_B'])
        for r in rows:
            w.writerow(r)
        w.writerow(['coverage', n_cov, '', '', '', '', '', '', '', '', ''])
        w.writerow(['n_gt', n_gt, '', '', '', '', '', '', '', '', ''])
        w.writerow(['low_consensus_A_lt_0.5', len(low), '', '', '', '', '', '', '', '', ''])
    print(f'\n已写 {OUT}')


if __name__ == '__main__':
    main()
