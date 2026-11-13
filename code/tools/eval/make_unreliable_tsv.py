# -*- coding: utf-8 -*-
"""生成「三颗种子都不认同」的训练集 GT 框清单，供 train_perturb.py --drop-file 使用。

判据
    对 train 集每个 GT 框，取三颗独立种子（s42/s43/s44）各自与它的最佳同类预测 IoU。
    三颗**全部** < --thr（默认 0.30）才判为「不可信」。

    「全部」这个限定是必须的：单颗模型 recall 只有 0.68，用一颗会把它自己漏检的
    正常框也标进来，等于拿模型的短板去「清洗」标注，最后变成靠削弱任务涨分。

必须一起看的诊断（脚本会打印，不要只看清单条数）
    1. 删除比例。合理区间大约 3%~12%。超过 20% 说明阈值太松或模型太弱，
       这条路不能走 —— 删太多等于改了标签分布，后续 mAP 不可比。
    2. 类别分布。若被删框压倒性地集中在 break，那更可能是「break 难检」而不是
       「break 标错」——因为三颗模型对 break 的 A（互相 IoU）本来就只有 0.695。
    3. 面积分布。若被删框显著更小，同上：小目标漏检，不是标错。
    4. **人工抽检**：脚本会导出前 20 个框的原图坐标，务必自己看图确认。
       只看统计就删标注，是最容易自欺的一步。

输出
    TSV: stem<TAB>gt_index<TAB>cls<TAB>iou_max<TAB>area，前两列给 --drop-file 用。

用法
    D:/yolo-env/Scripts/python.exe code/tools/eval/make_unreliable_tsv.py
    ... --thr 0.3 --split train --out exp_upscale/logs/unreliable_train.tsv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path('D:/2')
DS = ROOT / 'dataset_noleak'
RUNS = ROOT / 'exp_upscale/runs'
SEEDS = (42, 43, 44)
CLASS_NAMES = {0: 'break', 1: 'thunder', 2: 'wear'}


def read_gt(stem, split):
    p = DS / 'labels' / split / f'{stem}.txt'
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--thr', type=float, default=0.30)
    ap.add_argument('--imgsz', type=int, default=896)
    ap.add_argument('--out', default=str(ROOT / 'exp_upscale/logs/unreliable_train.tsv'))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    img_dir = DS / 'images' / args.split
    stems = sorted(p.stem for p in img_dir.iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    print(f'{args.split} 图 {len(stems)}')

    # 每颗种子对每个 (stem, gt_idx) 的最佳 IoU
    best = {}
    for s in SEEDS:
        w = RUNS / f'nf_v11_noleak_s{s}/weights/best.pt'
        if not w.exists():
            raise SystemExit(f'缺权重 {w}')
        print(f'  推理 seed={s} ...', flush=True)
        m = YOLO(str(w))
        rs = m.predict(source=str(img_dir), imgsz=args.imgsz, conf=0.001,
                       iou=0.7, verbose=False, save=False, stream=True,
                       batch=8, device=0)
        got = {}
        for r in rs:
            stem = Path(r.path).stem
            gcls, gbox = read_gt(stem, args.split)
            if len(gbox) == 0:
                continue
            b = r.boxes
            if b is None or len(b) == 0:
                got[stem] = np.zeros(len(gbox), np.float32)
                continue
            pb = b.xyxy.cpu().numpy()
            pc = b.cls.cpu().numpy().astype(int)
            # 预测是像素坐标 -> 归一化以与 GT 可比
            h, w_ = r.orig_shape
            pb = pb / np.array([w_, h, w_, h], np.float32)
            M = iou_matrix(pb, gbox)
            v = np.zeros(len(gbox), np.float32)
            for gi in range(len(gbox)):
                cand = np.where(pc == gcls[gi])[0]
                if len(cand):
                    v[gi] = M[cand, gi].max()
            got[stem] = v
        best[s] = got

    # 三颗都 < thr 才算不可信
    rows = []
    n_gt = 0
    for stem in stems:
        gcls, gbox = read_gt(stem, args.split)
        if len(gbox) == 0:
            continue
        vals = np.stack([best[s].get(stem, np.zeros(len(gbox), np.float32))
                         for s in SEEDS])          # (3, n_gt)
        vmax = vals.max(0)
        n_gt += len(gbox)
        for gi in range(len(gbox)):
            if vmax[gi] < args.thr:
                w = gbox[gi, 2] - gbox[gi, 0]
                h = gbox[gi, 3] - gbox[gi, 1]
                rows.append((stem, gi, int(gcls[gi]), float(vmax[gi]),
                             float(w * h), float(vals[:, gi].mean())))

    print('\n' + '=' * 68)
    print(f'判据：三种子最佳 IoU 全部 < {args.thr}')
    print('=' * 68)
    print(f'  训练 GT 框 {n_gt}   判为不可信 {len(rows)} = '
          f'{len(rows)/max(n_gt,1)*100:.2f}%')
    if not rows:
        print('  没有命中（阈值过严），清单为空')
    else:
        r = np.array([[x[2], x[3], x[4]] for x in rows], float)
        all_area = []
        for stem in stems:
            gcls, gbox = read_gt(stem, args.split)
            if len(gbox):
                all_area += list((gbox[:, 2] - gbox[:, 0]) * (gbox[:, 3] - gbox[:, 1]))
        print(f'  三种子 IoU 均值：中位 {np.median([x[5] for x in rows]):.3f}')
        print(f'  被删框类别分布：')
        for c in (0, 1, 2):
            n_c = int((r[:, 0] == c).sum())
            tot_c = sum(1 for stem in stems for v in read_gt(stem, args.split)[0] if v == c)
            print(f'    {CLASS_NAMES[c]:<8} {n_c:<4} / {tot_c:<4} = '
                  f'{n_c/max(tot_c,1)*100:5.1f}% 的该类框')
        print(f'  面积：被删中位 {np.median(r[:, 2]):.5f}  '
              f'全体中位 {np.median(all_area):.5f}')
        # 合理性检查
        ratio = len(rows) / n_gt
        if ratio > 0.20:
            print('\n  ⚠ 删除比例 > 20%，阈值过松或模型太弱，这条路不可用：')
            print('    删掉两成标注等于改了标签分布，之后 mAP 与基线不可比。')
        br = (r[:, 0] == 0).mean()
        if br > 0.7:
            print('\n  ⚠ 被删框 70% 以上是 break。break 的模型间一致性本来就只有 0.695，')
            print('    这更可能是「难检」而非「标错」。建议不要按这个清单删。')

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['stem', 'gt_index', 'cls', 'iou_max', 'area', 'iou_mean3'])
        for x in rows:
            w.writerow([x[0], x[1], x[2], f'{x[3]:.4f}', f'{x[4]:.6f}', f'{x[5]:.4f}'])
    print(f'\n已写 {outp}（{len(rows)} 行）')
    print('  ⚠ 下一步：随机抽 10~20 行，打开原图核实。只看统计就删标注是最容易自欺的一步。')


if __name__ == '__main__':
    main()
