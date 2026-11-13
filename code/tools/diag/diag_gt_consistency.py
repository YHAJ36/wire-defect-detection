# -*- coding: utf-8 -*-
"""审计 GT 标注自身的帧间一致性 —— 判断 break 的低 AP 是模型问题还是标注问题。

动机（A.9.47e 的分桶发现）
    按 GT 短边像素分桶后，>=96 px 那档 break AP50 只有 0.354，而同一桶里
    thunderbolt 是 0.966，且该桶漏检率仅 1.0%。大框在 IoU 0.5 下本应更容易达标，
    却差了 2.7 倍 ⇒ 强烈怀疑大缺陷的标注边界本身不一致（整个散开区 vs 只框断点）。

做法
    全库文件名统一为 firc_xianlan_<编号>。取编号相差 <= gap 的图对（默认 2），
    在其中做**同类** GT 框匹配（IoU >= thr），对匹配上的框对统计：
      - 相对宽/高差  |w1-w2| / mean(w)
      - 中心位移      dist / sqrt(mean(area))
      - 框对 IoU
    这三个量的分布就是「标注噪声地板」：模型与 GT 的偏差不可能低于它。

对照基准（A.9.41，模型侧）
    r_w std 0.1465、r_h std 0.1427、d_cx std 0.0731、rho std ~0.145
    若 GT 自身帧间相对差的中位数也在这个量级，则模型的抖动已被标注噪声完全解释，
    AP 上限由标注决定，任何结构改进都打不到靶。

用法
    D:/yolo-env/Scripts/python.exe code/tools/diag/diag_gt_consistency.py --gap 2
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = 'D:/2/dataset_final_bbox'
NAMES = ['break', 'thunderbolt', 'wear']


def load_gt(split):
    """返回 {stem: (cls(N,), xyxy(N,4) 像素, (W,H))}"""
    out = {}
    for p in glob.glob(f'{ROOT}/images/{split}/*.jpg'):
        stem = os.path.splitext(os.path.basename(p))[0]
        lab = f'{ROOT}/labels/{split}/{stem}.txt'
        if not os.path.exists(lab):
            continue
        with Image.open(p) as im:
            W, H = im.size
        rows = [l.split() for l in open(lab) if l.strip()]
        if rows:
            a = np.array([[float(x) for x in t[1:5]] for t in rows])
            c = np.array([int(t[0]) for t in rows])
            cx, cy, w, h = a[:, 0] * W, a[:, 1] * H, a[:, 2] * W, a[:, 3] * H
            xy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        else:
            c = np.zeros(0, int)
            xy = np.zeros((0, 4))
        out[stem] = (c, xy, (W, H))
    return out


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gap', type=int, default=2, help='编号最大间隔')
    ap.add_argument('--thr', type=float, default=0.3, help='帧间匹配 IoU 下限')
    ap.add_argument('--min-side', type=float, default=0.0,
                    help='只统计短边 >= 该像素值的框（用于只看大缺陷）')
    args = ap.parse_args()

    data = {}
    for sp in ('train', 'val', 'test'):
        data.update(load_gt(sp))
    print(f'总图数 {len(data)}')

    idx = {}
    for stem in data:
        m = re.match(r'^firc_xianlan_(\d+)$', stem)
        if m:
            idx[int(m.group(1))] = stem
    keys = sorted(idx)
    print(f'可解析编号 {len(keys)}，范围 {keys[0]}–{keys[-1]}')

    stats = {i: [] for i in range(3)}   # cls -> [(dw, dh, dcenter, iou, w_mean, h_mean)]
    pairs = 0
    for k in keys:
        for g in range(1, args.gap + 1):
            if k + g not in idx:
                continue
            c1, b1, _ = data[idx[k]]
            c2, b2, _ = data[idx[k + g]]
            if len(b1) == 0 or len(b2) == 0:
                continue
            pairs += 1
            for c in range(3):
                i1 = np.where(c1 == c)[0]
                i2 = np.where(c2 == c)[0]
                if len(i1) == 0 or len(i2) == 0:
                    continue
                A, B = b1[i1], b2[i2]
                M = iou_mat(A, B)
                # 贪心一对一
                used = set()
                for i in range(len(A)):
                    j = int(np.argmax(np.where([x in used for x in range(len(B))], -1, M[i])))
                    if M[i, j] < args.thr:
                        continue
                    used.add(j)
                    wa, ha = A[i, 2] - A[i, 0], A[i, 3] - A[i, 1]
                    wb, hb = B[j, 2] - B[j, 0], B[j, 3] - B[j, 1]
                    wm, hm = (wa + wb) / 2, (ha + hb) / 2
                    if min(wa, ha) < args.min_side:
                        continue
                    dw = abs(wa - wb) / (wm + 1e-9)
                    dh = abs(ha - hb) / (hm + 1e-9)
                    ca = np.array([(A[i, 0] + A[i, 2]) / 2, (A[i, 1] + A[i, 3]) / 2])
                    cb = np.array([(B[j, 0] + B[j, 2]) / 2, (B[j, 1] + B[j, 3]) / 2])
                    dc = float(np.linalg.norm(ca - cb) / np.sqrt(wm * hm + 1e-9))
                    stats[c].append((dw, dh, dc, float(M[i, j]), wm, hm))

    print(f'参与比较的图对 {pairs}')
    print(f'\n=== GT 帧间一致性（同一目标跨 <= {args.gap} 帧，IoU>={args.thr} 才算同一个）===')
    print(f'{"类":<12}{"n":>6}{"相对宽差":>22}{"相对高差":>22}{"中心位移":>20}{"IoU":>16}')
    print(f'{"":12}{"":>6}{"中位":>7}{"p90":>8}{"sd":>7}{"中位":>7}{"p90":>8}{"sd":>7}{"中位":>7}{"p90":>7}{"中位":>8}{"p10":>8}')
    out = {}
    for c in range(3):
        s = np.array(stats[c]) if stats[c] else np.zeros((0, 6))
        if len(s) == 0:
            print(f'{NAMES[c]:<12}  0')
            continue
        dw, dh, dc = s[:, 0], s[:, 1], s[:, 2]
        iou = s[:, 3]
        print(f'{NAMES[c]:<12}{len(s):>6}'
              f'{np.median(dw):>7.3f}{np.percentile(dw, 90):>8.3f}{dw.std():>7.3f}'
              f'{np.median(dh):>7.3f}{np.percentile(dh, 90):>8.3f}{dh.std():>7.3f}'
              f'{np.median(dc):>7.3f}{np.percentile(dc, 90):>7.3f}'
              f'{np.median(iou):>8.3f}{np.percentile(iou, 10):>8.3f}')
        out[NAMES[c]] = dict(
            n=int(len(s)),
            dw=dict(med=float(np.median(dw)), p90=float(np.percentile(dw, 90)), sd=float(dw.std())),
            dh=dict(med=float(np.median(dh)), p90=float(np.percentile(dh, 90)), sd=float(dh.std())),
            dc=dict(med=float(np.median(dc)), p90=float(np.percentile(dc, 90))),
            iou=dict(med=float(np.median(iou)), p10=float(np.percentile(iou, 10))),
            mean_w=float(np.median(s[:, 4])), mean_h=float(np.median(s[:, 5])),
        )

    print('\n模型侧对照（A.9.41）: r_w sd 0.1465  r_h sd 0.1427  d_cx sd 0.0731')

    p = Path(f'D:/2/report/_p2_diag/gt_consistency_gap{args.gap}.json')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(gap=args.gap, thr=args.thr,
                                 min_side=args.min_side, res=out),
                            ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'→ {p}')


if __name__ == '__main__':
    main()
