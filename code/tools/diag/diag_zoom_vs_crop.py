# -*- coding: utf-8 -*-
"""缩放 vs 裁剪对照：把「降采样抹细节」和「切片切边界」两个因素分开。

问题来源
    A.9.34 之后发现 test 里大部分目标在 1920x1080 的图上，letterbox 到 896 被缩小 2.14 倍。
    按缩放比分组，break 召回 缩小组 76.4% vs 放大组 90.0%。但这不能直接归因于"降采样"——
    两组的目标尺寸、场景、类别分布都不一样，是混杂的。

设计
    只取长边 > 1.5*896 的图，做两件事：
      A. full   : 整图 letterbox 到 896（缩放 0.467，细节被抹）
      B. crop896: 从原图中心裁 896x896 窗口（**像素 1:1，无任何缩放**）
    两者**只比较完整落在中心窗口内的 GT**，保证比的是同一批目标，唯一差异是"有没有被降采样"。
    代价：crop896 视野只有整图的约 19%，所以 GT 数少，但这是干净的对照。

    若 B 的召回明显高于 A ⇒ 降采样确实损害判别力。
    但这**不等于**切片是解法：切片还会引入"目标被切在边界"的新漏检（已在 6 张密集图上
    观察到 2x2 反而更差），那是另一笔代价，本实验不涉及。

口径
    conf 0.25 / NMS IoU 0.5 / 命中 IoU 0.5 / 置信度降序一配一。
    只计入与中心窗口交集 / 自身面积 > 0.9 的 GT（避免半个框）。

用法
    python code/tools/diag/diag_zoom_vs_crop.py [--run nf_v11_896_s42]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, 'D:/2/code/yolov11')
sys.path.insert(0, 'D:/2/code/tools/diag')
from ultralytics import YOLO                     # noqa: E402
from diag_tiled_infer import iou_matrix, load_items  # noqa: E402

RUNS = Path('D:/2/exp_upscale/runs')
IMGSZ = 896
CONF = 0.25
NMS_IOU = 0.5
MATCH_IOU = 0.5
MAX_DET = 300
NAMES = ['break', 'thunderbolt', 'wear']


def hit_mask(boxes, classes, scores, gts, idx):
    """按置信度降序一配一，返回 len(idx) 的 bool 数组。"""
    gd = np.zeros(len(idx), bool)
    if len(boxes) == 0 or not idx:
        return gd
    gb = np.array([gts[i]['box'] for i in idx], np.float32)
    gc = np.array([gts[i]['cls'] for i in idx], int)
    M = iou_matrix(boxes, gb)
    for j in np.argsort(-scores):
        cand = np.where((M[j] > MATCH_IOU) & (~gd) & (gc == classes[j]))[0]
        if not len(cand):
            cand = np.where((M[j] > MATCH_IOU) & (~gd))[0]
        if len(cand):
            gd[cand[np.argmax(M[j, cand])]] = True
    return gd


def unpack(r):
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, int), np.zeros(0, np.float32)
    return (r.boxes.xyxy.cpu().numpy().astype(np.float32),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy().astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='nf_v11_896_s42')
    args = ap.parse_args()

    items = load_items('test')
    big = [(p, g, w, h) for (p, g, w, h) in items if max(w, h) > 1.5 * IMGSZ]
    print(f'run={args.run}  test 总 {len(items)} 张，长边 > {1.5 * IMGSZ:.0f}px 的图 {len(big)} 张')
    print(f'口径 conf={CONF} NMS={NMS_IOU} 命中IoU={MATCH_IOU}，只评估完整落在中心 '
          f'{IMGSZ}x{IMGSZ} 窗口内的 GT\n')

    model = YOLO(str(RUNS / args.run / 'weights' / 'best.pt'))
    stat = defaultdict(lambda: [0, 0])
    per_img = []
    n_gt_center = 0

    for im_path, gts, w0, h0 in big:
        im = cv2.imread(str(im_path))
        x0 = max(0, (w0 - IMGSZ) // 2)
        y0 = max(0, (h0 - IMGSZ) // 2)
        x1, y1 = min(w0, x0 + IMGSZ), min(h0, y0 + IMGSZ)
        win = np.array([[x0, y0, x1, y1]], np.float32)

        idx = []
        for i, g in enumerate(gts):
            b = np.array(g['box'], np.float32)[None]
            area = (b[0, 2] - b[0, 0]) * (b[0, 3] - b[0, 1])
            if area <= 0:
                continue
            lt = np.maximum(b[:, :2], win[:, :2])
            rb = np.minimum(b[:, 2:], win[:, 2:])
            wh = np.clip(rb - lt, 0, None)
            if float(wh[0, 0] * wh[0, 1]) / max(area, 1e-6) > 0.9:
                idx.append(i)
        if not idx:
            continue
        n_gt_center += len(idx)

        bA, cA, sA = unpack(model.predict(im, imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                                          max_det=MAX_DET, verbose=False, stream=False)[0])
        bB, cB, sB = unpack(model.predict(im[y0:y1, x0:x1], imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                                          max_det=MAX_DET, verbose=False, stream=False)[0])
        if len(bB):
            bB = bB + np.array([x0, y0, x0, y0], np.float32)

        mA = hit_mask(bA, cA, sA, gts, idx)
        mB = hit_mask(bB, cB, sB, gts, idx)

        per_img.append((im_path.name, len(idx), int(mA.sum()), int(mB.sum())))
        for mode, mask in (('full', mA), ('crop896', mB)):
            for k, i_gt in enumerate(idx):
                key = (mode, NAMES[gts[i_gt]['cls']])
                stat[key][0] += 1
                stat[key][1] += int(mask[k])

    tot = defaultdict(lambda: [0, 0])
    for (mode, _), (n, h) in stat.items():
        tot[mode][0] += n
        tot[mode][1] += h

    print(f'参与评估的中心区 GT 共 {n_gt_center} 个（来自 {len(per_img)} 张图）\n')
    print(f'{"模式":<10}{"GT":>5}{"命中":>5}{"召回":>9}   break / thunder / wear')
    for mode in ('full', 'crop896'):
        n, h = tot[mode]
        det = '  '.join(f'{stat[(mode, c)][1]}/{stat[(mode, c)][0]}' for c in NAMES)
        print(f'{mode:<10}{n:>5}{h:>5}{h / max(n, 1) * 100:8.2f}%   {det}')

    print('\n逐图差异最大的（GT数 / full / crop896）:')
    for name, n, a, b in sorted(per_img, key=lambda x: -(x[3] - x[2]))[:8]:
        print(f'  {name:<26}{n:>3}{a:>4}{b:>4}  ← crop 更好')
    for name, n, a, b in sorted(per_img, key=lambda x: -(x[2] - x[3]))[:4]:
        if a > b:
            print(f'  {name:<26}{n:>3}{a:>4}{b:>4}  ← crop 更差')


if __name__ == '__main__':
    main()
