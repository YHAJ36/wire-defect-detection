# -*- coding: utf-8 -*-
"""漏检性质诊断：模型是「没看见」，还是「框了但跟标注对不上」。

为什么要查
    A.9.34 得到 44 个漏检、且三颗种子在 32-64px 段给出完全相同的 115/140。
    高度一致说明漏的是同一批固定目标，不像随机波动。可能有两种完全不同的原因：
      (a) 模型确实没检出（判别力不足）
      (b) 模型框出来了，但和标注的框重合度不够（定位偏差 / 标注本身偏差）
    这两者的解法完全不同：(a) 要改训练/增强，(b) 要核对标注或改回归目标。
    本条诊断不改任何东西，只把漏检按「GT 附近最佳同类预测框的 IoU」分桶。

输出
    1. 漏检 GT 的 IoU 分桶分布（最近同类预测框）
    2. 召回随命中阈值的松弛曲线（IoU 0.5 / 0.45 / 0.4 / 0.35 / 0.3 / 0.2）
    3. 完全无邻近预测框（IoU < 0.05）的漏检清单，按图聚合

用法
    python code/tools/diag/diag_miss_iou.py [--run nf_v11_896_s42] [--conf 0.25]
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, 'D:/2/code/yolov11')
sys.path.insert(0, 'D:/2/code/tools/diag')
from ultralytics import YOLO                     # noqa: E402
from diag_tiled_infer import iou_matrix, load_items  # noqa: E402

RUNS = Path('D:/2/exp_upscale/runs')
IMGSZ, NMS_IOU, MAX_DET = 896, 0.5, 300
NAMES = ['break', 'thunderbolt', 'wear']


def unpack(r):
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, int), np.zeros(0, np.float32)
    return (r.boxes.xyxy.cpu().numpy().astype(np.float32),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy().astype(np.float32))


def match_at(boxes, classes, scores, gts, thr):
    """返回 bool mask（len(gts)）。"""
    gd = np.zeros(len(gts), bool)
    if len(boxes) == 0 or not gts:
        return gd
    gb = np.array([g['box'] for g in gts], np.float32)
    gc = np.array([g['cls'] for g in gts], int)
    M = iou_matrix(boxes, gb)
    for j in np.argsort(-scores):
        cand = np.where((M[j] > thr) & (~gd) & (gc == classes[j]))[0]
        if not len(cand):
            cand = np.where((M[j] > thr) & (~gd))[0]
        if len(cand):
            gd[cand[np.argmax(M[j, cand])]] = True
    return gd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='nf_v11_896_s42')
    ap.add_argument('--conf', type=float, default=0.25)
    args = ap.parse_args()

    items = load_items('test')
    model = YOLO(str(RUNS / args.run / 'weights' / 'best.pt'))
    imgs = [p for p, _, _, _ in items]
    preds = []
    for i in range(0, len(imgs), 4):
        rs = model.predict(imgs[i:i + 4], imgsz=IMGSZ, conf=args.conf, iou=NMS_IOU,
                           max_det=MAX_DET, agnostic_nms=False, verbose=False, stream=False)
        preds.extend(unpack(r) for r in rs)

    # 1) IoU 0.5 下的漏检，逐个找最近同类预测框
    buckets = Counter()
    no_box = defaultdict(int)
    miss_rows = []
    cls_stat = defaultdict(lambda: Counter())
    for (path, gts, w0, h0), (b, c, s) in zip(items, preds):
        if not gts:
            continue
        gd = match_at(b, c, s, gts, 0.5)
        if not len(b):
            for k, g in enumerate(gts):
                buckets['无任何预测框'] += 1
                cls_stat[NAMES[g['cls']]]['无任何预测框'] += 1
                miss_rows.append((path.name, NAMES[g['cls']], -1.0, g['box']))
            continue
        M = iou_matrix(b, np.array([g['box'] for g in gts], np.float32))
        for k, g in enumerate(gts):
            if gd[k]:
                continue
            same = np.where(c == g['cls'])[0]
            best = float(M[:, k].max()) if len(M) else -1.0
            best_same = float(M[same, k].max()) if len(same) else -1.0
            ref = best_same if len(same) else best
            if ref < 0:
                tag = '无任何预测框'
            elif ref < 0.05:
                tag = '<0.05 完全没框'
            elif ref < 0.2:
                tag = '0.05-0.2'
            elif ref < 0.3:
                tag = '0.2-0.3'
            elif ref < 0.4:
                tag = '0.3-0.4'
            elif ref < 0.5:
                tag = '0.4-0.5'
            else:
                tag = '>=0.5(应为命中)'
            buckets[tag] += 1
            cls_stat[NAMES[g['cls']]][tag] += 1
            miss_rows.append((path.name, NAMES[g['cls']], ref, g['box']))
            if ref < 0.05:
                no_box[path.name] += 1

    total_miss = sum(buckets.values())
    print(f'run={args.run}  conf={args.conf}  NMS={NMS_IOU}  test {len(items)} 张')
    print(f'漏检 GT 共 {total_miss} 个（IoU 0.5 命中口径）\n')
    print('漏检按「GT 附近最佳同类预测框 IoU」分桶：')
    order = ['无任何预测框', '<0.05 完全没框', '0.05-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '>=0.5(应为命中)']
    for k in order:
        if buckets[k]:
            print(f'  {k:<18} {buckets[k]:>3} 个  ({buckets[k] / total_miss * 100:5.1f}%)')

    print('\n按类别:')
    for cname in NAMES:
        cc = cls_stat[cname]
        if not sum(cc.values()):
            continue
        s = sum(cc.values())
        det = '  '.join(f'{k}:{cc[k]}' for k in order if cc[k])
        print(f'  {cname:<12} 漏 {s:>3} 个   {det}')

    # 2) 命中阈值松弛曲线
    print('\n召回随命中阈值的松弛（同一批预测框，只改判定 IoU）:')
    print(f'  {"阈值":<8}{"总召回":>10}' + ''.join(f'{c:>14}' for c in NAMES))
    for thr in (0.5, 0.45, 0.4, 0.35, 0.3, 0.2):
        tot, hit = 0, 0
        per = defaultdict(lambda: [0, 0])
        for (path, gts, w0, h0), (b, c, s) in zip(items, preds):
            if not gts:
                continue
            gd = match_at(b, c, s, gts, thr)
            tot += len(gts)
            hit += int(gd.sum())
            for k, g in enumerate(gts):
                per[NAMES[g['cls']]][0] += 1
                per[NAMES[g['cls']]][1] += int(gd[k])
        print(f'  {thr:<8}{hit:>4}/{tot} {hit / tot * 100:5.2f}%' +
              ''.join(f'{per[c][1]:>7}/{per[c][0]:<6}' for c in NAMES))

    # 3) 完全没框的漏检，按图聚合
    if no_box:
        print(f'\n完全无邻近框（IoU<0.05）的漏检共 {sum(no_box.values())} 个，'
              f'分布在 {len(no_box)} 张图：')
        for name, n in sorted(no_box.items(), key=lambda x: -x[1])[:10]:
            print(f'  {name:<28} {n:>2} 个')
        print('\n这些 GT 的边长（原图像素）:', end=' ')
        sides = []
        for name, cls, iou, box in miss_rows:
            if iou >= 0 and iou < 0.05:
                sides.append(((box[2] - box[0]) * (box[3] - box[1])) ** 0.5)
        if sides:
            a = np.array(sides)
            print(f'n={len(a)} 最小 {a.min():.1f} 中位 {np.median(a):.1f} 最大 {a.max():.1f}')


if __name__ == '__main__':
    main()
