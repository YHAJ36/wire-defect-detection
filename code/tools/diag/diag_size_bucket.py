# -*- coding: utf-8 -*-
"""按 GT 框的**像素尺寸**分桶算 AP50，把「尺度」和「场景」两个因素分开。

为什么需要（A.9.44 的教训）
    A.9.44 按图片分辨率分桶（632x480 / 1920x1080 / 400x296 / 竖），结果每个桶
    恰好对应一个采集场景（水印录屏 / 白墙单绳 / 多绳设备背景），尺度和场景完全
    混杂，无法判断短板来自哪一边。

本工具改成**框级**分桶：同一个桶里的框跨越所有场景、所有分辨率，
因此桶间差异只能归因于「该目标在原图里有多少像素」，与场景无关。

分桶规则（按 GT 框在原图上的短边像素）
    t16   : < 16 px    极小，896 下会被缩到 <7 px
    t32   : 16-24 px
    t48   : 24-40 px
    t64   : 40-64 px
    t96   : 64-96 px
    big   : >= 96 px
    （分得比 COCO 细，因为本项目归一化面积中位 0.00567、896 下边长仅 ~67 px，
      整体偏小，COCO 的 small/medium/large 三档太粗）

AP 口径
    每桶独立算：GT 只取该桶的框；预测里凡是没匹配到该桶 GT 的一律计 FP
    （含未匹配的、以及匹配到别桶 GT 的）—— 与 COCO 按尺寸算 AP 的做法一致。
    101 点插值，IoU 0.5，NMS 0.5，conf 下限 0.001。

用法
    D:/yolo-env/Scripts/python.exe code/tools/diag/diag_size_bucket.py \
        --ckpt runs/nf_v11_896_s42/weights/last.pt --imgsz 896 --tag nf_s42_last
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from ultralytics import YOLO  # noqa: E402

ROOT = 'D:/2/dataset_final_bbox/_valtest'   # val+test 357 图
NAMES = ['break', 'thunderbolt', 'wear']
BUCKETS = [(0, 16, 't16'), (16, 24, 't32'), (24, 40, 't48'),
           (40, 64, 't64'), (64, 96, 't96'), (96, 1e9, 'big')]


def bucket_of(px):
    for lo, hi, nm in BUCKETS:
        if lo <= px < hi:
            return nm
    return 'big'


def iou_mat(a, b):
    """a:(N,4) b:(M,4) xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match_greedy(pred_boxes, pred_conf, gt_boxes, thr=0.5):
    """YOLO 口径：按 conf 降序，每个预测吃一个 IoU 最高的未占用 GT。
    返回每个预测匹配到的 GT 索引（-1 表示未匹配）。"""
    n = len(pred_boxes)
    out = np.full(n, -1, dtype=int)
    if n == 0 or len(gt_boxes) == 0:
        return out
    iou = iou_mat(pred_boxes, gt_boxes)
    order = np.argsort(-pred_conf)
    used = np.zeros(len(gt_boxes), dtype=bool)
    for i in order:
        cand = np.where(used, -1.0, iou[i])
        j = int(np.argmax(cand))
        if cand[j] >= thr:
            out[i] = j
            used[j] = True
    return out


def compute_ap(tp, conf, n_gt):
    """101 点插值 AP。tp: bool 数组（每个预测），conf 同长。"""
    if n_gt == 0:
        return float('nan')
    order = np.argsort(-conf)
    tp = tp[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(~tp)
    rec = ctp / n_gt
    prec = ctp / (ctp + cfp + 1e-9)
    # 单调化
    for i in range(len(prec) - 1, 0, -1):
        prec[i - 1] = max(prec[i - 1], prec[i])
    xs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, xs, side='left')
    ap = float(np.mean(np.where(idx < len(prec), prec[np.clip(idx, 0, len(prec) - 1)], 0.0)))
    return ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--imgsz', type=int, default=896)
    ap.add_argument('--iou', type=float, default=0.5, help='NMS IoU')
    ap.add_argument('--match-iou', type=float, default=0.5)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--device', default='0')
    args = ap.parse_args()

    ck = Path(args.ckpt)
    if not ck.is_absolute():
        ck = Path('D:/2/exp_upscale/runs') / args.ckpt
    model = YOLO(str(ck))

    imgs = sorted(glob.glob(ROOT + '/images/*.jpg'))
    print(f'images: {len(imgs)}')

    # 逐图：预测 -> 与 GT 匹配 -> 把每个预测记为 (桶, 类, conf, tp)
    recs = []                 # (bucket, cls, conf, is_tp)
    gt_count = defaultdict(int)   # (bucket, cls) -> n
    miss = []                 # 未匹配 GT: (bucket, cls, 图名)
    for s in range(0, len(imgs), args.batch):
        chunk = imgs[s:s + args.batch]
        res = model.predict(chunk, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            max_det=300, device=args.device, verbose=False)
        for p, r in zip(chunk, res):
            stem = os.path.splitext(os.path.basename(p))[0]
            lab = f'{ROOT}/labels/{stem}.txt'
            if not os.path.exists(lab):
                continue
            from PIL import Image
            with Image.open(p) as im:
                W, H = im.size
            g = [l.split() for l in open(lab) if l.strip()]
            gt = np.array([[float(x) for x in t[1:5]] for t in g], dtype=float).reshape(-1, 4)
            gcls = np.array([int(t[0]) for t in g], dtype=int)
            if len(gt):
                # 归一化 xywh -> 原图 xyxy（像素）
                cx, cy, w, h = gt[:, 0] * W, gt[:, 1] * H, gt[:, 2] * W, gt[:, 3] * H
                gxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
                gpx = np.minimum(w, h)   # 短边像素
                gb = [bucket_of(v) for v in gpx]
            else:
                gxy = np.zeros((0, 4)); gcls = np.zeros(0, dtype=int); gpx = np.zeros(0); gb = []
            for c, b in zip(gcls, gb):
                gt_count[(b, int(c))] += 1

            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                for c, b in zip(gcls, gb):
                    miss.append((b, int(c), stem))
                continue
            pb = boxes.xyxy.cpu().numpy()
            pc = boxes.conf.cpu().numpy()
            pcls = boxes.cls.cpu().numpy().astype(int)

            # 同类才允许匹配（YOLO 的 match_predictions 也是同类匹配）
            m = np.full(len(pb), -1, dtype=int)
            for c in np.unique(pcls):
                pi = np.where(pcls == c)[0]
                gi = np.where(gcls == c)[0]
                mm = match_greedy(pb[pi], pc[pi], gxy[gi], args.match_iou)
                for k, v in enumerate(mm):
                    if v >= 0:
                        m[pi[k]] = gi[v]
            for i in range(len(pb)):
                if m[i] >= 0:
                    recs.append((gb[m[i]], int(pcls[i]), float(pc[i]), True))
                else:
                    # 未匹配 -> 对所有桶都是 FP，按它自己的类别记
                    recs.append(('__fp__', int(pcls[i]), float(pc[i]), False))
            matched = set(m[m >= 0].tolist())
            for gi_, (c, b) in enumerate(zip(gcls, gb)):
                if gi_ not in matched:
                    miss.append((b, int(c), stem))

    # 每桶 × 每类算 AP50
    print('\n=== 按 GT 短边像素分桶（跨全部场景）===')
    header = f'{"桶":<6}{"GT":>6}{"TP":>6}{"AP50":>8}{"AP50-95略":>10}  逐类 AP50'
    print(header)
    out = {'ckpt': str(ck), 'tag': args.tag, 'buckets': {}, 'per_class': {}}
    for lo, hi, bnm in BUCKETS:
        ngt = sum(v for (b, c), v in gt_count.items() if b == bnm)
        if ngt == 0:
            continue
        # 该桶：TP = 匹配到本桶的预测；FP = 全部未匹配预测
        sel = [(c, cf, tp) for (b, c, cf, tp) in recs if b == bnm or b == '__fp__']
        per = {}
        for c in range(3):
            n_g = gt_count.get((bnm, c), 0)
            if n_g == 0:
                continue
            sub = [(cf, tp) for (cc, cf, tp) in sel if cc == c]
            if not sub:
                per[NAMES[c]] = 0.0
                continue
            conf = np.array([x[0] for x in sub])
            tp = np.array([x[1] for x in sub], dtype=bool)
            per[NAMES[c]] = compute_ap(tp, conf, n_g)
        vals = [v for v in per.values() if not np.isnan(v)]
        ap50 = float(np.mean(vals)) if vals else float('nan')
        ntp = sum(1 for (b, c, cf, tp) in recs if b == bnm and tp)
        print(f'{bnm:<6}{ngt:>6}{ntp:>6}{ap50:>8.4f}{"":>10}  ' +
              '  '.join(f'{k}={v:.3f}' for k, v in per.items()))
        out['buckets'][bnm] = dict(n_gt=int(ngt), n_tp=int(ntp), ap50=ap50,
                                   range=[lo, None if hi > 1e8 else hi])
        out['per_class'][bnm] = per

    # 漏检率
    print('\n=== 漏检（conf=0.001 下仍无匹配）===')
    for lo, hi, bnm in BUCKETS:
        ngt = sum(v for (b, c), v in gt_count.items() if b == bnm)
        nm = sum(1 for (b, c, s) in miss if b == bnm)
        if ngt:
            print(f'  {bnm:<6} GT={ngt:<5} 漏={nm:<4} ({nm / ngt * 100:.1f}%)')
            out['buckets'][bnm]['miss'] = int(nm)
            out['buckets'][bnm]['miss_rate'] = nm / ngt

    p = Path(f'D:/2/report/_p2_diag/diag_size_bucket_{args.tag}.json')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n→ {p}')


if __name__ == '__main__':
    main()
