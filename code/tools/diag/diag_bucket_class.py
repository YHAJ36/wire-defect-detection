# -*- coding: utf-8 -*-
"""单桶单类失败分解：break 在 1920x1080 桶为什么只有 AP50 0.803。

背景（A.9.44）
    break 整体 AP50 0.847 是三类最低（thunder 0.980 / wear 0.932）。
    按框数加权后定位到 1920x1080 桶：break 0.803、77 框 = 全部 break 框的 51%，
    而同桶 thunder 0.995、wear 0.925。
    => 同一个场景、同一批图、同一个模型，只有 break 掉 19 pt。
       这不是「断股天生难」，必须找出是哪一步掉的。

做法
    对该桶每一张图，conf 0.001 / NMS 0.5 取全部预测，逐 GT 做类内贪心匹配，
    把每个 GT 归到 5 类之一：
      hit       IoU>=0.5 且类别对               （已检到）
      loc_fail  IoU 0.1-0.5，有同类框但不够准    （框歪了）
      cross     IoU>=0.5 但类别错                （认错类）
      low_conf  IoU>=0.5 类别对，但 conf<阈值    （检到但没底气）
      miss      最近同类框 IoU<0.1               （真没框）
    并给出：命中框的 conf 分布、IoU 松弛曲线、按 GT 尺寸/长宽比分档。

用法
    D:/yolo-env/Scripts/python.exe code/tools/diag/diag_bucket_class.py \
        --run nf_v11_896_s42 --bucket 1920x1080 --split test --device cpu
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, 'D:/2/code/yolov11')
from ultralytics import YOLO  # noqa: E402

ROOT = Path('D:/2/dataset_final_bbox')
RUNS = Path('D:/2/exp_upscale/runs')
OUT = Path('D:/2/report/_p2_diag')
NAMES = ['break', 'thunderbolt', 'wear']
IMGSZ, NMS_IOU, MAX_DET = 896, 0.5, 300


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    a = a[:, None, :]
    b = b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0]); iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2]); iy2 = np.minimum(a[..., 3], b[..., 3])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ua = ((a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
          + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - inter)
    return (inter / np.maximum(ua, 1e-9)).astype(np.float32)


def unpack(r):
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, int), np.zeros(0, np.float32)
    return (r.boxes.xyxy.cpu().numpy().astype(np.float32),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy().astype(np.float32))


def load_bucket(split, bucket):
    w, h = [int(x) for x in bucket.split('x')]
    items = []
    for p in sorted((ROOT / 'images' / split).glob('*.jpg')):
        if Image.open(p).size != (w, h):
            continue
        f = ROOT / 'labels' / split / f'{p.stem}.txt'
        gts = []
        if f.exists():
            for ln in f.read_text(encoding='utf-8').splitlines():
                t = ln.split()
                if len(t) < 5:
                    continue
                c, cx, cy, bw, bh = int(float(t[0])), *map(float, t[1:5])
                gts.append(dict(cls=c, box=np.array([
                    (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h], np.float32)))
        items.append((p, gts, w, h))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='nf_v11_896_s42')
    ap.add_argument('--split', default='test')
    ap.add_argument('--bucket', default='1920x1080')
    ap.add_argument('--conf', type=float, default=0.001, help='取框下限（mAP 口径）')
    ap.add_argument('--deploy-conf', type=float, default=0.25, help='部署阈值，用于判定 low_conf')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    items = load_bucket(args.split, args.bucket)
    print(f'{args.run} / {args.split} / 桶 {args.bucket}：{len(items)} 张，'
          f'GT {sum(len(g) for _, g, _, _ in items)} 框，conf {args.conf} / NMS {NMS_IOU}')

    model = YOLO(str(RUNS / args.run / 'weights' / 'best.pt'))
    imgs = [p for p, _, _, _ in items]
    preds = []
    for i in range(0, len(imgs), 4):
        rs = model.predict(imgs[i:i + 4], imgsz=IMGSZ, conf=args.conf, iou=NMS_IOU,
                           max_det=MAX_DET, agnostic_nms=False, verbose=False,
                           device=args.device, stream=False)
        preds.extend(unpack(r) for r in rs)

    # 逐 GT 分解
    stat = defaultdict(lambda: Counter())
    conf_of_hit = defaultdict(list)
    iou_of_hit = defaultdict(list)
    n_pred = Counter()        # 该类预测框总数（conf 0.001 起，NMS 后）
    n_tp = Counter()          # 其中与同类 GT 在 IoU 0.5 内匹配上的
    fp_conf = defaultdict(list)
    rows = []
    for (path, gts, w, h), (b, c, s) in zip(items, preds):
        for cc in c:                      # 预测框计数
            n_pred[NAMES[int(cc)]] += 1
        gb = np.array([g['box'] for g in gts], np.float32) if gts else np.zeros((0, 4), np.float32)
        M = iou_matrix(b, gb)            # (n_pred, n_gt)
        used = np.zeros(len(gts), bool)
        used_pred = np.zeros(len(b), bool)
        # 贪心匹配：按 conf 从高到低，同类优先
        order = np.argsort(-s)
        for j in order:
            same = np.where((M[j] > 0.5) & (~used) & (np.array([g['cls'] for g in gts], int) == c[j]))[0]
            if not len(same):
                same = np.where((M[j] > 0.5) & (~used))[0]
            if len(same):
                k = same[np.argmax(M[j, same])]
                used[k] = True
                used_pred[j] = True
                if NAMES[int(c[j])] == NAMES[int(gts[k]['cls'])]:
                    n_tp[NAMES[int(c[j])]] += 1
                g = gts[k]
                nm = NAMES[g['cls']]
                if g['cls'] == c[j] and s[j] >= args.deploy_conf:
                    stat[nm]['hit'] += 1
                    conf_of_hit[nm].append(float(s[j]))
                    iou_of_hit[nm].append(float(M[j, k]))
                elif g['cls'] == c[j]:
                    stat[nm]['low_conf'] += 1
                    conf_of_hit[nm].append(float(s[j]))
                    iou_of_hit[nm].append(float(M[j, k]))
                else:
                    stat[nm]['cross'] += 1
                    rows.append(dict(img=path.name, cls=nm, tag='cross',
                                     iou=float(M[j, k]), conf=float(s[j]),
                                     pred_cls=NAMES[int(c[j])], gt=[float(x) for x in g['box']]))
        for k, g in enumerate(gts):
            if used[k]:
                continue
            nm = NAMES[g['cls']]
            same = np.where(c == g['cls'])[0]
            best_same = float(M[same, k].max()) if len(same) else 0.0
            if best_same >= 0.5:
                stat[nm]['low_conf'] += 1        # 理论上已在上一步处理
            elif best_same >= 0.1:
                stat[nm]['loc_fail'] += 1
            else:
                stat[nm]['miss'] += 1
            rows.append(dict(img=path.name, cls=nm,
                             tag='loc_fail' if best_same >= 0.1 else 'miss',
                             iou=best_same, conf=0.0, pred_cls='',
                             gt=[float(x) for x in g['box']],
                             side=float(np.sqrt(max((g['box'][2] - g['box'][0]) *
                                                    (g['box'][3] - g['box'][1]), 1e-9)))))
        for j2 in range(len(b)):          # 未匹配上的预测框 = 误检
            if not used_pred[j2]:
                fp_conf[NAMES[int(c[j2])]].append(float(s[j2]))

    print(f'\n{"类":12s} {"GT":>4s} {"hit":>5s} {"low_conf":>9s} {"cross":>6s} {"loc_fail":>9s} {"miss":>5s}   检出率(IoU.5)  部署可用')
    for nm in NAMES:
        c = stat[nm]
        tot = sum(c.values())
        if not tot:
            continue
        hit = c['hit']
        print(f'{nm:12s} {tot:4d} {hit:5d} {c["low_conf"]:9d} {c["cross"]:6d} '
              f'{c["loc_fail"]:9d} {c["miss"]:5d}   '
              f'{(hit + c["low_conf"]) / tot * 100:6.1f}%      {hit / tot * 100:6.1f}%')

    print(f'\n预测框与误检（conf 0.001 起、NMS 后；FP 按「未被任何 GT 匹配」计）:')
    print(f'{"类":12s} {"预测框":>6s} {"TP":>5s} {"FP":>6s} {"FP占比":>7s} {"FP的conf中位":>11s} {"FP<0.25占比":>11s}')
    for nm in NAMES:
        npd, tp = n_pred[nm], n_tp[nm]
        if not npd:
            continue
        fp = npd - tp
        fv = np.array(fp_conf[nm])
        med = f'{np.median(fv):.3f}' if len(fv) else '—'
        lo = f'{(fv < args.deploy_conf).mean() * 100:.0f}%' if len(fv) else '—'
        print(f'{nm:12s} {npd:6d} {tp:5d} {fp:6d} {fp / npd * 100:6.1f}% {med:>11s} {lo:>11s}')

    print(f'\n命中框 conf 分布（hit + low_conf，IoU>=0.5，按类）:')
    print(f'{"类":12s} {"n":>4s} {"<0.1":>6s} {"0.1-0.25":>9s} {"0.25-0.5":>9s} {"0.5-0.75":>9s} {">0.75":>7s}   conf中位')
    for nm in NAMES:
        v = np.array(conf_of_hit[nm])
        if not len(v):
            continue
        b1 = (v < 0.1).sum(); b2 = ((v >= 0.1) & (v < 0.25)).sum()
        b3 = ((v >= 0.25) & (v < 0.5)).sum(); b4 = ((v >= 0.5) & (v < 0.75)).sum()
        b5 = (v >= 0.75).sum()
        print(f'{nm:12s} {len(v):4d} {b1:6d} {b2:9d} {b3:9d} {b4:9d} {b5:7d}   {np.median(v):.3f}')

    print(f'\n命中框 IoU 分布（hit，IoU>=0.5 的那些）:')
    for nm in NAMES:
        v = [i for i, cc in zip(iou_of_hit[nm], conf_of_hit[nm]) if cc >= args.deploy_conf]
        if not v:
            continue
        a = np.array(v)
        print(f'  {nm:12s} n={len(a):3d}  中位 {np.median(a):.3f}  p10 {np.percentile(a, 10):.3f}  '
              f'IoU<0.6 占 {(a < 0.6).mean() * 100:.0f}%  IoU>=0.9 占 {(a >= 0.9).mean() * 100:.0f}%')

    # break 的失败按 GT 尺寸分档
    print('\nbreak 失败案例按 GT 边长分档（原图像素）:')
    fails = [r for r in rows if r['cls'] == 'break']
    if fails:
        buckets = [(0, 40), (40, 80), (80, 160), (160, 1e9)]
        for lo, hi in buckets:
            v = [r for r in fails if lo <= r.get('side', 0) < hi]
            if not v:
                continue
            lf = sum(1 for r in v if r['tag'] == 'loc_fail')
            ms = sum(1 for r in v if r['tag'] == 'miss')
            cr = sum(1 for r in v if r['tag'] == 'cross')
            print(f'  边长 {lo}-{hi if hi < 1e9 else "+"}: 失败 {len(v):2d} 个  '
                  f'框歪 {lf:2d}  漏检 {ms:2d}  认错 {cr:2d}  '
                  f'失败时最佳同类IoU中位 {np.median([r["iou"] for r in v]):.3f}')

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f'break_bucket_{args.run}_{args.bucket}_{args.split}.json'
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n失败明细 → {out}')


if __name__ == '__main__':
    main()
