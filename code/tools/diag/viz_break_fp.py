# -*- coding: utf-8 -*-
"""把 break 的高置信度误检画出来，人眼核对是「模型乱猜」还是「标注漏了」。

背景（A.9.45）
    1920x1080 桶里 break 的 AP50 只有 0.803，但 0 个真漏检、0 个认错类；
    77 个 GT 中 71 个在 IoU 0.5 内框到了。真正的问题是预测框数量：
    break 报 438 个（每图 6.2 个），GT 只有 77 个（每图 1.08 个），
    368 个误检里 95% 的 conf < 0.25（极低置信噪声）。
    thunder 只报 2.35 倍、AP 0.995 ⇒ 倍数与 AP 一一对应。

    剩下要判的是：那些「conf 不算太低」的误检，是模型在背景上乱响应，
    还是那些位置本来就有断股、只是标注没标？两者结论完全相反：
      前者 => 要压背景响应（正则/负样本）
      后者 => 是标注不完整，修标注直接抬 AP，且这批 95% 的低分框
              也可能大量是漏标的真目标，意味着当前 AP 被系统性低估。

做法
    只画 conf >= thr 且未匹配任何同类 GT 的 break 预测（红），
    以及匹配上的 TP（蓝）、GT（绿）。按误检数排序取前 N 张拼图。

用法
    D:/yolo-env/Scripts/python.exe code/tools/diag/viz_break_fp.py \
        --run nf_v11_896_s42 --bucket 1920x1080 --thr 0.25 --top 8 --device cpu
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, 'D:/2/code/yolov11')
from ultralytics import YOLO  # noqa: E402

ROOT = Path('D:/2/dataset_final_bbox')
RUNS = Path('D:/2/exp_upscale/runs')
OUT = Path('D:/2/report/_p2_diag/break_fp')
NAMES = ['break', 'thunderbolt', 'wear']
IMGSZ, NMS_IOU = 896, 0.5


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    a = a[:, None, :]; b = b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0]); iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2]); iy2 = np.minimum(a[..., 3], b[..., 3])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ua = ((a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
          + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - inter)
    return (inter / np.maximum(ua, 1e-9)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='nf_v11_896_s42')
    ap.add_argument('--split', default='test')
    ap.add_argument('--bucket', default='1920x1080')
    ap.add_argument('--thr', type=float, default=0.25, help='只画 conf>=thr 的误检')
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    w, h = [int(x) for x in args.bucket.split('x')]
    items = []
    for p in sorted((ROOT / 'images' / args.split).glob('*.jpg')):
        if Image.open(p).size != (w, h):
            continue
        f = ROOT / 'labels' / args.split / f'{p.stem}.txt'
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
        items.append((p, gts))

    model = YOLO(str(RUNS / args.run / 'weights' / 'best.pt'))
    imgs = [p for p, _ in items]
    preds = []
    for i in range(0, len(imgs), 4):
        rs = model.predict(imgs[i:i + 4], imgsz=IMGSZ, conf=0.001, iou=NMS_IOU,
                           max_det=300, agnostic_nms=False, verbose=False,
                           device=args.device, stream=False)
        for r in rs:
            if r.boxes is None or len(r.boxes) == 0:
                preds.append((np.zeros((0, 4), np.float32), np.zeros(0, int), np.zeros(0, np.float32)))
            else:
                preds.append((r.boxes.xyxy.cpu().numpy().astype(np.float32),
                              r.boxes.cls.cpu().numpy().astype(int),
                              r.boxes.conf.cpu().numpy().astype(np.float32)))

    cases = []
    for (path, gts), (b, c, s) in zip(items, preds):
        if not gts:
            continue
        gb = np.array([g['box'] for g in gts], np.float32)
        M = iou_matrix(b, gb)
        used = np.zeros(len(gts), bool)
        matched_pred = np.zeros(len(b), bool)
        for j in np.argsort(-s):
            same = np.where((M[j] > 0.5) & (~used) & (np.array([g['cls'] for g in gts], int) == c[j]))[0]
            if not len(same):
                same = np.where((M[j] > 0.5) & (~used))[0]
            if len(same):
                k = same[np.argmax(M[j, same])]
                used[k] = True; matched_pred[j] = True
        fpidx = [j for j in range(len(b))
                 if c[j] == 0 and not matched_pred[j] and s[j] >= args.thr]
        if fpidx:
            cases.append((len(fpidx), float(max(s[j] for j in fpidx)), path, gts, b, c, s, fpidx))

    cases.sort(key=lambda x: (-x[0], -x[1]))
    print(f'conf>={args.thr} 的 break 误检，涉及 {len(cases)} 张图；'
          f'前 {min(args.top, len(cases))} 张已画出')
    for n, mx, path, *_ in cases[:args.top]:
        print(f'  {path.name:34s} 误检 {n:2d} 个  最高 conf {mx:.3f}')

    OUT.mkdir(parents=True, exist_ok=True)
    tiles = []
    for n, mx, path, gts, b, c, s, fpidx in cases[:args.top]:
        im = Image.open(path).convert('RGB')
        d = ImageDraw.Draw(im)
        for g in gts:                       # 绿 = 标注
            if g['cls'] == 0:
                d.rectangle([float(x) for x in g['box']], outline=(0, 255, 60), width=4)
        for j in range(len(b)):
            if c[j] != 0:
                continue
            if matched_pred_flag := (j not in fpidx):
                d.rectangle([float(x) for x in b[j]], outline=(70, 150, 255), width=2)   # 蓝 = 命中
            else:
                d.rectangle([float(x) for x in b[j]], outline=(255, 40, 40), width=4)    # 红 = 误检
                d.text((float(b[j][0]) + 5, float(b[j][1]) + 3), f'{s[j]:.2f}', fill=(255, 40, 40))
        W = 760
        im = im.resize((W, int(im.height * W / im.width)))
        canv = Image.new('RGB', (W, im.height + 20), (255, 255, 255))
        canv.paste(im, (0, 20))
        ImageDraw.Draw(canv).text((3, 3), f'{path.name}  FP={n}  maxconf={mx:.2f}', fill=(0, 0, 0))
        tiles.append(canv)

    if tiles:
        cols = 2
        rows = (len(tiles) + cols - 1) // cols
        tw, th = tiles[0].size
        sheet = Image.new('RGB', (tw * cols, th * rows), (235, 235, 235))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
        outp = OUT / f'break_fp_{args.run}_{args.bucket}_conf{args.thr:g}.jpg'
        sheet.save(outp, quality=86)
        print(f'\n→ {outp}  ({sheet.size[0]}x{sheet.size[1]})')


if __name__ == '__main__':
    main()
