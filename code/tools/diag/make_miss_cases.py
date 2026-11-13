# -*- coding: utf-8 -*-
"""漏检案例的可视化拼图：判断「框了但 IoU 不够」到底是标注偏还是模型偏。

背景
    diag_miss_iou.py 显示 break 的 27 个漏检里，20 个 IoU 落在 0.2-0.5，即模型框出来了
    但没和标注对上；而 wear 的 13 个漏检里 11 个完全无邻近框。
    前者可能是标注框偏移，后者是真正的召回不足 —— 但光看数字分不清，必须看图。

做法
    对指定的漏检 GT，裁出以它为中心的区域（边长 = max(4*GT边长, 288)），放大到统一尺寸，
    把 GT（绿）与最近的同类预测框（红，标注 IoU）画上去，拼成网格图。
    框贴得准不准、标注有没有偏，一眼可辨。

用法
    python code/tools/diag/make_miss_cases.py --band 0.2,0.5 --cls break --out break_mid
    python code/tools/diag/make_miss_cases.py --band 0,0.05 --cls wear --out wear_none
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, 'D:/2/code/yolov11')
sys.path.insert(0, 'D:/2/code/tools/diag')
from ultralytics import YOLO                     # noqa: E402
from diag_tiled_infer import iou_matrix, load_items  # noqa: E402

RUNS = Path('D:/2/exp_upscale/runs')
OUT = Path('D:/2/report/_p2_diag/miss_cases')
IMGSZ, NMS_IOU, MAX_DET, CONF = 896, 0.5, 300, 0.25
NAMES = ['break', 'thunderbolt', 'wear']
CELL = 300


def unpack(r):
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, int), np.zeros(0, np.float32)
    return (r.boxes.xyxy.cpu().numpy().astype(np.float32),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy().astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='nf_v11_896_s42')
    ap.add_argument('--band', default='0.2,0.5', help='IoU 区间 lo,hi')
    ap.add_argument('--cls', default='break')
    ap.add_argument('--out', default='cases')
    ap.add_argument('--cols', type=int, default=5)
    args = ap.parse_args()

    lo, hi = (float(v) for v in args.band.split(','))
    cls_id = NAMES.index(args.cls)
    OUT.mkdir(parents=True, exist_ok=True)

    items = load_items('test')
    model = YOLO(str(RUNS / args.run / 'weights' / 'best.pt'))
    imgs = [p for p, _, _, _ in items]
    preds = []
    for i in range(0, len(imgs), 4):
        rs = model.predict(imgs[i:i + 4], imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                           max_det=MAX_DET, agnostic_nms=False, verbose=False, stream=False)
        preds.extend(unpack(r) for r in rs)

    # 收集落入 band 的漏检（该类且未在 IoU 0.5 下命中）
    cases = []
    for (path, gts, w0, h0), (b, c, s) in zip(items, preds):
        if not gts:
            continue
        gb = np.array([g['box'] for g in gts], np.float32)
        gc = np.array([g['cls'] for g in gts], int)
        M = iou_matrix(b, gb) if len(b) else np.zeros((0, len(gts)), np.float32)
        # IoU 0.5 下已命中的 GT
        matched = np.zeros(len(gts), bool)
        used = np.zeros(len(b), bool)
        for j in np.argsort(-s) if len(s) else []:
            cand = np.where((M[j] > 0.5) & (~matched) & (gc == c[j]))[0]
            if not len(cand):
                cand = np.where((M[j] > 0.5) & (~matched))[0]
            if len(cand):
                matched[cand[np.argmax(M[j, cand])]] = True
                used[j] = True
        for k, g in enumerate(gts):
            if matched[k] or g['cls'] != cls_id:
                continue
            same = np.where(c == cls_id)[0] if len(b) else np.array([], int)
            if lo == 0 and hi <= 0.05:
                ref = float(M[:, k].max()) if len(M) else -1.0
            else:
                ref = float(M[same, k].max()) if len(same) else -1.0
            if not (lo <= ref < hi):
                continue
            jbest = int(same[np.argmax(M[same, k])]) if len(same) and M[same, k].max() > 0.02 else -1
            cases.append(dict(path=path, gt=g, iou=ref, j=jbest, b=b, c=c, s=s))

    print(f'{args.cls} 在 IoU [{lo},{hi}) 的漏检 {len(cases)} 个，生成拼图…')
    if not cases:
        return

    cells = []
    for cs in cases:
        im = cv2.imread(str(cs['path']))
        g = cs['gt']
        gx = np.array(g['box'], np.float32)
        side = max((gx[2] - gx[0]), (gx[3] - gx[1]))
        half = max(side * 2.0, CELL / 2)
        cx, cy = (gx[0] + gx[2]) / 2, (gx[1] + gx[3]) / 2
        x0 = int(max(0, cx - half)); y0 = int(max(0, cy - half))
        x1 = int(min(im.shape[1], cx + half)); y1 = int(min(im.shape[0], cy + half))
        crop = im[y0:y1, x0:x1].copy()
        sc = CELL / max(crop.shape[:2])
        crop = cv2.resize(crop, (int(crop.shape[1] * sc), int(crop.shape[0] * sc)),
                          interpolation=cv2.INTER_CUBIC)
        # 坐标变换到 crop 空间
        def tf(bx):
            return (int((bx[0] - x0) * sc), int((bx[1] - y0) * sc),
                    int((bx[2] - x0) * sc), int((bx[3] - y0) * sc))
        # 同类预测框（红）
        for bi, ci, si in zip(cs['b'], cs['c'], cs['s']):
            if ci != cls_id:
                continue
            p1, p2 = tf(bi)[:2], tf(bi)[2:]
            if p2[0] < 0 or p2[1] < 0 or p1[0] > CELL or p1[1] > CELL:
                continue
            cv2.rectangle(crop, p1, p2, (40, 40, 230), 1)
        # GT（绿粗）
        p1, p2 = tf(gx)[:2], tf(gx)[2:]
        cv2.rectangle(crop, p1, p2, (60, 210, 60), 2)
        # 画布
        cell = np.full((CELL + 26, CELL, 3), 255, np.uint8)
        ch, cw = crop.shape[:2]
        cell[26:26 + ch, :cw] = crop
        txt = f"{cs['path'].name.replace('firc_xianlan_', '')}  IoU={cs['iou']:.2f}"
        cv2.putText(cell, txt, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cells.append(cell)

    cols = args.cols
    rows = (len(cells) + cols - 1) // cols
    H = (CELL + 26)
    grid = np.full((rows * H + 40, cols * CELL, 3), 255, np.uint8)
    cv2.putText(grid, f'{args.cls}  miss cases  IoU [{lo},{hi})  n={len(cells)}  '
                      f'GREEN=GT  RED=pred', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 2, cv2.LINE_AA)
    for i, c in enumerate(cells):
        r, cc = divmod(i, cols)
        grid[40 + r * H:40 + (r + 1) * H, cc * CELL:(cc + 1) * CELL] = c
        cv2.line(grid, (cc * CELL, 40 + r * H), (cc * CELL, 40 + (r + 1) * H), (200, 200, 200), 1)
        cv2.line(grid, (0, 40 + r * H), (cols * CELL, 40 + r * H), (200, 200, 200), 1)

    p = OUT / f'{args.out}.jpg'
    cv2.imwrite(str(p), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print('输出:', p, grid.shape)
    for cs in cases:
        print(f"  {cs['path'].name:<28} IoU={cs['iou']:.3f}  "
              f"GT边长={max(cs['gt']['box'][2] - cs['gt']['box'][0], cs['gt']['box'][3] - cs['gt']['box'][1]):.0f}px")


if __name__ == '__main__':
    main()
