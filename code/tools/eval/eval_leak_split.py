# -*- coding: utf-8 -*-
"""近邻帧泄漏分层评估：test 里有多少指标是「记住的」。

背景（A.9.44）
    数据是视频截帧（632x480 桶带 BANDICAM 水印），但划分是按文件随机抽的，
    编号在 train/val/test 三个集合里完全交错。实测最像的几对是相邻编号
    （test xianlan_1300 ← train xianlan_1301，64x64 灰度 MAE=1.0），
    即同一段视频的相邻帧被分到了不同集合。

做法
    对 test 每张图，在同分辨率桶的 train 图里找 64x64 灰度 MAE 最小的一张，
    按 MAE 分层（leak: <lo，mid: [lo,hi)，clean: >=hi），
    各层写清单 yaml，走与 eval_by_resolution.py 完全相同的 val 口径，
    得到「无近邻」「轻度」「高度重复」三个子集各自的 mAP50。

用法
    D:/yolo-env/Scripts/python.exe code/tools/eval/eval_leak_split.py \
        --run nf_v11_896_s42 --split test --device cpu
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from ultralytics import YOLO  # noqa: E402

ROOT = Path('D:/2/dataset_final_bbox')
RUNS = Path('D:/2/exp_upscale/runs')
WORK = Path('D:/2/report/_p2_diag/leak_split')
NAMES = ['break', 'thunderbolt', 'wear']
S = 64


def arr(p):
    im = Image.open(p).convert('L').resize((S, S), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--imgsz', type=int, default=896)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--lo', type=float, default=4.0, help='MAE 低于此值 = 高度重复（近邻帧）')
    ap.add_argument('--hi', type=float, default=8.0, help='MAE 低于此值 = 轻度相似')
    args = ap.parse_args()

    w_path = RUNS / args.run / 'weights' / 'best.pt'
    if not w_path.exists():
        raise SystemExit(f'❌ 缺权重：{w_path}')
    img_dir = ROOT / 'images' / args.split
    lbl_dir = ROOT / 'labels' / args.split

    # 1) 建 train 索引（按分辨率分桶，只跟同桶比，跨桶尺寸/构图不可比）
    tr = defaultdict(list)
    for p in sorted((ROOT / 'images' / 'train').glob('*.jpg')):
        tr[Image.open(p).size].append(arr(p))

    # 2) 给每张评估图找最近邻 MAE
    rows = []
    for p in sorted(img_dir.glob('*.jpg')):
        size = Image.open(p).size
        cand = tr.get(size)
        if not cand:                      # 该尺寸 train 里没有，无法判定 → 归为 clean
            rows.append((p, 999.0, None))
            continue
        a = arr(p).ravel()
        T = np.stack(cand).reshape(len(cand), -1)
        maes = np.abs(T - a.reshape(1, -1)).mean(axis=1)
        j = int(maes.argmin())
        rows.append((p, float(maes[j]), j))

    # 名字里不能带 < >（Windows 文件名非法），用 L/M/C 前缀
    layers = {
        f'L_leak_lt{args.lo:g}': [p for p, m, _ in rows if m < args.lo],
        f'M_mid{args.lo:g}to{args.hi:g}': [p for p, m, _ in rows if args.lo <= m < args.hi],
        f'C_clean_ge{args.hi:g}': [p for p, m, _ in rows if m >= args.hi],
    }
    allm = np.array([m for _, m, _ in rows])
    print(f'=== {args.run} / {args.split} 近邻 MAE 分布（n={len(rows)}）===')
    print(f'  中位={np.median(allm):.2f}  p25={np.percentile(allm, 25):.2f}  '
          f'p75={np.percentile(allm, 75):.2f}  最小={allm.min():.2f}')
    for k, v in layers.items():
        print(f'  {k:20s} 图{len(v):4d}  ({len(v)/len(rows)*100:5.1f}%)')

    # 3) 逐层评估（口径与 eval_by_resolution.py 一致）
    WORK.mkdir(parents=True, exist_ok=True)
    results = []
    for name, paths in layers.items():
        if not paths:
            continue
        lst = WORK / f'{args.split}_{name}.txt'
        lst.write_text('\n'.join(str(p) for p in paths), encoding='utf-8')
        yml = WORK / f'{args.split}_{name}.yaml'
        yml.write_text(
            f'path: {ROOT}\ntrain: images/train\nval: {lst}\ntest: {lst}\n'
            f'nc: 3\nnames: {NAMES}\n', encoding='utf-8')

        n_gt = 0
        cls_n = defaultdict(int)
        for p in paths:
            f = lbl_dir / f'{p.stem}.txt'
            if f.exists():
                for ln in f.read_text(encoding='utf-8').splitlines():
                    if ln.strip():
                        n_gt += 1
                        cls_n[NAMES[int(float(ln.split()[0]))]] += 1

        m = YOLO(str(w_path))
        r = m.val(data=str(yml), split=args.split, imgsz=args.imgsz, batch=8,
                  workers=0, device=args.device, plots=False, verbose=False,
                  iou=args.iou, agnostic_nms=False,
                  project=str(WORK), name=f'{args.run}_{args.split}_{name}',
                  exist_ok=True)
        b = r.box
        pc = {r.names[int(c)]: float(b.ap50[j]) for j, c in enumerate(b.ap_class_index)}
        results.append(dict(layer=name, n_img=len(paths), n_gt=n_gt,
                            mAP50=float(b.map50), mAP95=float(b.map),
                            P=float(b.mp), R=float(b.mr), per_class=pc,
                            cls_n=dict(cls_n)))
        print(f'  {name:20s} 图{len(paths):4d} 框{n_gt:4d}  '
              f'mAP50={float(b.map50):.4f}  mAP50-95={float(b.map):.4f}  '
              f'R={float(b.mr):.4f}  ' + ', '.join(f'{k}={v:.3f}' for k, v in pc.items()))

    out = Path(f'D:/2/report/_p2_diag/leak_{args.run}_{args.split}.json')
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
