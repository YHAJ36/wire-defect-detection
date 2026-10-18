# -*- coding: utf-8 -*-
"""Phase 0 follow-up: sweep NMS IoU on the *val* split, then reuse the winner on test.

Why: the first NMS sweep was done on test, so choosing 0.5 from it would be
test-set tuning. This script selects the threshold on val (178 images) only.

Usage:
    python report/phase0_nms_val_sweep.py --imgsz 896 --ious 0.5 0.6 0.7 0.8
"""
import argparse
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'report', '_phase0')
DATA = os.path.join(BASE, 'dataset_final_bbox', 'data.yaml')
SEEDS = [42, 43, 44]


def ckpt_of(seed):
    return os.path.join(BASE, 'exp_upscale', 'runs', 'nf_v11_896_s%d' % seed, 'weights', 'best.pt')


def run_one(model, imgsz, split, iou, tag):
    m = model.val(data=DATA, split=split, imgsz=imgsz, batch=8, workers=0,
                  plots=False, verbose=False, iou=iou, agnostic_nms=False,
                  project=OUT, name=tag, exist_ok=True)
    return {
        'mAP50': float(m.box.map50),
        'mAP50-95': float(m.box.map),
        'P': float(m.box.mp),
        'R': float(m.box.mr),
        'per_class': {m.names[i]: {'ap50': float(m.box.ap50[i]), 'ap': float(m.box.ap[i]),
                                   'p': float(m.box.p[i]), 'r': float(m.box.r[i])}
                      for i in m.names},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imgsz', type=int, default=896)
    ap.add_argument('--ious', type=float, nargs='+', default=[0.5, 0.6, 0.7, 0.8])
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--tag', default='nms_val_sweep')
    args = ap.parse_args()

    from ultralytics import YOLO
    rows = {}
    for seed in SEEDS:
        ckpt = ckpt_of(seed)
        model = YOLO(ckpt)
        params = sum(p.numel() for p in model.model.parameters())
        rows['s%d' % seed] = {'ckpt': ckpt, 'params': int(params), 'runs': {}}
        for iou in args.ious:
            name = '%s_s%d_iou%.1f' % (args.tag, seed, iou)
            r = run_one(model, args.imgsz, args.split, iou, name)
            r['params'] = int(params)
            rows['s%d' % seed]['runs']['%.1f' % iou] = r
            print('s%d iou=%.1f mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f' % (
                seed, iou, r['mAP50'], r['mAP50-95'], r['P'], r['R']), flush=True)

    out_json = os.path.join(OUT, args.tag + '.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'imgsz': args.imgsz, 'split': args.split, 'seeds': rows}, f,
                  ensure_ascii=False, indent=1)

    # markdown summary
    lines = ['# NMS IoU 在 %s 上的扫描（imgsz=%d）' % (args.split, args.imgsz), '',
             '目的：用 val 选 NMS 阈值，避免在 test 上调参。', '']
    for iou in args.ious:
        vals = [rows['s%d' % s]['runs']['%.1f' % iou]['mAP50'] for s in SEEDS]
        vals95 = [rows['s%d' % s]['runs']['%.1f' % iou]['mAP50-95'] for s in SEEDS]
        ps = [rows['s%d' % s]['runs']['%.1f' % iou]['P'] for s in SEEDS]
        rs = [rows['s%d' % s]['runs']['%.1f' % iou]['R'] for s in SEEDS]
        lines.append('- IoU %.1f: mAP50 均值 %.4f [%.4f, %.4f]；mAP50-95 均值 %.4f；P %.4f；R %.4f' % (
            iou, sum(vals) / 3, min(vals), max(vals), sum(vals95) / 3,
            sum(ps) / 3, sum(rs) / 3))
    lines += ['', '## 逐 seed', '',
              '| seed | NMS IoU | mAP50 | mAP50-95 | P | R | break AP50 | thunder AP50 | wear AP50 |',
              '|---|---|---|---|---|---|---|---|---|']
    for seed in SEEDS:
        for iou in args.ious:
            r = rows['s%d' % seed]['runs']['%.1f' % iou]
            pc = r['per_class']
            lines.append('| s%d | %.1f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |' % (
                seed, iou, r['mAP50'], r['mAP50-95'], r['P'], r['R'],
                pc['break']['ap50'], pc['thunderbolt']['ap50'], pc['wear']['ap50']))
    md = os.path.join(BASE, 'report', 'phase0_%s.md' % args.tag)
    with open(md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK', out_json)
    print('OK', md)


if __name__ == '__main__':
    main()
