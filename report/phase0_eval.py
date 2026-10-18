# -*- coding: utf-8 -*-
"""Phase 0: re-evaluate a checkpoint on the test split and dump detections.

Usage:
    python report/phase0_eval.py --ckpt <best.pt> --imgsz 896 --tag v11_896_s42 --mode both
"""
import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# WIoU v3 的 checkpoint 里 criterion 的类型是 wiou_v3.PatchedBboxLoss；
# torch.load 反序列化时需要该模块可导入。
sys.path.insert(0, os.path.join(BASE, 'code', 'yolov11'))
DATA = os.path.join(BASE, 'dataset_final_bbox', 'data.yaml')
TEST_IMAGES = os.path.join(BASE, 'dataset_final_bbox', 'images', 'test')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--imgsz', type=int, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--mode', choices=['val', 'pred', 'both'], default='both')
    ap.add_argument('--split', choices=['val', 'test'], default='test')
    ap.add_argument('--out', default=os.path.join(BASE, 'report', '_phase0'))
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--nms', type=float, default=0.7)
    ap.add_argument('--agnostic', action='store_true')
    args = ap.parse_args()

    global OUT
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.ckpt)
    out = {'ckpt': args.ckpt, 'imgsz': args.imgsz, 'tag': args.tag,
           'split': args.split, 'nms': args.nms, 'agnostic': args.agnostic}

    if args.mode in ('val', 'both'):
        m = model.val(data=DATA, split=args.split, imgsz=args.imgsz, batch=args.batch,
                      plots=False, verbose=False, workers=0,
                      iou=args.nms, agnostic_nms=args.agnostic,
                      project=OUT, name=args.tag + '_val', exist_ok=True)
        names = m.names
        per_class = []
        for i, n in names.items():
            per_class.append({
                'class': n,
                'ap50': float(m.box.ap50[i]),
                'ap': float(m.box.ap[i]),
                'p': float(m.box.p[i]),
                'r': float(m.box.r[i]),
            })
        out['val'] = {
            'mAP50': float(m.box.map50), 'mAP50-95': float(m.box.map),
            'P': float(m.box.mp), 'R': float(m.box.mr),
            'per_class': per_class,
        }
        print('VAL', args.tag, 'mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f' % (
            out['val']['mAP50'], out['val']['mAP50-95'], out['val']['P'], out['val']['R']))
        for c in per_class:
            print('  %-12s AP50=%.4f AP=%.4f P=%.4f R=%.4f' % (
                c['class'], c['ap50'], c['ap'], c['p'], c['r']))

    if args.mode in ('pred', 'both'):
        rows = []
        results = model.predict(source=TEST_IMAGES, imgsz=args.imgsz, conf=0.001,
                                iou=args.nms, agnostic_nms=args.agnostic, max_det=300,
                                batch=args.batch, workers=0, stream=True,
                                verbose=False, save=False)
        for r in results:
            dets = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy().tolist()
                conf = r.boxes.conf.cpu().numpy().tolist()
                cls = r.boxes.cls.cpu().numpy().astype(int).tolist()
                for b, s, c in zip(xyxy, conf, cls):
                    dets.append([c, round(float(s), 6)] + [round(float(v), 2) for v in b])
            rows.append({'image': os.path.basename(r.path),
                         'width': int(r.orig_shape[1]), 'height': int(r.orig_shape[0]),
                         'dets': dets})
        fn = os.path.join(OUT, args.tag + '_preds.json')
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False)
        print('PRED', fn, len(rows), 'images')

    with open(os.path.join(OUT, args.tag + '_eval.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
