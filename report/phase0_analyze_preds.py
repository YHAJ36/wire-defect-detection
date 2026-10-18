# -*- coding: utf-8 -*-
"""Phase 0: error decomposition for one detections dump.

Usage:
    python report/phase0_analyze_preds.py --tag v11_896_s42 --imgsz 896
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'report', '_phase0')
DS = os.path.join(BASE, 'dataset_final_bbox')
NAMES = ['break', 'thunderbolt', 'wear']


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def size_bucket(side_px):
    a = side_px * side_px
    if a < 32 * 32:
        return 'small(<32px)'
    if a < 96 * 96:
        return 'medium(32-96px)'
    return 'large(>96px)'


def load_gt(rec):
    h, w = rec['height'], rec['width']
    stem = os.path.splitext(rec['image'])[0]
    path = os.path.join(DS, 'labels', 'test', stem + '.txt')
    gt = []
    if os.path.exists(path):
        for ln in open(path, encoding='utf-8'):
            p = ln.split()
            if len(p) != 5:
                continue
            c = int(float(p[0]))
            cx, cy, bw, bh = [float(v) for v in p[1:]]
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            gt.append({'cls': c, 'box': [x1, y1, x2, y2],
                       'area': bw * bh, 'side896': math.sqrt(bw * bh) * 896})
    return gt


def match(rec, conf_thr):
    gt = load_gt(rec)
    dets = [d for d in rec['dets'] if d[1] >= conf_thr]
    dets.sort(key=lambda x: -x[1])
    for g in gt:
        g['hit'] = False
    tp = fp_bg = fp_cls = fp_dup = 0
    fn_boxes = []
    fp_boxes = []
    for c, s, x1, y1, x2, y2 in dets:
        box = [x1, y1, x2, y2]
        best_i, best_v = -1, 0.0
        for k, g in enumerate(gt):
            if g['cls'] != c:
                continue
            v = iou(box, g['box'])
            if v > best_v:
                best_i, best_v = k, v
        if best_v >= 0.5 and not gt[best_i]['hit']:
            gt[best_i]['hit'] = True
            tp += 1
        elif best_v >= 0.5 and gt[best_i]['hit']:
            fp_dup += 1
            fp_boxes.append({'cls': c, 'conf': s, 'box': box, 'side896': math.sqrt((x2 - x1) * (y2 - y1))})
        else:
            # class confusion: overlap >= 0.5 with an unmatched GT of another class
            other, ov = None, 0.0
            for g in gt:
                if g['cls'] == c or g['hit']:
                    continue
                v = iou(box, g['box'])
                if v > ov:
                    other, ov = g['cls'], v
            if ov >= 0.5:
                fp_cls += 1
            else:
                fp_bg += 1
            fp_boxes.append({'cls': c, 'conf': s, 'box': box, 'side896': math.sqrt((x2 - x1) * (y2 - y1))})
    for g in gt:
        if not g['hit']:
            fn_boxes.append({'cls': g['cls'], 'area': g['area'], 'side896': g['side896'],
                             'box': g['box']})
    return {'image': rec['image'], 'gt': len(gt), 'tp': tp, 'fp_bg': fp_bg,
            'fp_cls': fp_cls, 'fp_dup': fp_dup, 'fn': len(fn_boxes),
            'fn_boxes': fn_boxes, 'fp_boxes': fp_boxes, 'gt_boxes': gt}


def ceiling(rec):
    """Per-GT best same-class IoU at conf=0.001, no one-to-one constraint."""
    gt = load_gt(rec)
    dets = rec['dets']
    hits = {0.5: 0, 0.3: 0, 0.1: 0}
    for g in gt:
        best = 0.0
        for c, s, x1, y1, x2, y2 in dets:
            if c != g['cls']:
                continue
            best = max(best, iou([x1, y1, x2, y2], g['box']))
        for t in hits:
            if best >= t:
                hits[t] += 1
    return {'gt': len(gt), 'iou50': hits[0.5], 'iou30': hits[0.3], 'iou10': hits[0.1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--imgsz', type=int, default=896)
    args = ap.parse_args()
    fn = os.path.join(OUT, args.tag + '_preds.json')
    recs = json.load(open(fn, encoding='utf-8'))

    result = {'tag': args.tag, 'imgsz': args.imgsz, 'thresholds': {}}
    for thr in (0.25, 0.5):
        agg = {n: Counter() for n in NAMES}
        per_img = []
        for rec in recs:
            r = match(rec, thr)
            per_img.append({k: r[k] for k in ('image', 'gt', 'tp', 'fp_bg', 'fp_cls', 'fp_dup', 'fn')})
            for g in r['gt_boxes']:
                pass
            # attribute counts by predicted class for FP; by GT class for FN
            for key in ('tp',):
                pass
            # recompute per-class attribution cheaply
            # TP/FN from GT matched flags
            for g in r['gt_boxes']:
                n = NAMES[g['cls']]
                if g['hit']:
                    agg[n]['tp'] += 1
                else:
                    agg[n]['fn'] += 1
            # FP by predicted class needs the boxes; derive from totals is not enough.
            # Re-run a compact match to collect per-class FP.
            gt = load_gt(rec)
            for g in gt:
                g['hit'] = False
            dets = sorted([d for d in rec['dets'] if d[1] >= thr], key=lambda x: -x[1])
            for c, s, x1, y1, x2, y2 in dets:
                box = [x1, y1, x2, y2]
                best_i, best_v = -1, 0.0
                for k, g in enumerate(gt):
                    if g['cls'] != c:
                        continue
                    v = iou(box, g['box'])
                    if v > best_v:
                        best_i, best_v = k, v
                n = NAMES[c]
                if best_v >= 0.5 and not gt[best_i]['hit']:
                    gt[best_i]['hit'] = True
                elif best_v >= 0.5 and gt[best_i]['hit']:
                    agg[n]['fp_dup'] += 1
                else:
                    other, ov = None, 0.0
                    for g in gt:
                        if g['cls'] == c or g['hit']:
                            continue
                        v = iou(box, g['box'])
                        if v > ov:
                            other, ov = g['cls'], v
                    if ov >= 0.5:
                        agg[n]['fp_cls'] += 1
                    else:
                        agg[n]['fp_bg'] += 1

        # FN size / class breakdown
        fn_size = defaultdict(Counter)
        fp_size = defaultdict(Counter)
        fn_conf = defaultdict(Counter)
        for rec in recs:
            r = match(rec, thr)
            for b in r['fn_boxes']:
                fn_size[NAMES[b['cls']]][size_bucket(b['side896'])] += 1
            for b in r['fp_boxes']:
                fp_size[NAMES[b['cls']]][size_bucket(b['side896'])] += 1
        result['thresholds'][str(thr)] = {
            'per_class': {n: dict(agg[n]) for n in NAMES},
            'fn_by_size': {n: dict(fn_size[n]) for n in NAMES},
            'fp_by_size': {n: dict(fp_size[n]) for n in NAMES},
            'top_images': sorted(per_img, key=lambda x: -(x['fn'] + x['fp_bg'] + x['fp_cls'] + x['fp_dup']))[:10],
        }

    # detection ceiling at conf=0.001, per class
    ceil = {n: Counter() for n in NAMES}
    for rec in recs:
        gt = load_gt(rec)
        for g in gt:
            n = NAMES[g['cls']]
            ceil[n]['gt'] += 1
            best = 0.0
            for c, s, x1, y1, x2, y2 in rec['dets']:
                if c != g['cls']:
                    continue
                best = max(best, iou([x1, y1, x2, y2], g['box']))
            if best >= 0.5:
                ceil[n]['detected_iou50'] += 1
            if best >= 0.3:
                ceil[n]['detected_iou30'] += 1
            if best >= 0.1:
                ceil[n]['detected_iou10'] += 1
    result['ceiling_conf0001'] = {n: dict(ceil[n]) for n in NAMES}

    # confidence -> F1 sweep per class
    grid = [round(0.05 + 0.05 * k, 2) for k in range(18)]
    f1 = {}
    for n in NAMES:
        rows = []
        for thr in grid:
            tp = fp = fn = 0
            for rec in recs:
                r = match(rec, thr)
                for g in r['gt_boxes']:
                    if NAMES[g['cls']] == n:
                        if g['hit']:
                            tp += 1
                        else:
                            fn += 1
                for b in r['fp_boxes']:
                    if NAMES[b['cls']] == n:
                        fp += 1
            p = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f = 2 * p * rc / (p + rc) if p + rc else 0.0
            rows.append({'conf': thr, 'tp': tp, 'fp': fp, 'fn': fn, 'P': p, 'R': rc, 'F1': f})
        best = max(rows, key=lambda x: x['F1'])
        f1[n] = {'best': best, 'rows': rows}
    result['f1_sweep'] = f1

    # Diagnose FN at conf 0.25: low-confidence correct box vs localization vs no box.
    fn_diag = {n: Counter() for n in NAMES}
    fn_examples = defaultdict(list)
    thr = 0.25
    for rec in recs:
        gt = load_gt(rec)
        dets = [d for d in rec['dets'] if d[1] >= thr]
        dets.sort(key=lambda x: -x[1])
        for g in gt:
            g['hit'] = False
        for c, s, x1, y1, x2, y2 in dets:
            box = [x1, y1, x2, y2]
            best_i, best_v = -1, 0.0
            for k, g in enumerate(gt):
                if g['cls'] != c:
                    continue
                v = iou(box, g['box'])
                if v > best_v:
                    best_i, best_v = k, v
            if best_v >= 0.5 and not gt[best_i]['hit']:
                gt[best_i]['hit'] = True
        for g in gt:
            if g['hit']:
                continue
            n = NAMES[g['cls']]
            best_v, best_conf = 0.0, 0.0
            for c, s, x1, y1, x2, y2 in rec['dets']:
                if c != g['cls']:
                    continue
                v = iou([x1, y1, x2, y2], g['box'])
                if v > best_v:
                    best_v, best_conf = v, s
            if best_v >= 0.5 and best_conf >= thr:
                cat = 'conflict(同框竞争)'
            elif best_v >= 0.5:
                cat = 'low_conf(有正确框但<0.25)'
            elif best_v >= 0.3:
                cat = 'localization(0.3-0.5)'
            elif best_v >= 0.1:
                cat = 'gross(<0.3)'
            else:
                cat = 'no_box(无同类框)'
            fn_diag[n][cat] += 1
            if len(fn_examples[n]) < 8:
                fn_examples[n].append({'image': rec['image'], 'best_iou': round(best_v, 3),
                                       'best_conf': round(best_conf, 4), 'cat': cat,
                                       'gt_side896': round(g['side896'], 1)})
    result['fn_diagnosis_conf025'] = {n: dict(fn_diag[n]) for n in NAMES}
    result['fn_examples'] = {n: fn_examples[n] for n in NAMES}

    out = os.path.join(OUT, args.tag + '_errors.json')
    json.dump(result, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    lines = [f'# Phase 0 误差分解：{args.tag}（imgsz={args.imgsz}）', '',
             '匹配规则：按置信度降序、同类、IoU≥0.5、一对一；conf 阈值分别 0.25 / 0.5。', '']
    for thr in (0.25, 0.5):
        t = result['thresholds'][str(thr)]
        lines += [f'## conf ≥ {thr}', '',
                  '| 类别 | TP | FN | FP(背景) | FP(类间) | FP(重复) | 查全率 | 查准率 |',
                  '|---|---|---|---|---|---|---|---|']
        for n in NAMES:
            a = t['per_class'][n]
            tp = a.get('tp', 0); fn = a.get('fn', 0)
            fp = a.get('fp_bg', 0) + a.get('fp_cls', 0) + a.get('fp_dup', 0)
            rec = tp / (tp + fn) if tp + fn else 0
            prec = tp / (tp + fp) if tp + fp else 0
            lines.append('| %s | %d | %d | %d | %d | %d | %.3f | %.3f |' % (
                n, tp, fn, a.get('fp_bg', 0), a.get('fp_cls', 0), a.get('fp_dup', 0), rec, prec))
        lines += ['', 'FN 按 GT 尺寸（896 像素）：', '',
                  '| 类别 | small | medium | large |', '|---|---|---|---|']
        for n in NAMES:
            b = t['fn_by_size'][n]
            lines.append('| %s | %d | %d | %d |' % (
                n, b.get('small(<32px)', 0), b.get('medium(32-96px)', 0), b.get('large(>96px)', 0)))
        lines.append('')
    lines += ['## 检测天花板（conf=0.001，不限制一对一）', '',
              '| 类别 | GT | IoU≥0.5 有同类框 | IoU≥0.3 | IoU≥0.1 |', '|---|---|---|---|---|']
    for n in NAMES:
        c = result['ceiling_conf0001'][n]
        lines.append('| %s | %d | %d | %d | %d |' % (
            n, c.get('gt', 0), c.get('detected_iou50', 0), c.get('detected_iou30', 0), c.get('detected_iou10', 0)))
    lines += ['', '## 置信度–F1 扫描（IoU 0.5）', '',
              '| 类别 | 最佳 conf | 最佳 F1 | P | R |', '|---|---|---|---|---|']
    for n in NAMES:
        b = result['f1_sweep'][n]['best']
        lines.append('| %s | %.2f | %.4f | %.4f | %.4f |' % (n, b['conf'], b['F1'], b['P'], b['R']))
    lines += ['', '## FN 归因（conf≥0.25，IoU 0.5）', '',
              '| 类别 | 低置信正确框 | 定位 0.3–0.5 | 严重错位 <0.3 | 无同类框 | 同框竞争 |',
              '|---|---|---|---|---|---|']
    for n in NAMES:
        d = result['fn_diagnosis_conf025'][n]
        lines.append('| %s | %d | %d | %d | %d | %d |' % (
            n, d.get('low_conf(有正确框但<0.25)', 0), d.get('localization(0.3-0.5)', 0),
            d.get('gross(<0.3)', 0), d.get('no_box(无同类框)', 0), d.get('conflict(同框竞争)', 0)))
    lines.append('')
    md_out = os.path.join(BASE, 'report', 'phase0_error_decomp_%s.md' % args.tag)
    with open(md_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK', out)
    print('OK', md_out)


if __name__ == '__main__':
    main()
