# -*- coding: utf-8 -*-
"""Phase 0: actual vs oracle AP50 from a detections dump.

Oracle AP = AP50 when every box that can match a GT at IoU 0.5 is ranked above
every other box (i.e. the best achievable AP from the current proposal set).
"""
import argparse
import json
import math
import os

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


def gt_of(rec):
    h, w = rec['height'], rec['width']
    stem = os.path.splitext(rec['image'])[0]
    path = os.path.join(DS, 'labels', 'test', stem + '.txt')
    out = []
    if os.path.exists(path):
        for ln in open(path, encoding='utf-8'):
            p = ln.split()
            if len(p) != 5:
                continue
            c = int(float(p[0]))
            cx, cy, bw, bh = [float(v) for v in p[1:]]
            out.append({'cls': c, 'box': [(cx - bw / 2) * w, (cy - bh / 2) * h,
                                          (cx + bw / 2) * w, (cy + bh / 2) * h]})
    return out


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]) if xs[i] > xs[i - 1] else 0.0
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def voc_ap(rec, prec):
    """101-point interpolated AP (same convention as Ultralytics)."""
    mrec = [0.0] + rec + [1.0]
    mpre = [1.0] + prec + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    xs = [k / 100.0 for k in range(101)]
    ps = [_interp(x, mrec, mpre) for x in xs]
    return sum((ps[i] + ps[i + 1]) * 0.5 * 0.01 for i in range(100))


def collect(recs, mode, conf_min=0.001):
    """Return per-class list of (score, tp). mode=actual or oracle."""
    buckets = {n: [] for n in NAMES}
    for rec in recs:
        gt = gt_of(rec)
        for c in range(len(NAMES)):
            gts = [g for g in gt if g['cls'] == c]
            dets = [d for d in rec['dets'] if d[0] == c and d[1] >= conf_min]
            if not gts and not dets:
                continue
            for g in gts:
                g['hit'] = False
            if mode == 'actual':
                dets.sort(key=lambda x: -x[1])
                scored = [(d[1], d) for d in dets]
            else:
                # rank by best IoU with an unmatched GT to maximize matches
                tmp = []
                for d in dets:
                    box = d[2:6]
                    best = max([iou(box, g['box']) for g in gts] or [0.0])
                    tmp.append((best, d))
                tmp.sort(key=lambda x: -x[0])
                scored = [(d[1], d) for _, d in tmp]
            for s, d in scored:
                box = d[2:6]
                hit = False
                best_i, best_v = -1, 0.0
                for k, g in enumerate(gts):
                    v = iou(box, g['box'])
                    if v > best_v:
                        best_i, best_v = k, v
                if best_v >= 0.5 and not gts[best_i]['hit']:
                    gts[best_i]['hit'] = True
                    hit = True
                score = s if mode == 'actual' else (1.0 if hit else 0.0)
                buckets[NAMES[c]].append((score, 1 if hit else 0))
    return buckets


def ap50(pairs):
    if not pairs:
        return 0.0
    pairs = sorted(pairs, key=lambda x: -x[0])
    tp = fp = 0
    rec, prec = [], []
    npos = 0
    # npos not known here; caller supplies via gt count -> recompute with total
    for s, t in pairs:
        tp += t
        fp += 1 - t
        rec.append(tp)
        prec.append(tp / (tp + fp))
    return rec, prec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    recs = json.load(open(os.path.join(OUT, args.tag + '_preds.json'), encoding='utf-8'))
    gt_count = {n: 0 for n in NAMES}
    for rec in recs:
        for g in gt_of(rec):
            gt_count[NAMES[g['cls']]] += 1

    res = {'tag': args.tag, 'gt': gt_count, 'classes': {}}
    for mode in ('actual', 'oracle'):
        b = collect(recs, mode)
        for n in NAMES:
            pairs = b[n]
            pairs = sorted(pairs, key=lambda x: -x[0])
            tp = fp = 0
            rec_list, prec_list = [], []
            for s, t in pairs:
                tp += t
                fp += 1 - t
                rec_list.append(tp / gt_count[n] if gt_count[n] else 0)
                prec_list.append(tp / (tp + fp) if tp + fp else 0)
            res['classes'].setdefault(n, {})[mode] = voc_ap(rec_list, prec_list)
            res['classes'][n][mode + '_tp'] = tp
            res['classes'][n][mode + '_fp'] = fp
    m_actual = sum(res['classes'][n]['actual'] for n in NAMES) / len(NAMES)
    m_oracle = sum(res['classes'][n]['oracle'] for n in NAMES) / len(NAMES)
    res['mAP50_actual'] = m_actual
    res['mAP50_oracle'] = m_oracle
    json.dump(res, open(os.path.join(OUT, args.tag + '_ap_oracle.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    lines = ['# AP50 实际 vs 天花板（当前候选框集合）', '',
             f'检测：`{args.tag}`。Oracle = 把能匹配 GT（IoU≥0.5）的框全部排在前面时的 AP50。', '',
             '| 类别 | GT | 实际 AP50 | Oracle AP50 | 差距 | TP 框 | 全部框 |',
             '|---|---|---|---|---|---|---|']
    for n in NAMES:
        c = res['classes'][n]
        lines.append('| %s | %d | %.4f | %.4f | %.4f | %d | %d |' % (
            n, gt_count[n], c['actual'], c['oracle'], c['oracle'] - c['actual'],
            c['oracle_tp'], c['oracle_tp'] + c['oracle_fp']))
    lines += ['', '| 总体 | — | %.4f | %.4f | %.4f | — | — |' % (m_actual, m_oracle, m_oracle - m_actual), '']
    md = os.path.join(BASE, 'report', 'phase0_ap_oracle_%s.md' % args.tag)
    open(md, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('OK', md)
    print('actual %.4f oracle %.4f' % (m_actual, m_oracle))


if __name__ == '__main__':
    main()
