# -*- coding: utf-8 -*-
"""Phase 0: dataset and label audit (read-only).

Outputs:
    report/_phase0/dataset_audit.json
    report/phase0_dataset_audit.md
"""
import json
import math
import os
import statistics
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(BASE, 'dataset_final_bbox')
OUT = os.path.join(BASE, 'report', '_phase0')
os.makedirs(OUT, exist_ok=True)

NAMES = ['break', 'thunderbolt', 'wear']
SPLITS = ['train', 'val', 'test']
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def images_of(split):
    d = os.path.join(DS, 'images', split)
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def labels_of(split):
    d = os.path.join(DS, 'labels', split)
    return sorted(f for f in os.listdir(d) if f.endswith('.txt'))


def px_stats(area_norm, imgsz):
    side = math.sqrt(max(area_norm, 0) * imgsz * imgsz)
    return side


def bucket(area_norm, imgsz):
    side = px_stats(area_norm, imgsz)
    a = side * side
    if a < 32 * 32:
        return 'small(<32px)'
    if a < 96 * 96:
        return 'medium(32-96px)'
    return 'large(>96px)'


def pct(vals, q):
    if not vals:
        return None
    vals = sorted(vals)
    k = (len(vals) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - k) + vals[hi] * (k - lo)


report = {'dataset': DS, 'names': NAMES, 'splits': {}}
md = ['# Phase 0 数据集与标签审计', '',
      f'数据根目录：`{DS}`；类别：{", ".join(NAMES)}。', '']

for split in SPLITS:
    imgs = images_of(split)
    labs = labels_of(split)
    img_stems = {os.path.splitext(f)[0] for f in imgs}
    lab_stems = {os.path.splitext(f)[0] for f in labs}
    missing_lab = sorted(img_stems - lab_stems)
    orphan_lab = sorted(lab_stems - img_stems)
    empty_lab = []
    invalid = []
    dup_lines = []
    counts = Counter()
    areas = defaultdict(list)
    widths = defaultdict(list)
    heights = defaultdict(list)
    aspects = defaultdict(list)
    buckets896 = defaultdict(Counter)
    buckets1024 = defaultdict(Counter)
    per_img = defaultdict(Counter)
    edge_boxes = 0
    tiny_boxes = 0
    total_boxes = 0

    for f in labs:
        path = os.path.join(DS, 'labels', split, f)
        lines = [ln.strip() for ln in open(path, encoding='utf-8') if ln.strip()]
        if not lines:
            empty_lab.append(f)
            continue
        seen = set()
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                invalid.append((f, ln, 'field_count'))
                continue
            try:
                c = int(float(parts[0]))
                x, y, w, h = [float(v) for v in parts[1:]]
            except ValueError:
                invalid.append((f, ln, 'parse'))
                continue
            if c < 0 or c >= len(NAMES):
                invalid.append((f, ln, 'class'))
                continue
            if not (0 <= x <= 1 and 0 <= y <= 1) or w <= 0 or h <= 0:
                invalid.append((f, ln, 'range'))
                continue
            if w > 1 or h > 1:
                invalid.append((f, ln, 'size_gt_1'))
                continue
            key = (c, round(x, 6), round(y, 6), round(w, 6), round(h, 6))
            if key in seen:
                dup_lines.append((f, ln))
            seen.add(key)
            counts[NAMES[c]] += 1
            per_img[NAMES[c]][os.path.splitext(f)[0]] += 1
            a = w * h
            areas[NAMES[c]].append(a)
            widths[NAMES[c]].append(w)
            heights[NAMES[c]].append(h)
            aspects[NAMES[c]].append(w / h if h else 0)
            buckets896[NAMES[c]][bucket(a, 896)] += 1
            buckets1024[NAMES[c]][bucket(a, 1024)] += 1
            total_boxes += 1
            if x - w / 2 < 0.002 or x + w / 2 > 0.998 or y - h / 2 < 0.002 or y + h / 2 > 0.998:
                edge_boxes += 1
            if px_stats(a, 896) < 8:
                tiny_boxes += 1

    split_rep = {
        'images': len(imgs), 'labels': len(labs),
        'missing_labels': len(missing_lab), 'orphan_labels': len(orphan_lab),
        'empty_labels': len(empty_lab), 'invalid': len(invalid), 'duplicate_lines': len(dup_lines),
        'instances': total_boxes, 'class_instances': dict(counts),
        'edge_boxes': edge_boxes, 'boxes_lt_8px_at_896': tiny_boxes,
        'classes': {},
    }
    for name in NAMES:
        a = areas[name]
        split_rep['classes'][name] = {
            'instances': len(a),
            'images': len(per_img[name]),
            'area_median': statistics.median(a) if a else None,
            'area_mean': statistics.mean(a) if a else None,
            'area_p05': pct(a, 0.05), 'area_p95': pct(a, 0.95),
            'side_px_896_median': px_stats(statistics.median(a), 896) if a else None,
            'side_px_896_p05': px_stats(pct(a, 0.05), 896) if a else None,
            'side_px_896_p95': px_stats(pct(a, 0.95), 896) if a else None,
            'aspect_median': statistics.median(aspects[name]) if a else None,
            'buckets_896': dict(buckets896[name]),
            'buckets_1024': dict(buckets1024[name]),
        }
    report['splits'][split] = split_rep

    md += [f'## {split}', '',
           f'- 图像 {len(imgs)} 张，标签文件 {len(labs)} 个，标注框 {total_boxes} 个；'
           f'缺标签 {len(missing_lab)}，孤立标签 {len(orphan_lab)}，空标签 {len(empty_lab)}，'
           f'非法行 {len(invalid)}，重复行 {len(dup_lines)}。',
           f'- 贴边框 {edge_boxes} 个；896 下短边 < 8px 的框 {tiny_boxes} 个。',
           '', '| 类别 | 框数 | 含该类图像 | 面积中位数 | 896 中位边长 | 896 P05–P95 边长 | 宽高比中位 | small / medium / large (896) |',
           '|---|---|---|---|---|---|---|---|']
    for name in NAMES:
        c = split_rep['classes'][name]
        b = c['buckets_896']
        md.append('| %s | %d | %d | %.6f | %.1f px | %.1f–%.1f px | %.2f | %d / %d / %d |' % (
            name, c['instances'], c['images'], c['area_median'] or 0,
            c['side_px_896_median'] or 0, c['side_px_896_p05'] or 0, c['side_px_896_p95'] or 0,
            c['aspect_median'] or 0, b.get('small(<32px)', 0), b.get('medium(32-96px)', 0),
            b.get('large(>96px)', 0)))
    md += ['']
    if invalid:
        md += ['非法标注示例：', '']
        for x in invalid[:8]:
            md.append(f'- `{x[0]}`: `{x[1]}` ({x[2]})')
        md.append('')
    if missing_lab:
        md += [f'缺标签图像（前 8）：{", ".join(missing_lab[:8])}', '']
    if orphan_lab:
        md += [f'孤立标签（前 8）：{", ".join(orphan_lab[:8])}', '']

with open(os.path.join(OUT, 'dataset_audit.json'), 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
with open(os.path.join(BASE, 'report', 'phase0_dataset_audit.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(md) + '\n')
print('OK', os.path.join(OUT, 'dataset_audit.json'))
print('OK', os.path.join(BASE, 'report', 'phase0_dataset_audit.md'))
