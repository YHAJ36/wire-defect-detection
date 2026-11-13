# -*- coding: utf-8 -*-
"""时序一致性后处理：跨帧平滑检测框，压制抖动，再算 AP50。

依据（A.9.47f）
    GT 自身相邻帧抖动 sd = 0.138（gap=1），模型 r_w sd = 0.1465 —— 模型抖动已
    几乎等于标注噪声地板。A.9.41 判定「抖动主导（14.6%）而非系统偏差（2%）」，
    WBF 多模型集成因此能让 s95 +2.13 却不动 mAP50。
    抖动是零均值噪声 ⇒ **跨帧平均理论上能把方差降到 1/K**，且不引入偏差。

    这是唯一「零训练、不改数据、机理与诊断直接对应」的方向。

做法
    1. 按文件名编号排序（firc_xianlan_<n>），逐图检测（conf 低阈值、NMS 0.5）。
    2. 跨帧关联：同一类别、与上帧某轨迹 IoU 最大且 >= 阈值 → 接上；否则新开轨迹。
    3. 对每个轨迹保留最近 K 帧的框；输出框 = 最近 K 帧的（可选按 conf 加权）平均。
       置信度保留当前帧原值（平滑框若用平均 conf 会人为抬高低分框的排序）。
    4. 平滑前后各自算 AP50（101 点插值，同类匹配）。

    对照组：不平滑（K=0）必须复现原始 AP50，作为口径自检。

用法
    D:/yolo-env/Scripts/python.exe code/tools/diag/diag_temporal_smooth.py \
        --ckpt runs/nf_v11_896_s42/weights/last.pt --imgsz 896
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from ultralytics import YOLO  # noqa: E402

ROOT = 'D:/2/dataset_final_bbox/_valtest'
NAMES = ['break', 'thunderbolt', 'wear']


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match_greedy(pb, pc, gb, thr=0.5):
    n = len(pb)
    out = np.full(n, -1, int)
    if n == 0 or len(gb) == 0:
        return out
    M = iou_mat(pb, gb)
    used = np.zeros(len(gb), bool)
    for i in np.argsort(-pc):
        cand = np.where(used, -1.0, M[i])
        j = int(np.argmax(cand))
        if cand[j] >= thr:
            out[i] = j
            used[j] = True
    return out


def compute_ap(tp, conf, n_gt):
    if n_gt == 0:
        return float('nan')
    o = np.argsort(-conf)
    tp = tp[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    rec, prec = ctp / n_gt, ctp / (ctp + cfp + 1e-9)
    for i in range(len(prec) - 1, 0, -1):
        prec[i - 1] = max(prec[i - 1], prec[i])
    xs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, xs, side='left')
    return float(np.mean(np.where(idx < len(prec), prec[np.clip(idx, 0, len(prec) - 1)], 0)))


def load_gt(stem):
    lab = f'{ROOT}/labels/{stem}.txt'
    if not os.path.exists(lab):
        return np.zeros(0, int), np.zeros((0, 4))
    from PIL import Image
    with Image.open(f'{ROOT}/images/{stem}.jpg') as im:
        W, H = im.size
    rows = [l.split() for l in open(lab) if l.strip()]
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.array([[float(x) for x in t[1:5]] for t in rows])
    c = np.array([int(t[0]) for t in rows])
    cx, cy, w, h = a[:, 0] * W, a[:, 1] * H, a[:, 2] * W, a[:, 3] * H
    return c, np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def ap50_of(records):
    """records: list of (stem, cls, conf, box, is_match_ignored) -> 用 GT 重算 AP50"""
    per = {}
    for c in range(3):
        conf, tp, n_gt = [], [], 0
        for stem in sorted({r[0] for r in records}):
            gc, _ = load_gt(stem)
            n_gt += int((gc == c).sum())
        # 逐图匹配（用记录里该图的框）
        by_img = {}
        for r in records:
            by_img.setdefault(r[0], []).append(r)
        for stem, rs in by_img.items():
            gc, gb = load_gt(stem)
            rs_c = [x for x in rs if x[1] == c]
            if not rs_c:
                continue
            pb = np.array([x[3] for x in rs_c], float)
            pc = np.array([x[2] for x in rs_c], float)
            gi = np.where(gc == c)[0]
            m = match_greedy(pb, pc, gb[gi], 0.5) if len(gi) else np.full(len(pb), -1)
            for k in range(len(pb)):
                conf.append(pc[k])
                tp.append(m[k] >= 0)
        if conf:
            per[NAMES[c]] = compute_ap(np.array(tp), np.array(conf), n_gt)
        else:
            per[NAMES[c]] = 0.0
    vals = [v for v in per.values() if not np.isnan(v)]
    return float(np.mean(vals)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--imgsz', type=int, default=896)
    ap.add_argument('--iou', type=float, default=0.5, help='NMS')
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--link-iou', type=float, default=0.25, help='跨帧关联 IoU 下限')
    ap.add_argument('--device', default='0')
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()

    ck = Path(args.ckpt)
    if not ck.is_absolute():
        ck = Path('D:/2/exp_upscale/runs') / args.ckpt
    model = YOLO(str(ck))

    # 只取 test 的 179 图，按编号排序（val 也混在 _valtest 里，用原 test 列表）
    stems = [os.path.splitext(os.path.basename(p))[0]
             for p in glob.glob('D:/2/dataset_final_bbox/images/test/*.jpg')]
    def num(s):
        m = re.search(r'(\d+)$', s)
        return int(m.group(1)) if m else 10 ** 9
    stems.sort(key=num)
    print(f'test 图 {len(stems)}，编号 {num(stems[0])}–{num(stems[-1])}')

    # 逐图检测
    dets = {}
    for s in range(0, len(stems), 8):
        chunk = stems[s:s + 8]
        res = model.predict([f'D:/2/dataset_final_bbox/images/test/{c}.jpg' for c in chunk],
                            imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            max_det=300, device=args.device, verbose=False)
        for st, r in zip(chunk, res):
            b = r.boxes
            if b is None or len(b) == 0:
                dets[st] = (np.zeros(0, 4), np.zeros(0), np.zeros(0, int))
            else:
                dets[st] = (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                            b.cls.cpu().numpy().astype(int))

    def run(K, mode='mean', alpha=0.5):
        """K=0 表示不平滑。返回 records"""
        tracks = []   # dict(cls, hist[list of box], last_box)
        records = []
        for st in stems:
            pb, pc, pcls = dets[st]
            out_boxes = np.zeros_like(pb)
            used_tr = set()
            for i in range(len(pb)):
                best, bi = 0.0, -1
                if len(tracks):
                    cand = [(t, iou_mat(pb[i:i + 1], np.array([t['last']]))[0, 0])
                            for t in tracks if t['cls'] == pcls[i] and id(t) not in used_tr]
                    if cand:
                        best, bi_t = max(((v, k) for k, (t, v) in enumerate(cand)), default=(0.0, -1))
                        if best >= args.link_iou:
                            bi = cand[bi_t][0]['id']
                if K == 0:
                    out_boxes[i] = pb[i]
                    continue
                if bi >= 0:
                    t = next(x for x in tracks if x['id'] == bi)
                    t['hist'].append(pb[i])
                    t['hist'] = t['hist'][-K:]
                    if mode == 'mean':
                        nb = np.mean(t['hist'], 0)
                    else:  # ema
                        nb = alpha * pb[i] + (1 - alpha) * t['last']
                    out_boxes[i] = nb
                    t['last'] = nb
                    used_tr.add(id(t))
                else:
                    t = dict(id=len(tracks) + 1000, cls=int(pcls[i]),
                             hist=[pb[i].copy()], last=pb[i].copy())
                    tracks.append(t)
                    used_tr.add(id(t))
                    out_boxes[i] = pb[i]
            for i in range(len(pb)):
                records.append((st, int(pcls[i]), float(pc[i]), out_boxes[i].tolist()))
        return records

    print('\n=== 时序平滑效果（test 179 图，NMS 0.5，conf 0.001）===')
    print(f'{"配置":<22}{"mAP50":>9}{"Δ":>8}  逐类')
    base = None
    res = {}
    for K, mode, al in [(0, 'mean', 0), (3, 'mean', 0), (5, 'mean', 0), (9, 'mean', 0),
                        (5, 'ema', 0.4), (5, 'ema', 0.6)]:
        rec = run(K, mode, al)
        m, per = ap50_of(rec)
        name = '不平滑(自检)' if K == 0 else (f'K={K} mean' if mode == 'mean' else f'ema a={al} (≈K{int(3/al)})')
        if base is None:
            base = m
        d = '' if K == 0 else f'{(m - base) * 100:+.2f}'
        print(f'{name:<22}{m:>9.4f}{d:>8}  ' + '  '.join(f'{k}={v:.3f}' for k, v in per.items()))
        res[name] = dict(ap50=m, per_class=per)

    p = Path(f'D:/2/report/_p2_diag/temporal_smooth_{args.tag}.json')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(ckpt=str(ck), link_iou=args.link_iou, res=res),
                            ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n→ {p}')


if __name__ == '__main__':
    main()
