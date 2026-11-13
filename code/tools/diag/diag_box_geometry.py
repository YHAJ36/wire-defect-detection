# -*- coding: utf-8 -*-
"""框回归几何分解：把定位误差拆成"系统性偏差"和"随机抖动"。

动机
----
地板 test mAP50 = 92.12%，mAP50-95 = 54.23%。38 pt 的坑全在定位上。
但此前所有诊断只回答"漏没漏"（召回 95.76%），没回答"框歪在哪"。
本工具对 TP 框做几何分解，输出：

  d_cx / d_cy  = 中心偏移（以 GT 宽/高为单位）
  r_w / r_h    = 预测宽高 / GT 宽高（1.0 为无偏）
  rho          = 长宽比之比，(w_p/h_p)/(w_g/h_g)
  IoU          = 匹配质量

判读
----
  median(r_w) 显著 < 1  → 框系统性偏窄（标注紧、模型回缩）
  median(r_w) 显著 > 1  → 框系统性偏宽
  |median(d_cx)| 大      → 中心系统性偏移
  std 大而 median≈1     → 纯随机抖动，改结构无用，只能加数据/加预算
  按 IoU 分档对比        → 找 IoU 0.5~0.75 那批框错在哪一维

用法
----
  D:/yolo-env/Scripts/python.exe code/tools/diag/diag_box_geometry.py \
      --weights D:/2/exp_upscale/runs/nf_v11_896_s42/weights/best.pt \
      --split test --device cpu --threads 4
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('D:/2/dataset_final_bbox')
NAMES = ['break', 'thunderbolt', 'wear']


def load_gt(lbl_path):
    """返回 Nx5: cls, cx, cy, w, h（归一化）。"""
    if not lbl_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for ln in lbl_path.read_text(encoding='utf-8').strip().splitlines():
        p = ln.split()
        if len(p) >= 5:
            rows.append([float(x) for x in p[:5]])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def xywh2xyxy(b):
    """b: Nx4 (cx,cy,w,h) → Nx4 (x1,y1,x2,y2)，像素。"""
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


def iou_matrix(a, b):
    """a: Nx4 xyxy, b: Mx4 xyxy → NxM IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0).astype(np.float32)


def greedy_match(gt_xyxy, gt_cls, pr_xyxy, pr_cls, pr_conf, thr):
    """类内贪心匹配，返回 [(gi, pi, iou)]。"""
    pairs = []
    used = set()
    order = np.argsort(-pr_conf)
    for pi in order:
        cand = [gi for gi in range(len(gt_xyxy))
                if gi not in used and gt_cls[gi] == pr_cls[pi]]
        if not cand:
            continue
        ious = iou_matrix(gt_xyxy[cand], pr_xyxy[pi:pi + 1]).ravel()
        k = int(np.argmax(ious))
        if ious[k] >= thr:
            gi = cand[k]
            pairs.append((gi, int(pi), float(ious[k])))
            used.add(gi)
    return pairs


def summarize(vals, name):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return f'  {name:6s}  n=0'
    return (f'  {name:6s}  n={len(v):4d}  median={np.median(v):+7.4f}  '
            f'mean={v.mean():+7.4f}  std={v.std():7.4f}  '
            f'p10={np.percentile(v, 10):+7.4f}  p90={np.percentile(v, 90):+7.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--imgsz', type=int, default=896)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--nms', type=float, default=0.5)
    ap.add_argument('--match', type=float, default=0.5)
    ap.add_argument('--chunk', type=int, default=8)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default='D:/2/report/_p2_diag/box_geometry.json')
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    from ultralytics import YOLO

    img_dir = ROOT / 'images' / args.split
    lbl_dir = ROOT / 'labels' / args.split
    imgs = sorted(img_dir.glob('*.jpg'))
    if not imgs:
        raise SystemExit(f'没有图：{img_dir}')

    model = YOLO(args.weights)
    rec = []          # 每行一个匹配对
    n_gt = 0
    n_det = 0

    for s in range(0, len(imgs), args.chunk):
        batch = imgs[s:s + args.chunk]
        res = model.predict([str(p) for p in batch], imgsz=args.imgsz, conf=args.conf,
                            iou=args.nms, device=args.device, verbose=False)
        for p, r in zip(batch, res):
            W, H = Image.open(p).size
            gt = load_gt(lbl_dir / f'{p.stem}.txt')
            n_gt += len(gt)
            if len(gt) == 0:
                continue
            gt_xyxy = xywh2xyxy(np.stack([gt[:, 1] * W, gt[:, 2] * H,
                                          gt[:, 3] * W, gt[:, 4] * H], 1))
            gt_cls = gt[:, 0].astype(int)

            if r.boxes is None or len(r.boxes) == 0:
                continue
            pr = r.boxes.xyxy.cpu().numpy()
            pr_cls = r.boxes.cls.cpu().numpy().astype(int)
            pr_conf = r.boxes.conf.cpu().numpy()
            n_det += len(pr)

            for gi, pi, iou in greedy_match(gt_xyxy, gt_cls, pr, pr_cls, pr_conf, args.match):
                gw, gh = gt_xyxy[gi, 2] - gt_xyxy[gi, 0], gt_xyxy[gi, 3] - gt_xyxy[gi, 1]
                pw, ph = pr[pi, 2] - pr[pi, 0], pr[pi, 3] - pr[pi, 1]
                if gw <= 0 or gh <= 0:
                    continue
                gc = (gt_xyxy[gi, 0] + gw / 2, gt_xyxy[gi, 1] + gh / 2)
                pc = (pr[pi, 0] + pw / 2, pr[pi, 1] + ph / 2)
                rec.append(dict(
                    cls=int(gt_cls[gi]), iou=float(iou), side=float(np.sqrt(gw * gh)),
                    d_cx=float((pc[0] - gc[0]) / gw), d_cy=float((pc[1] - gc[1]) / gh),
                    r_w=float(pw / gw), r_h=float(ph / gh),
                    rho=float((pw / ph) / (gw / gh)) if ph > 0 else float('nan'),
                ))

    if not rec:
        raise SystemExit('没有任何匹配对，检查 --match / --conf')

    A = {k: np.array([r[k] for r in rec]) for k in rec[0]}
    print(f'\n=== 匹配统计（conf={args.conf}, NMS={args.nms}, match IoU>={args.match}）===')
    print(f'  GT 框 {n_gt}  预测框 {n_det}  匹配上 {len(rec)}  '
          f'（召回 {100 * len(rec) / max(n_gt, 1):.2f}%）')
    print(f'  IoU 分布：median={np.median(A["iou"]):.3f}  '
          f'p10={np.percentile(A["iou"], 10):.3f}  p90={np.percentile(A["iou"], 90):.3f}')

    print('\n=== 全局几何分解 ===')
    for k, lab in [('d_cx', 'd_cx'), ('d_cy', 'd_cy'), ('r_w', 'r_w'),
                   ('r_h', 'r_h'), ('rho', 'rho')]:
        print(summarize(A[k], lab))

    print('\n=== 按类别 ===')
    for c in range(3):
        m = A['cls'] == c
        if m.sum() == 0:
            print(f'  [{NAMES[c]}] 无匹配')
            continue
        print(f'  [{NAMES[c]}] n={m.sum()}  IoU中位={np.median(A["iou"][m]):.3f}  '
              f'r_w中位={np.median(A["r_w"][m]):.4f}  r_h中位={np.median(A["r_h"][m]):.4f}  '
              f'rho中位={np.median(A["rho"][m]):.4f}  '
              f'|d_cx|中位={np.median(np.abs(A["d_cx"][m])):.4f}')

    print('\n=== 按 GT 尺寸分档（px，sqrt(w*h)）===')
    for lo, hi in [(0, 16), (16, 24), (24, 40), (40, 80), (80, 1e9)]:
        m = (A['side'] >= lo) & (A['side'] < hi)
        if m.sum():
            print(f'  {lo:3.0f}-{hi if hi < 1e8 else "inf":>4}  n={m.sum():4d}  '
                  f'IoU中位={np.median(A["iou"][m]):.3f}  '
                  f'r_w中位={np.median(A["r_w"][m]):.4f}  r_h中位={np.median(A["r_h"][m]):.4f}')

    print('\n=== 按 IoU 分档（找 0.5~0.75 那批错在哪一维）===')
    for lo, hi in [(0.5, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 1.01)]:
        m = (A['iou'] >= lo) & (A['iou'] < hi)
        if m.sum():
            print(f'  IoU {lo:.2f}-{hi:.2f}  n={m.sum():4d}  '
                  f'r_w中位={np.median(A["r_w"][m]):.4f}  r_h中位={np.median(A["r_h"][m]):.4f}  '
                  f'rho中位={np.median(A["rho"][m]):.4f}  '
                  f'|d_cx|中位={np.median(np.abs(A["d_cx"][m])):.4f}  '
                  f'|d_cy|中位={np.median(np.abs(A["d_cy"][m])):.4f}')

    # 系统性偏差 vs 随机抖动：把每维按中位数去偏后，能拿回多少 IoU
    gt_wh = None
    dev = np.stack([A['r_w'] - np.median(A['r_w']), A['r_h'] - np.median(A['r_h'])], 1)
    print('\n=== 偏差构成 ===')
    for k, lab in [('r_w', '宽'), ('r_h', '高')]:
        bias = abs(np.median(A[k]) - 1) * 100
        sd = A[k].std() * 100
        print(f'  {lab}: 系统性偏差 {bias:5.2f}%   随机抖动(std) {sd:5.2f}%   '
              f'{"偏差主导" if bias > sd else "抖动主导"}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'weights': args.weights, 'split': args.split,
                               'n_gt': n_gt, 'n_det': n_det, 'n_match': len(rec),
                               'rows': rec}, ensure_ascii=False), encoding='utf-8')
    print(f'\n原始记录 → {out}')


if __name__ == '__main__':
    main()
