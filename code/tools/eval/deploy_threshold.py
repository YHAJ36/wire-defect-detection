"""
部署工作点表（工业巡检口径）

为什么需要这个工具：
    mAP50 / mAP50-95 是阈值无关的整体能力指标，无法回答巡检现场真正要决策的问题：
      - 把 conf 调到 X，能保住多少召回？漏掉多少真缺陷？
      - 代价是每张图平均多报几个框（= 巡检员要人工复查几次）？
    本项目此前报告 R=85.17%，那是「F1 最大点」的召回，不是巡检该用的工作点。
    巡检的代价严重不对称：漏一个断股 = 断线事故；多报一个 = 多看一秒。

口径：
    预测一次 conf=0.001 / NMS 0.5（与 eval_nms05 一致），然后**离线按阈值切**，
    这样一条 PR 曲线只跑一次推理。
    匹配：类内、按 conf 降序贪心、IoU>=0.5、每个 GT 只被占用一次（与 COCO 一致）。

用法：
    python code/tools/eval/deploy_threshold.py --run nf_v11_896_s42 --split test --device cpu
    python code/tools/eval/deploy_threshold.py --weights <path.pt> --name my_model --split test
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('D:/2/dataset_final_bbox')
RUNS = Path('D:/2/exp_upscale/runs')
OUT = Path('D:/2/report/_p2_diag')
NAMES = ['break', 'thunderbolt', 'wear']

CONF_FLOOR = 0.001   # 预测下限，与 mAP 口径一致
NMS_IOU = 0.5


def iou_matrix(a, b):
    """a:(n,4) b:(m,4) xyxy -> (n,m)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0][:, None], a[:, 1][:, None], a[:, 2][:, None], a[:, 3][:, None]
    bx1, by1, bx2, by2 = b[:, 0][None], b[:, 1][None], b[:, 2][None], b[:, 3][None]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    aa = np.clip((ax2 - ax1) * (ay2 - ay1), 1e-9, None)
    bb = np.clip((bx2 - bx1) * (by2 - by1), 1e-9, None)
    return inter / (aa + bb - inter)


def load_gt(split):
    """返回 [(img_path, np.array(n,5) [cls,x1,y1,x2,y2])]"""
    from PIL import Image
    items = []
    for p in sorted((ROOT / 'images' / split).glob('*.jpg')):
        w, h = Image.open(p).size
        f = ROOT / 'labels' / split / (p.stem + '.txt')
        rows = []
        if f.exists():
            for ln in f.read_text().strip().splitlines():
                s = ln.split()
                if len(s) < 5:
                    continue
                c = int(float(s[0]))
                cx, cy, bw, bh = map(float, s[1:5])
                rows.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                             (cx + bw / 2) * w, (cy + bh / 2) * h])
        items.append((p, np.array(rows, np.float32) if rows else np.zeros((0, 5), np.float32)))
    return items


def predict(weights, items, device, imgsz):
    """一次性推理，返回每张图的 (boxes[n,4], cls[n], conf[n])；按小批切片避免 OOM"""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    out = []
    B = 4
    for i in range(0, len(items), B):
        chunk = [str(p) for p, _ in items[i:i + B]]
        res = model.predict(chunk, imgsz=imgsz, conf=CONF_FLOOR, iou=NMS_IOU,
                            device=device, verbose=False)
        for r in res:
            if r.boxes is None or len(r.boxes) == 0:
                out.append((np.zeros((0, 4), np.float32), np.zeros(0, int), np.zeros(0, np.float32)))
            else:
                out.append((r.boxes.xyxy.cpu().numpy().astype(np.float32),
                            r.boxes.cls.cpu().numpy().astype(int),
                            r.boxes.conf.cpu().numpy().astype(np.float32)))
    return out


def match_at(items, preds, thr):
    """
    在给定 conf 阈值下统计 TP/FP/FN（类内，IoU 0.5，每个 GT 只占用一次）。
    返回逐类计数器与逐图 FP 数。
    """
    stat = defaultdict(lambda: dict(tp=0, fp=0, fn=0))
    fp_per_img = []
    for (path, gt), (b, c, s) in zip(items, preds):
        keep = s >= thr
        b, c = b[keep], c[keep]
        # 按 conf 降序（predict 已按 conf 排序，但切片后重排一次保险）
        order = np.argsort(-s[keep]) if keep.any() else np.array([], int)
        b, c = b[order], c[order]
        used = np.zeros(len(gt), bool)
        n_fp_img = 0
        for j in range(len(b)):
            cls_j = int(c[j])
            cand = np.where((gt[:, 0] == cls_j) & (~used))[0] if len(gt) else np.array([], int)
            hit = False
            if len(cand):
                ious = iou_matrix(b[j:j + 1], gt[cand, 1:5])[0]
                k = int(np.argmax(ious))
                if ious[k] >= 0.5:
                    used[cand[k]] = True
                    hit = True
            if hit:
                stat[NAMES[cls_j]]['tp'] += 1
            else:
                stat[NAMES[cls_j]]['fp'] += 1
                n_fp_img += 1
        # 未被匹配的 GT = 漏检
        for gi in range(len(gt)):
            if not used[gi]:
                stat[NAMES[int(gt[gi, 0])]]['fn'] += 1
        fp_per_img.append(n_fp_img)
    return stat, np.array(fp_per_img, np.float32)


def pr_at(stat):
    tp = sum(v['tp'] for v in stat.values())
    fp = sum(v['fp'] for v in stat.values())
    fn = sum(v['fn'] for v in stat.values())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, tp, fp, fn


def per_class_pr(stat):
    out = {}
    for nm in NAMES:
        v = stat[nm]
        out[nm] = (v['tp'] / (v['tp'] + v['fp']) if v['tp'] + v['fp'] else float('nan'),
                   v['tp'] / (v['tp'] + v['fn']) if v['tp'] + v['fn'] else float('nan'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='nf_v11_896_s42')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--name', default=None, help='输出名，默认用 run 名')
    ap.add_argument('--split', default='test')
    ap.add_argument('--imgsz', type=int, default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    w = Path(args.weights) if args.weights else RUNS / args.run / 'weights' / 'best.pt'
    name = args.name or args.run
    imgsz = args.imgsz
    if imgsz is None:
        import re
        m = re.search(r'_(640|896|1024)(?:_|$)', name)
        imgsz = int(m.group(1)) if m else 896

    items = load_gt(args.split)
    n_gt = sum(len(g) for _, g in items)
    print(f'=== {name} / {args.split} / imgsz={imgsz} / conf 下限 {CONF_FLOOR} / NMS {NMS_IOU} ===')
    print(f'{len(items)} 张图 / {n_gt} 个 GT\n')

    preds = predict(w, items, args.device, imgsz)

    # 阈值网格：从 0.01 到 0.90
    grid = sorted(set([round(x, 3) for x in np.arange(0.01, 0.91, 0.01)] +
                      [0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]))

    rows = []
    for thr in grid:
        stat, fpimg = match_at(items, preds, thr)
        p, r, f1, tp, fp, fn = pr_at(stat)
        pc = per_class_pr(stat)
        rows.append(dict(conf=thr, P=p, R=r, F1=f1, TP=tp, FP=fp, FN=fn,
                         fp_per_img=float(fpimg.mean()),
                         img_with_fp=float((fpimg > 0).mean()),
                         per_class={k: {'P': v[0], 'R': v[1]} for k, v in pc.items()}))

    # 关键工作点
    f1_best = max(rows, key=lambda x: x['F1'])
    print('【关键工作点】')
    print(f'{"工作点":22s} {"conf":>6s} {"P":>7s} {"R":>7s} {"F1":>7s} {"漏检":>5s} {"每图误报":>8s}')
    def line(tag, row):
        print(f'{tag:22s} {row["conf"]:6.3f} {row["P"]:7.4f} {row["R"]:7.4f} '
              f'{row["F1"]:7.4f} {row["FN"]:5d} {row["fp_per_img"]:8.2f}')
    line('F1 最大（现报告口径）', f1_best)
    for target in (0.98, 0.95, 0.92, 0.90, 0.85):
        cand = [r for r in rows if r['R'] >= target]
        if cand:
            best = max(cand, key=lambda x: x['P'])
            line(f'召回>={target:.0%} 时 P 最高', best)
    # 每图误报 <= 1 / <= 2 的工作点
    for cap in (0.5, 1.0, 2.0, 3.0):
        cand = [r for r in rows if r['fp_per_img'] <= cap]
        if cand:
            best = max(cand, key=lambda x: x['R'], )
            line(f'每图误报<={cap:g} 时 R 最高', best)

    print(f'\n【逐类 P/R（F1 最大点 conf={f1_best["conf"]:.2f}）】')
    for nm in NAMES:
        v = f1_best['per_class'][nm]
        print(f'  {nm:12s} P={v["P"]:.4f}  R={v["R"]:.4f}')

    print(f'\n【阈值扫描全表】')
    print(f'{"conf":>6s} {"P":>7s} {"R":>7s} {"F1":>7s} {"漏检":>5s} {"每图报框/误报":>14s}  逐类R')
    for r in rows:
        if r['conf'] not in (0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            continue
        rs = ' '.join(f'{r["per_class"][n]["R"]:.3f}' for n in NAMES)
        print(f'{r["conf"]:6.3f} {r["P"]:7.4f} {r["R"]:7.4f} {r["F1"]:7.4f} '
              f'{r["FN"]:5d} {r["fp_per_img"]:8.2f}      {rs}')

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f'deploy_{name}_{args.split}.json'
    out.write_text(json.dumps(dict(name=name, split=args.split, imgsz=imgsz,
                                   n_img=len(items), n_gt=n_gt,
                                   f1_best_conf=f1_best['conf'], rows=rows),
                              ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n保存 -> {out}')


if __name__ == '__main__':
    main()
