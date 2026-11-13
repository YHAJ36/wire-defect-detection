# -*- coding: utf-8 -*-
"""类别错误结构诊断：三类的错到底是漏检、错类、还是误检。

背景
    A.9.33 之后 P2 方向判负。三类 AP50 里 break 是短板（0.8678 / thunderbolt 0.9698 /
    wear 0.9259），三类等权平均后 break 一个类就压了整体约 3.4 pt。要动 break，先得知道
    它的错误长什么样 —— 漏检？被认成别的类？还是框位置不准？

口径
    test / 896 / conf 0.25 / NMS IoU 0.5 / class-wise / max_det 300，与 A.9.32 逐尺度归因一致。
    匹配方式：按置信度降序遍历预测框，先找同类且 IoU>0.5 的未匹配 GT（=正确）；
    找不到同类就找任意类 IoU>0.5 的未匹配 GT（=错类，记下被认成谁）；再找不到就是误检。

用法：
    python code/tools/diag/diag_class_error.py [run ...]
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, 'D:/2/code/yolov11')
from ultralytics import YOLO  # noqa: E402

DS = Path('D:/2/dataset_final_bbox')
RUNS = Path('D:/2/exp_upscale/runs')
IMGSZ, CONF, NMS_IOU, MAX_DET = 896, 0.25, 0.5, 300
# 注意：CONF=0.25 是**部署口径**，不是 mAP 口径（ultralytics val 用 0.001）。
# 两者结论会差很远：conf 0.25 下 wear 有 11 个"完全无框"，conf 0.001 下 wear 召回 100%。
# 命令行可用 --conf 覆盖。
# 命中判定的阈值固定 0.5，**不能**跟着 NMS_IOU 走。
# 第一版把两者混用，扫描 NMS 时匹配阈值也一起涨到 0.9，
# 结果召回随 NMS 放宽单调下降（82%→10%），完全是假的。
MATCH_IOU = 0.5
NAMES = ['break', 'thunderbolt', 'wear']
BUCKETS = [(0, 16), (16, 32), (32, 64), (64, 96), (96, 1e9)]


def bucket_of(side):
    for lo, hi in BUCKETS:
        if lo <= side < hi:
            return f'{lo}-{hi if hi < 1e9 else "+"}'
    return '?'


def iou_matrix(a, b):
    """a: (N,4) b: (M,4) 都是 xyxy。返回 (N,M)。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None] - inter, 1e-9, None)


def load_split(split):
    """返回 [(img_path, gts)]，gts 的 box 是原图像素坐标，side 是 letterbox 到 896 后的边长。"""
    out = []
    for lb in sorted((DS / 'labels' / split).glob('*.txt')):
        cand = list((DS / 'images' / split).glob(lb.stem + '.*'))
        if not cand:
            continue
        im = cand[0]
        w0, h0 = Image.open(im).size
        r = min(IMGSZ / h0, IMGSZ / w0)
        gts = []
        for line in lb.read_text().strip().splitlines():
            if not line.strip():
                continue
            p = line.split()
            c = int(float(p[0]))
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            x1, y1 = (cx - bw / 2) * w0, (cy - bh / 2) * h0
            x2, y2 = (cx + bw / 2) * w0, (cy + bh / 2) * h0
            gts.append(dict(cls=c, box=[x1, y1, x2, y2],
                            side=float(np.sqrt(max((bw * w0 * r) * (bh * h0 * r), 0)))))
        out.append((str(im), gts))
    return out


def analyse(run, items):
    ck = RUNS / run / 'weights' / 'best.pt'
    model = YOLO(str(ck))
    outcome = defaultdict(Counter)          # 真实类 -> {correct/wrong:xx/miss:xx}
    fp_by_cls = Counter()                   # 误检：预测类 -> 数量
    bucket_out = defaultdict(Counter)       # break 的尺寸桶 -> 结局
    imgs = [p for p, _ in items]
    # 不能把整份 list 交给 predict：ultralytics 会一次性预处理全部图片，
    # 179 张 896 输入直接爆显存（实测 10.24 GiB 已分配）。这里手动分批。
    results = []
    for i in range(0, len(imgs), 4):
        results.extend(model.predict(imgs[i:i + 4], imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                                     max_det=MAX_DET, agnostic_nms=False,
                                     verbose=False, stream=False))

    for (path, gts), r in zip(items, results):
        pb = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        pc = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(0, int)
        ps = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)

        if len(gts) == 0:
            for c in pc:
                fp_by_cls[NAMES[c]] += 1
            continue

        gb = np.array([g['box'] for g in gts], dtype=np.float32)
        gc = np.array([g['cls'] for g in gts], dtype=int)
        M = iou_matrix(pb, gb)                       # (P, G)
        gt_done = np.zeros(len(gts), bool)
        pred_done = np.zeros(len(pb), bool)

        for j in np.argsort(-ps):                    # 按置信度降序领框
            if pred_done[j]:
                continue
            # 1) 同类命中
            cand = np.where((M[j] > MATCH_IOU) & (~gt_done) & (gc == pc[j]))[0]
            if len(cand):
                gi = cand[np.argmax(M[j, cand])]
                outcome[NAMES[gc[gi]]]['correct'] += 1
                bucket_out[bucket_of(gts[gi]['side'])]['correct'] += 1
                gt_done[gi] = pred_done[j] = True
                continue
            # 2) 定位命中但类别错
            cand = np.where((M[j] > MATCH_IOU) & (~gt_done))[0]
            if len(cand):
                gi = cand[np.argmax(M[j, cand])]
                outcome[NAMES[gc[gi]]]['wrong:' + NAMES[pc[j]]] += 1
                bucket_out[bucket_of(gts[gi]['side'])]['wrong:' + NAMES[pc[j]]] += 1
                gt_done[gi] = pred_done[j] = True
                continue
            # 3) 误检
            fp_by_cls[NAMES[pc[j]]] += 1
            pred_done[j] = True

        for gi in np.where(~gt_done)[0]:
            outcome[NAMES[gc[gi]]]['miss'] += 1
            bucket_out[bucket_of(gts[gi]['side'])]['miss'] += 1

    print(f'\n===== {run}  (test, conf {CONF}, NMS {NMS_IOU})')
    print(f'{"类别":>12} {"GT":>5} {"正确":>6} {"错类":>6} {"漏检":>6} {"误检":>6} {"召回":>7} {"精确":>7}')
    for c in NAMES:
        o = outcome[c]
        n = sum(o.values())
        corr = o['correct']
        wrong = sum(v for k, v in o.items() if k.startswith('wrong:'))
        miss = o['miss']
        fp = fp_by_cls[c]
        rec = corr / max(n, 1)
        pre = corr / max(corr + wrong + fp, 1)
        print(f'{c:>12} {n:>5} {corr:>6} {wrong:>6} {miss:>6} {fp:>6} '
              f'{rec * 100:>6.1f}% {pre * 100:>6.1f}%')
        for k, v in o.items():
            if k.startswith('wrong:'):
                print(f'{"":>12}   └ 被认成 {k[6:]}: {v} 个 ({v / max(n, 1) * 100:.1f}%)')

    print(f'\n  break 按尺寸分桶（letterbox 到 {IMGSZ} 后的边长）：')
    print(f'{"边长":>10} {"总数":>5} {"正确":>6} {"错类":>6} {"漏检":>6} {"召回":>7}')
    for lo, hi in BUCKETS:
        k = f'{lo}-{hi if hi < 1e9 else "+"}'
        o = bucket_out.get(k)
        if not o:
            continue
        n = sum(o.values())
        corr = o['correct']
        wrong = sum(v for kk, v in o.items() if kk.startswith('wrong:'))
        miss = o['miss']
        print(f'{k:>10} {n:>5} {corr:>6} {wrong:>6} {miss:>6} {corr / max(n, 1) * 100:>6.1f}%')


def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--conf', type=float, default=None)
    ap.add_argument('--nms', type=float, default=None)
    return ap.parse_args()


def main():
    global CONF, NMS_IOU
    args = parse_cli()
    if args.conf is not None:
        CONF = args.conf
    if args.nms is not None:
        NMS_IOU = args.nms
    runs = args.runs or ['nf_v11_896_s42']
    items = load_split('test')
    print(f'test 共 {len(items)} 张，GT {sum(len(g) for _, g in items)} 个  '
          f'(conf={CONF} nms={NMS_IOU} match={MATCH_IOU})')
    for r in runs:
        analyse(r, items)


if __name__ == '__main__':
    main()
