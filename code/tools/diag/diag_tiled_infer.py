# -*- coding: utf-8 -*-
"""切片推理探针：不改训练、不改权重，只改推理时的切图策略。

要回答的问题
    A.9.34 把 break 的短板定位到「4 张密集图占一半漏检」。但进一步查图片原始尺寸发现：
    那 4 张全是 1920x1080，letterbox 到 896 时被缩小 2.14 倍；而 400x296 的小图反而被
    放大 2.24 倍。按缩放比分组的召回差 13.6 pt（放大组 break 90.0% vs 缩小组 76.4%）。
    因此怀疑真正的瓶颈是**大图被降采样、细节被抹掉**，而不是目标密度。

    本脚本把同一批权重按不同切图策略推理，比较召回：
      full  : 整图 letterbox 到 896（= 现有评估口径，对照）
      tile  : 按长边自适应切块，每块单独 letterbox 到 896，框映射回原图后做全局 NMS

    如果 tile 能把 break 召回从 76% 拉回 88% 以上，说明这是**纯推理侧**可拿的收益，
    不动训练、不动结构、不占额外显存（训练仍是 896 一键）。

口径
    conf 0.25 / 全局 NMS IoU 0.5（class-wise）/ 命中阈值 IoU 0.5 / 置信度降序一配一。
    tile 内部允许更宽松的 NMS(0.7) 以保留跨块候选，全局再收一次 0.5 —— full 模式也走
    同样的「内部 0.7 → 全局 0.5」流程，保证两者唯一差异是切不切块。

用法
    python code/tools/diag/diag_tiled_infer.py                       # 全部 test
    python code/tools/diag/diag_tiled_infer.py --only 151,139,150    # 只跑指定编号
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms

sys.path.insert(0, 'D:/2/code/yolov11')
from ultralytics import YOLO  # noqa: E402

DS = Path('D:/2/dataset_final_bbox')
RUNS = Path('D:/2/exp_upscale/runs')
IMGSZ = 896
CONF = 0.25
NMS_IOU = 0.5          # 全局 NMS
TILE_NMS_IOU = 0.7     # 块内 NMS（放宽，留给全局收）
MATCH_IOU = 0.5        # 命中判定，独立于 NMS 参数
MAX_DET = 300
BATCH = 4
NAMES = ['break', 'thunderbolt', 'wear']


def load_items(split='test', only=None):
    """返回 [(img_path, gts, w0, h0)]，gts 的 box 是**原图像素** xyxy。"""
    out = []
    for lb in sorted((DS / 'labels' / split).glob('*.txt')):
        cand = list((DS / 'images' / split).glob(lb.stem + '.*'))
        if not cand:
            continue
        im = cand[0]
        if only and not any(re.search(rf'_{n}$', lb.stem) for n in only):
            continue
        w0, h0 = Image.open(im).size
        gts = []
        for line in lb.read_text().strip().splitlines():
            if not line.strip():
                continue
            p = line.split()
            c = int(float(p[0]))
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            gts.append(dict(cls=c, box=[(cx - bw / 2) * w0, (cy - bh / 2) * h0,
                                        (cx + bw / 2) * w0, (cy + bh / 2) * h0]))
        out.append((im, gts, w0, h0))
    return out


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None] - inter, 1e-9, None)


def tile_grid(w, h, nx, ny, ov):
    """nx/ny 为 1 时返回整图；否则按 overlap 均分并夹到图内。"""
    if nx == 1 and ny == 1:
        return [(0, 0, w, h)]
    tw = w / (nx - (nx - 1) * ov)
    th = h / (ny - (ny - 1) * ov)
    tiles = []
    for iy in range(ny):
        for ix in range(nx):
            x0 = int(round(ix * tw * (1 - ov)))
            y0 = int(round(iy * th * (1 - ov)))
            x1 = min(w, int(round(x0 + tw)))
            y1 = min(h, int(round(y0 + th)))
            x0, y0 = max(0, x0), max(0, y0)
            if x1 - x0 >= 64 and y1 - y0 >= 64:
                tiles.append((x0, y0, x1, y1))
    return tiles


def infer(model, im, nx, ny, ov):
    """返回 (boxes_xyxy(N,4) 原图坐标, scores(N,), classes(N,))。"""
    h, w = im.shape[:2]
    tiles = tile_grid(w, h, nx, ny, ov)
    crops = [im[y0:y1, x0:x1] for (x0, y0, x1, y1) in tiles]
    B, S, C = [], [], []
    for i in range(0, len(crops), BATCH):
        rs = model.predict(crops[i:i + BATCH], imgsz=IMGSZ, conf=CONF, iou=TILE_NMS_IOU,
                           max_det=MAX_DET, agnostic_nms=False, verbose=False, stream=False)
        for (x0, y0, _, _), r in zip(tiles[i:i + BATCH], rs):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy().copy()
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            B.append(b)
            S.append(r.boxes.conf.cpu().numpy())
            C.append(r.boxes.cls.cpu().numpy().astype(int))
    if not B:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    b = np.concatenate(B).astype(np.float32)
    s = np.concatenate(S).astype(np.float32)
    c = np.concatenate(C)
    # 全局 NMS（class-wise）：batched_nms 用类别号做偏移，等价逐类 NMS
    keep = batched_nms(torch.from_numpy(b), torch.from_numpy(s),
                       torch.from_numpy(c), NMS_IOU).numpy()
    return b[keep], s[keep], c[keep]


def match(pb, pc, ps, gts):
    """返回 (命中 mask over gts, 命中数, 误检数)。"""
    gb = np.array([g['box'] for g in gts], np.float32) if gts else np.zeros((0, 4), np.float32)
    gc = np.array([g['cls'] for g in gts], int) if gts else np.zeros(0, int)
    gd = np.zeros(len(gts), bool)
    pd = np.zeros(len(pb), bool)
    M = iou_matrix(pb, gb)
    for j in np.argsort(-ps) if len(ps) else []:
        if pd[j]:
            continue
        hit = np.where((M[j] > MATCH_IOU) & (~gd) & (gc == pc[j]))[0]
        if len(hit):
            gd[hit[np.argmax(M[j, hit])]] = True
            pd[j] = True
            continue
        hit = np.where((M[j] > MATCH_IOU) & (~gd))[0]   # 错类也算命中（本脚本不统计错类）
        if len(hit):
            gd[hit[np.argmax(M[j, hit])]] = True
        pd[j] = True
    return gd, int(gd.sum()), int((~pd).sum())


def auto_grid(w, h, target=1024):
    """按长边自适应：让每块长边落在 ~target，避免小图被过度放大。"""
    nx = max(1, int(round(w / target)))
    ny = max(1, int(round(h / target)))
    return nx, ny


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='nf_v11_896_s42')
    ap.add_argument('--split', default='test')
    ap.add_argument('--only', default=None, help='只跑指定图片编号，逗号分隔，如 151,139')
    ap.add_argument('--modes', default='full,auto,2x2,3x2',
                    help='full=整图; auto=按长边自适应; 2x2/3x2=固定切分(仅对大图生效)')
    args = ap.parse_args()

    only = args.only.split(',') if args.only else None
    items = load_items(args.split, only)
    model = YOLO(str(RUNS / args.run / 'weights' / 'best.pt'))
    print(f'run={args.run}  split={args.split}  图数={len(items)}')
    print(f'口径 conf={CONF} 全局NMS={NMS_IOU} 块内NMS={TILE_NMS_IOU} 命中IoU={MATCH_IOU}\n')

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    stat = {}
    for mode in modes:
        tot, hit, fp = 0, 0, 0
        per_cls = defaultdict(lambda: [0, 0])
        per_zoom = defaultdict(lambda: [0, 0])   # 按整图缩放比分组的召回
        tiles_used = []
        for im_path, gts, w0, h0 in items:
            im = cv2.imread(str(im_path))
            if mode == 'full':
                nx, ny = 1, 1
            elif mode == 'auto':
                nx, ny = auto_grid(w0, h0)
            else:
                m = re.match(r'(\d+)x(\d+)', mode)
                nx, ny = int(m.group(1)), int(m.group(2))
                # 只对「明显大于输入」的图切，否则不切（小图切了等于过度放大）
                if max(w0, h0) < IMGSZ * 1.15:
                    nx, ny = 1, 1
            tiles_used.append(nx * ny)
            pb, ps, pc = infer(model, im, nx, ny, 0.25)
            gd, h_, f_ = match(pb, pc, ps, gts)
            tot += len(gts)
            hit += h_
            fp += f_
            for k, g in enumerate(gts):
                per_cls[NAMES[g['cls']]][0] += 1
                per_cls[NAMES[g['cls']]][1] += int(gd[k])
                r_s = min(IMGSZ / h0, IMGSZ / w0)
                zk = '缩小>2x' if r_s < 0.5 else ('缩小1.25-2x' if r_s < 0.8
                                                 else ('几乎原尺寸' if r_s < 1.2 else '放大>1.2x'))
                per_zoom[zk][0] += 1
                per_zoom[zk][1] += int(gd[k])
        stat[mode] = dict(tot=tot, hit=hit, fp=fp, per_cls=dict(per_cls),
                          per_zoom=dict(per_zoom), npred=sum(tiles_used))

    print('=' * 72)
    print(f'{"模式":<8} {"GT":>5} {"命中":>5} {"召回":>8} {"误检":>6}   break / thunder / wear')
    for mode in modes:
        s = stat[mode]
        pc = s['per_cls']
        det = '  '.join(f'{pc.get(c, [0, 0])[1]}/{pc.get(c, [0, 0])[0]}' for c in NAMES)
        print(f'{mode:<8} {s["tot"]:>5} {s["hit"]:>5} {s["hit"] / max(s["tot"], 1) * 100:7.2f}% '
              f'{s["fp"]:>6}   {det}')

    print('\n按整图缩放比的召回（看切片是否把"缩小组"拉平到"放大组"）:')
    zooms = ['缩小>2x', '缩小1.25-2x', '几乎原尺寸', '放大>1.2x']
    print(f'{"模式":<8}' + ''.join(f'{z:>14}' for z in zooms))
    for mode in modes:
        z = stat[mode]['per_zoom']
        row = ''
        for k in zooms:
            n, h_ = z.get(k, [0, 0])
            row += f'{(f"{h_}/{n} {h_ / n * 100:.1f}%" if n else "-"):>14}'
        print(f'{mode:<8}{row}')


if __name__ == '__main__':
    main()
