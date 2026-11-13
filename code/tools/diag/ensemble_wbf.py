# -*- coding: utf-8 -*-
"""三种子输出融合（WBF / 框拼接）探针：零训练成本，看集成还能不能涨。

为什么值得一试
    A.9.31 判负了 model soup（权重平均），依据是三种子相对权重距离 relL2≈0.33，
    远超 soup 的有效前提（<0.05）。但**输出融合（ensemble）与权重平均的前提正好相反**：
    权重平均要三个解落在同一个损失盆地，输出融合要三个解**足够不同**、错的地方不重合。
    relL2=0.33 恰是集成理想的多样性条件。两者不是同一件事，之前只否了前者。

三种模式
    single : 单模型，用来**校验本脚本与官方 val 同口径**（s42 应复现 0.9197）
    wbf    : 加权框融合（Solovyev 2019）：按 conf 降序贪心聚簇，簇内按 conf 加权平均坐标，
             conf = 簇内均值 × min(n_models, n_boxes)/n_models
    nms    : 朴素拼接 + 全局 NMS（对照组，通常是 WBF 的下界）

指标计算
    直接复用官方 `DetectionValidator.match_predictions`（返回 N×10 的 tp 矩阵）与
    `DetMetrics.process`，因此 AP 算法与 ultralytics val 完全一致，不自己实现 AP。
    这是 A.9.32 教训的直接应用：任何对比必须同一条计算路径。

用法
    python code/tools/diag/ensemble_wbf.py --mode single --models nf_v11_896_s42
    python code/tools/diag/ensemble_wbf.py --mode wbf --models nf_v11_896_s42,nf_v11_896_s43,nf_v11_896_s44
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, 'D:/2/code/yolov11')
sys.path.insert(0, 'D:/2/code/tools/diag')
from ultralytics import YOLO                                      # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402
from ultralytics.utils.metrics import DetMetrics, box_iou          # noqa: E402
from diag_tiled_infer import load_items                            # noqa: E402

RUNS = Path('D:/2/exp_upscale/runs')
DATA = 'D:/2/dataset_final_bbox/data.yaml'
NAMES = {0: 'break', 1: 'thunderbolt', 2: 'wear'}
IMGSZ = 896
CONF = 0.001
MAX_DET = 300
NAMES_L = ['break', 'thunderbolt', 'wear']


def predict_one(ckpt, items, nms_iou, device='0'):
    """返回每张图的 (boxes xyxy, scores, classes)。"""
    m = YOLO(str(RUNS / ckpt / 'weights' / 'best.pt'))
    imgs = [p for p, _, _, _ in items]
    out = []
    for i in range(0, len(imgs), 4):
        rs = m.predict(imgs[i:i + 4], imgsz=IMGSZ, conf=CONF, iou=nms_iou, max_det=MAX_DET,
                       agnostic_nms=False, verbose=False, stream=False, device=device)
        for r in rs:
            if r.boxes is None or len(r.boxes) == 0:
                out.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)))
            else:
                out.append((r.boxes.xyxy.cpu().numpy().astype(np.float32),
                            r.boxes.conf.cpu().numpy().astype(np.float32),
                            r.boxes.cls.cpu().numpy().astype(int)))
    del m
    if str(device) != 'cpu':
        torch.cuda.empty_cache()
    return out


def wbf_fuse(packs, iou_thr=0.55, n_models=3):
    """packs: [(b,s,c), ...] 每个模型的输出。返回融合后的 (b,s,c)。

    为什么用「与簇首比较」而不是「与簇内所有框比较」：
    conf 0.001 下每张图可达数百框、三模型上千框，全比较是 O(n²) 的纯 Python 循环，
    实测跑不完。WBF 原论文的簇内 max-IoU 与「与簇首比」在框高度重叠的场景下结果几乎相同，
    这里取快的那种，并在结果里注明。
    """
    B = np.concatenate([p[0] for p in packs]) if packs else np.zeros((0, 4), np.float32)
    S = np.concatenate([p[1] for p in packs]) if packs else np.zeros(0, np.float32)
    C = np.concatenate([p[2] for p in packs]) if packs else np.zeros(0, int)
    M = np.concatenate([np.full(len(p[0]), k) for k, p in enumerate(packs)]) if packs else np.zeros(0, int)
    if len(B) == 0:
        return B, S, C

    order = np.argsort(-S)
    # 预计算全对 IoU，聚类时查表（一次性 numpy，比循环里反复算快两个数量级）
    IoU = box_iou(torch.from_numpy(B), torch.from_numpy(B)).numpy()

    members = []        # 每簇的框下标
    reps = []           # 每簇的簇首框下标
    rep_cls = []        # 每簇类别
    for i in order:
        if not members:
            members.append([i]); reps.append(i); rep_cls.append(C[i])
            continue
        rc = np.asarray(rep_cls)
        cand = np.where(rc == C[i])[0]
        if len(cand):
            ious = IoU[i, np.asarray(reps)[cand]]
            j = int(np.argmax(ious))
            if ious[j] > iou_thr:
                members[cand[j]].append(i)
                continue
        members.append([i]); reps.append(i); rep_cls.append(C[i])

    fb, fs, fc = [], [], []
    for cl in members:
        idx = np.asarray(cl)
        w = S[idx]
        fb.append((B[idx] * w[:, None]).sum(0) / w.sum())
        n_in = len(np.unique(M[idx]))
        fs.append(float(w.mean()) * min(n_in, n_models) / n_models)
        fc.append(int(C[idx[0]]))
    return (np.asarray(fb, np.float32).reshape(-1, 4),
            np.asarray(fs, np.float32), np.asarray(fc, int))


def compute_metric(all_pred, items, save_dir, tag):
    """用官方 match_predictions + DetMetrics 算指标。"""
    val = DetectionValidator()
    dm = DetMetrics(names=NAMES)
    dm.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    for img_id, ((path, gts, w0, h0), (b, s, c)) in enumerate(zip(items, all_pred)):
        if len(gts):
            gb = torch.from_numpy(np.array([g['box'] for g in gts], np.float32))
            gc = torch.from_numpy(np.array([g['cls'] for g in gts], int))
        else:
            gb = torch.zeros((0, 4))
            gc = torch.zeros(0, dtype=torch.int64)
        pb = torch.from_numpy(b)
        pc = torch.from_numpy(c).int()
        if len(pb) and len(gb):
            iou = box_iou(gb, pb)
        else:
            iou = torch.zeros((len(gb), len(pb)))
        tp = val.match_predictions(pc, gc, iou)          # (N,10)
        dm.stats['tp'].append(np.asarray(tp))
        dm.stats['conf'].append(s)
        dm.stats['pred_cls'].append(c)
        dm.stats['target_cls'].append(np.asarray(gc))
        dm.stats['target_img'].append(np.full(len(gc), img_id))

    dm.process(save_dir=Path(save_dir), plot=False)
    return dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='single', choices=['single', 'wbf', 'nms'])
    ap.add_argument('--models', default='nf_v11_896_s42')
    ap.add_argument('--nms', type=float, default=0.5, help='单模型内部 NMS')
    ap.add_argument('--wbf-iou', type=float, default=0.55)
    ap.add_argument('--split', default='test')
    ap.add_argument('--device', default='0', help='0 / cpu；训练占卡时用 cpu')
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    items = load_items(args.split)
    print(f'mode={args.mode}  models={models}  split={args.split}  {len(items)} 张  '
          f'conf={CONF} NMS={args.nms} max_det={MAX_DET}')

    packs_all = []
    for ck in models:
        packs_all.append(predict_one(ck, items, args.nms, args.device))
        print(f'  已推理 {ck}')

    if args.mode == 'single':
        all_pred = packs_all[0]
    elif args.mode == 'wbf':
        all_pred = [wbf_fuse([p[i] for p in packs_all], args.wbf_iou, len(models))
                    for i in range(len(items))]
    else:
        all_pred = []
        for i in range(len(items)):
            b = np.concatenate([p[i][0] for p in packs_all])
            s = np.concatenate([p[i][1] for p in packs_all])
            c = np.concatenate([p[i][2] for p in packs_all])
            if len(b):
                keep = torch.ops.torchvision.nms(torch.from_numpy(b), torch.from_numpy(s), args.nms)
                if len(c):
                    # torchvision nms 不分类别，这里逐类做一次
                    ks = []
                    for cl in np.unique(c):
                        m = c == cl
                        k = torch.ops.torchvision.nms(torch.from_numpy(b[m]),
                                                      torch.from_numpy(s[m]), args.nms)
                        ks.append(np.where(m)[0][k.numpy()])
                    keep = torch.from_numpy(np.sort(np.concatenate(ks)))
                b, s, c = b[keep.numpy()], s[keep.numpy()], c[keep.numpy()]
            all_pred.append((b, s, c))

    outdir = Path('D:/2/exp_upscale/eval_ensemble') / f'{args.mode}_{len(models)}m'
    outdir.mkdir(parents=True, exist_ok=True)
    dm = compute_metric(all_pred, items, outdir, args.mode)
    bx = dm.box
    print(f'\n{"":<14}{"mAP50":>9}{"mAP50-95":>11}{"P":>9}{"R":>9}')
    print(f'{args.mode:<14}{float(bx.map50):>9.4f}{float(bx.map):>11.4f}'
          f'{float(bx.mp):>9.4f}{float(bx.mr):>9.4f}')
    pc = bx.ap_class_index
    for k, ci in enumerate(pc):
        print(f'  {NAMES_L[ci]:<12} AP50={bx.ap50[k]:.4f}  AP={bx.ap[k]:.4f}')


if __name__ == '__main__':
    main()
