# -*- coding: utf-8 -*-
"""P2v2 判负的归因诊断：多出来的那个尺度到底干了什么。

背景
    A.9.31 结论是「896 上加 P2/4 尺度，mAP50 −0.68 pt、方差放大 42.7 倍、break −1.54 pt」。
    当时给的解释（"候选框翻倍、低质量框进了 FP"）只是推理，没有测量。本脚本去测。

三个问题
    Q1 每个尺度各自输出了多少框、其中多少是真命中（TP）—— P2 分支在干什么？
    Q2 每个真值框最终被哪个尺度命中 —— P2 有没有抢走本该 P3 负责的目标？
    Q3 <32px 的小目标（break 类的主力）到底有没有因为 P2 而被救回来？

口径（与正式判读一致）
    test 179 张 / 896 letterbox / conf 0.25 / NMS IoU 0.5 / class-wise（multi_label=False）
    / max_det 300。自实现的 NMS 会先与 ultralytics 官方 non_max_suppression 对拍，
    框数不一致就报警——不然下面的归因全部无效。

用法：
    python code/tools/diag/diag_scale_attrib.py <run> [<run> ...]
"""
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from ultralytics import YOLO                                    # noqa: E402
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy  # noqa: E402

DS = Path('D:/2/dataset_final_bbox')
RUNS = Path('D:/2/exp_upscale/runs')
OUT = Path('D:/2/report/_p2_diag')
IMGSZ = 896
CONF = 0.25
NMS_IOU = 0.5
MAX_DET = 300
NAMES = ['break', 'thunderbolt', 'wear']
# 自实现贪心 NMS 是 O(n²) 的 Python 循环，P2v2 单图候选上万时极慢；
# 归因并不依赖它（已改用官方 NMS + 坐标反查），故默认关闭，置 1 才做对拍。
NMS_CHECK = os.environ.get('DIAG_NMS_CHECK', '0') == '1'


def letterbox_np(im0, size=IMGSZ, color=(114, 114, 114)):
    """自实现 letterbox：等比缩放 + 居中灰边。

    ultralytics 8.3.x 把 letterbox 改成了 LetterBox 类，接口与 8.2 不同；
    这里自己实现 6 行，避免跟着版本改。整数 pad 保证 GT 与预测在同一坐标网格上。
    """
    h0, w0 = im0.shape[:2]
    r = min(size / h0, size / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    im = cv2.resize(im0, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w0, h0) else im0
    top, left = (size - nh) // 2, (size - nw) // 2
    im = cv2.copyMakeBorder(im, top, size - nh - top, left, size - nw - left,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, left, top


# ---------------------------------------------------------------- 基础工具
def load_gt(img_path):
    """读标签 → letterbox 坐标系下的 xyxy 与边长。"""
    lbl = DS / 'labels' / 'test' / (img_path.stem + '.txt')
    im0 = cv2.imread(str(img_path))
    h0, w0 = im0.shape[:2]
    im, r, dw, dh = letterbox_np(im0)
    gts = []
    if lbl.exists():
        for line in lbl.read_text().strip().splitlines():
            if not line.strip():
                continue
            p = line.split()
            c = int(float(p[0]))
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            x1 = (cx - bw / 2) * w0 * r + dw
            y1 = (cy - bh / 2) * h0 * r + dh
            x2 = (cx + bw / 2) * w0 * r + dw
            y2 = (cy + bh / 2) * h0 * r + dh
            gts.append(dict(cls=c, box=[x1, y1, x2, y2],
                            area=(x2 - x1) * (y2 - y1),
                            side=np.sqrt(max((x2 - x1) * (y2 - y1), 0))))
    return im, im0, gts


def iou_matrix(a, b):
    """a:(n,4) b:(m,4) → (n,m)。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    ua = ((a[:, None, 2] - a[:, None, 0]) * (a[:, None, 3] - a[:, None, 1])
          + (b[None, :, 2] - b[None, :, 0]) * (b[None, :, 3] - b[None, :, 1]) - inter)
    return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0.0)


def my_nms(boxes, scores, cls, iou_thr=NMS_IOU, max_det=MAX_DET):
    """class-wise 贪心 NMS，返回保留下标（按置信度降序）。

    与 ultralytics non_max_suppression(multi_label=False) 等价：
    类别用坐标偏移隔开，使跨类 IoU 恒为 0。
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    off = cls.astype(np.float32)[:, None] * (IMGSZ * 2.0)   # 类别偏移，跨类不相交
    b = boxes + off
    order = np.argsort(-scores)
    keep = []
    alive = np.ones(len(b), dtype=bool)
    for i in order:
        if not alive[i]:
            continue
        keep.append(i)
        if len(keep) >= max_det:
            break
        iou = iou_matrix(b[i:i + 1], b)[0]
        alive &= ~(iou > iou_thr)
        alive[i] = False
    return np.array(keep, dtype=int)


BUCKETS = [('small <32px', 0, 32 * 32), ('medium 32-96px', 32 * 32, 96 * 96),
           ('large >96px', 96 * 96, 1e18)]


def bucket_of(area):
    for name, lo, hi in BUCKETS:
        if lo <= area < hi:
            return name
    return BUCKETS[-1][0]


def run_one(run):
    ck = RUNS / run / 'weights' / 'best.pt'
    model = YOLO(str(ck))
    net = model.model.float().eval()
    det = net.model[-1]
    strides = [float(s) for s in det.stride]
    nsc = len(strides)

    cap = {}

    def hook(mod, inp, out):
        cap['o'] = out

    h = det.register_forward_hook(hook)

    imgs = sorted((DS / 'images' / 'test').glob('*.*'))
    # 每尺度累计
    stat = {i: dict(n_conf=0, n_keep=0, tp=0, fp=0, scores=[]) for i in range(nsc)}
    # GT 命中的尺度来源（逐类逐桶）
    hit_src = defaultdict(Counter)     # (cls, bucket) -> Counter(scale)
    missed = Counter()                 # (cls, bucket) -> n
    conf_hi = Counter()                # conf>0.5 的框数（按尺度）
    n_nms_total = 0
    n_official = 0                     # 官方 NMS 交叉验证

    with torch.no_grad():
        for p in imgs:
            im, im0, gts = load_gt(p)
            t = torch.from_numpy(im.copy()).permute(2, 0, 1)[None].float() / 255.0
            if t.shape[-2:] != (IMGSZ, IMGSZ):
                t = torch.nn.functional.interpolate(t, (IMGSZ, IMGSZ))
            net(t)
            y = cap['o'][0]                                   # (1, 4+nc, N)
            y = y[0].float()
            seglen = []
            for i in range(nsc):
                hh = IMGSZ // int(strides[i])
                seglen.append(hh * hh)
            offs = np.cumsum([0] + seglen)

            raw = y.cpu().numpy()
            boxes_xywh, cls_score = raw[:4].T, raw[4:].T        # (N,4) (N,nc)
            conf = cls_score.max(1)
            cls = cls_score.argmax(1)
            src = np.zeros(len(conf), dtype=int)
            for i in range(nsc):
                src[offs[i]:offs[i + 1]] = i
                m = conf[offs[i]:offs[i + 1]] > CONF
                stat[i]['n_conf'] += int(m.sum())
                stat[i]['scores'].extend(conf[offs[i]:offs[i + 1]][m].tolist())
            conf_hi_local = conf > 0.5
            for i in range(nsc):
                sl = slice(offs[i], offs[i + 1])
                conf_hi[i] += int(conf_hi_local[sl].sum())

            sel = conf > CONF
            if sel.sum() == 0:
                for g in gts:
                    missed[(g['cls'], bucket_of(g['area']))] += 1
                continue
            bx = xywh2xyxy(torch.from_numpy(boxes_xywh[sel])).numpy()
            sc, cl, sr = conf[sel], cls[sel], src[sel]

            # 归因必须建立在**官方 NMS 的框集合**上：自实现贪心 NMS 与
            # torchvision.ops.nms 在边界 IoU 上会有差异（实测单颗差 12/278 框），
            # 用自实现的结果归因等于换了一套预测。这里先用官方 NMS 定框，
            # 再按坐标精确反查每个保留框来自哪个尺度段（NMS 只筛选、不改坐标）。
            pred = torch.from_numpy(
                np.concatenate([boxes_xywh[sel], cls_score[sel]], 1).T.copy())[None]
            off = non_max_suppression(pred, CONF, NMS_IOU, multi_label=False,
                                      max_det=MAX_DET, nc=len(NAMES))[0]
            if off is None or len(off) == 0:
                for g in gts:
                    missed[(g['cls'], bucket_of(g['area']))] += 1
                continue
            off = off.cpu().numpy()
            kb, ks, kc = off[:, :4], off[:, 4], off[:, 5].astype(int)
            n_official += len(off)
            if NMS_CHECK:                                # 仅当需要与自实现对拍时才跑（很慢）
                n_nms_total += len(my_nms(bx, sc, cl))

            # 坐标指纹：官方输出 (x1,y1,x2,y2) 在候选中找到同值 → 取该候选的尺度
            lut = {}
            for i in range(len(bx)):
                lut[(round(float(bx[i, 0]), 2), round(float(bx[i, 1]), 2),
                     round(float(bx[i, 2]), 2), round(float(bx[i, 3]), 2))] = int(sr[i])
            ksrc = np.empty(len(kb), dtype=int)
            for j in range(len(kb)):
                key = (round(float(kb[j, 0]), 2), round(float(kb[j, 1]), 2),
                       round(float(kb[j, 2]), 2), round(float(kb[j, 3]), 2))
                if key in lut:
                    ksrc[j] = lut[key]
                else:                                    # 兜底：最近邻（理论上不该走到）
                    d = np.abs(bx - kb[j]).sum(1)
                    ksrc[j] = int(sr[int(d.argmin())])

            # 一配一匹配：按 conf 降序，每个预测框最多配一个同类的 GT
            gb = np.array([g['box'] for g in gts], dtype=np.float32) if gts else np.zeros((0, 4))
            gc = np.array([g['cls'] for g in gts], dtype=int) if gts else np.zeros(0, dtype=int)
            gt_used = np.zeros(len(gts), dtype=bool)
            order = np.argsort(-ks)
            for j in order:
                iou = iou_matrix(kb[j:j + 1], gb)[0] if len(gb) else np.zeros(0)
                best, bi = 0.0, -1
                for gi in range(len(gb)):
                    if gt_used[gi] or gc[gi] != kc[j]:
                        continue
                    if iou[gi] > best:
                        best, bi = iou[gi], gi
                if best >= NMS_IOU and bi >= 0:
                    gt_used[bi] = True
                    stat[int(ksrc[j])]['tp'] += 1
                    g = gts[bi]
                    hit_src[(g['cls'], bucket_of(g['area']))][int(strides[ksrc[j]])] += 1
                else:
                    stat[int(ksrc[j])]['fp'] += 1
            for i in range(nsc):
                stat[i]['n_keep'] += int((ksrc == i).sum())
            for gi, g in enumerate(gts):
                if not gt_used[gi]:
                    missed[(g['cls'], bucket_of(g['area']))] += 1

    h.remove()
    return dict(run=run, strides=strides, stat=stat, hit_src=hit_src, missed=missed,
                conf_hi=conf_hi, n_nms=n_nms_total, n_official=n_official,
                n_img=len(imgs))


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    runs = sys.argv[1:]
    if not runs:
        raise SystemExit('用法：diag_scale_attrib.py <run> [<run> ...]')
    OUT.mkdir(parents=True, exist_ok=True)
    res = [run_one(r) for r in runs]

    for R in res:
        print('\n' + '=' * 74)
        print('run:', R['run'], '| strides', [int(s) for s in R['strides']])
        if NMS_CHECK:
            print('  NMS 对拍：自实现贪心 %d vs 官方 %d（差 %d，%+.2f%%）'
                  '→ 归因统一采用官方框集合' % (
                      R['n_nms'], R['n_official'], R['n_official'] - R['n_nms'],
                      100.0 * (R['n_nms'] - R['n_official']) / max(R['n_official'], 1)))
        else:
            print('  NMS：官方框集合 %d 个（归因口径）' % R['n_official'])
        print('  %-8s %-10s %-8s %-8s %-9s %-10s' % (
            'stride', 'conf>0.25', 'NMS后', 'conf>0.5', 'TP', '精度TP/保留'))
        tot_k = sum(s['n_keep'] for s in R['stat'].values())
        tot_tp = sum(s['tp'] for s in R['stat'].values())
        for i in range(len(R['strides'])):
            st = R['stat'][i]
            pr = st['tp'] / st['n_keep'] if st['n_keep'] else float('nan')
            print('  %-8d %-10d %-8d %-8d %-9d %.3f' % (
                int(R['strides'][i]), st['n_conf'], st['n_keep'], R['conf_hi'][i],
                st['tp'], pr))
        print('  %-8s %-10s %-8d %-8s %-9d %.3f' % (
            '合计', '', tot_k, '', tot_tp, tot_tp / tot_k if tot_k else float('nan')))

        print('  -- 真值框被哪个尺度的框命中（test, conf>0.25, NMS0.5）--')
        for c in range(len(NAMES)):
            for bname, _, _ in BUCKETS:
                cnt = R['hit_src'].get((c, bname), Counter())
                n = sum(cnt.values()) + R['missed'][(c, bname)]
                if n == 0:
                    continue
                det = ' '.join('s%d:%d' % (k, v) for k, v in sorted(cnt.items()))
                print('    %-12s %-14s 命中/总数 %2d/%-2d  漏 %d   %s' % (
                    NAMES[c], bname, sum(cnt.values()), n, R['missed'][(c, bname)], det))

    # 基线 vs P2v2 的尺度映射对照（只看共有的 s8/s16/s32）
    print('\n' + '=' * 74)
    print('【小目标（<32px）逐类汇总：被命中数 / 总数】')
    print('%-30s %-12s %-12s' % ('run', 'break', 'wear'))
    for R in res:
        cells = []
        for c in (0, 2):
            cnt = R['hit_src'].get((c, 'small <32px'), Counter())
            n = sum(cnt.values()) + R['missed'][(c, 'small <32px')]
            cells.append('%d/%d' % (sum(cnt.values()), n))
        print('%-30s %-12s %-12s' % (R['run'], cells[0], cells[1]))

    import json
    dump = [dict(run=R['run'], strides=[int(s) for s in R['strides']],
                 per_scale={int(R['strides'][i]): dict(
                     n_conf=R['stat'][i]['n_conf'], n_keep=R['stat'][i]['n_keep'],
                     tp=R['stat'][i]['tp'], fp=R['stat'][i]['fp'],
                     conf_gt05=R['conf_hi'][i],
                     median_score=(float(np.median(R['stat'][i]['scores']))
                                   if R['stat'][i]['scores'] else None))
                     for i in range(len(R['strides']))},
                 hit={f'{NAMES[c]}|{b}': {str(k): v for k, v in cnt.items()}
                      for (c, b), cnt in R['hit_src'].items()},
                 miss={f'{NAMES[c]}|{b}': v for (c, b), v in R['missed'].items()},
                 nms_self=R['n_nms'], nms_official=R['n_official'], n_img=R['n_img'])
            for R in res]
    fp = OUT / 'scale_attrib.json'
    fp.write_text(json.dumps(dump, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n→ 明细已写入', fp)


if __name__ == '__main__':
    main()
