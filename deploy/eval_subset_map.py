# -*- coding: utf-8 -*-
"""在测试集子集上算 ONNX / kmodel 的 mAP50，直接给出「量化掉多少」这一个数。

为什么值得花这个时间：
  verify_decode.py 给的是「相对 ONNX 的召回与 IoU」，那是**代理指标**；
  分数压缩（logits 量化步长 ≈0.28，sigmoid 之后在 p=0.4 附近就被放大成 0.067）
  对排序的破坏，代理指标看不出来。只有对着 GT 算 AP 才知道真实代价。

口径
  · 预测在 640 letterbox 空间，先反 letterbox 回原图坐标，再和 GT 比。
  · conf=0.001（和 ultralytics 验证一致）、逐类 NMS 0.45、IoU 阈值 0.5。
  · AP 用 101 点插值，和 COCO/ultralytics 的 mAP50 口径一致。
  · 子集是等距抽样，同一个子集跑 ONNX 和 kmodel，配对可比。

用法
    set DOTNET_ROLL_FORWARD=LatestMajor
    D:/yolo-env/Scripts/python.exe D:/2/deploy/eval_subset_map.py --n 30 \
        --kmodel D:/2/deploy/yolo11s_640_u8r.kmodel --u8
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault('DOTNET_ROLL_FORWARD', 'LatestMajor')
_SP = Path(sys.prefix) / 'Lib' / 'site-packages'
for _d in (_SP, _SP / 'nncase', _SP / 'nncase' / 'modules' / 'kpu'):
    if _d.is_dir():
        try:
            os.add_dll_directory(str(_d))
        except Exception:
            pass
os.environ.setdefault('NNCASE_PLUGIN_PATH', str(_SP / 'nncase' / 'modules' / 'kpu'))
os.environ['PATH'] = str(_SP) + os.pathsep + os.environ.get('PATH', '')

import nncase  # noqa: E402
import onnxruntime as ort  # noqa: E402

NAMES = ['break', 'thunder', 'wear']


def letterbox_param(h, w, new=640):
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    return r, (new - nh) // 2, (new - nw) // 2


def load_img(path, size=640):
    im = cv2.imread(str(path))
    if im is None:
        return None, None, None
    h, w = im.shape[:2]
    r, pt, pl = letterbox_param(h, w, size)
    rgb = cv2.resize(im, (int(round(w * r)), int(round(h * r))),
                     interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[pt:pt + rgb.shape[0], pl:pl + rgb.shape[1]] = rgb
    return canvas, (r, pt, pl), (h, w)


def nms(boxes, scores, thr=0.45):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        b = boxes[order[1:]]
        xx1 = np.maximum(boxes[i, 0], b[:, 0])
        yy1 = np.maximum(boxes[i, 1], b[:, 1])
        xx2 = np.minimum(boxes[i, 2], b[:, 2])
        yy2 = np.minimum(boxes[i, 3], b[:, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        order = order[1:][inter / (a1 + a2 - inter + 1e-9) < thr]
    return np.array(keep, dtype=int)


def decode(y, geo, imsz, conf=0.001, iou_thr=0.45):
    """y (7,8400) -> (N,4) xyxy 原图坐标, scores, classes"""
    y = np.asarray(y).reshape(7, -1)
    cx, cy, w, h = y[0], y[1], y[2], y[3]
    cls = y[4:]
    best = cls.max(0)
    m = best > conf
    if not m.any():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    b = np.stack([cx[m] - w[m] / 2, cy[m] - h[m] / 2,
                  cx[m] + w[m] / 2, cy[m] + h[m] / 2], 1).astype(np.float32)
    s = best[m].astype(np.float32)
    c = cls[:, m].argmax(0).astype(int)
    keep_b, keep_s, keep_c = [], [], []
    for k in range(3):
        idx = np.where(c == k)[0]
        if not idx.size:
            continue
        sel = idx[nms(b[idx], s[idx], iou_thr)]
        keep_b.append(b[sel]); keep_s.append(s[sel]); keep_c.append(np.full(sel.size, k))
    if not keep_b:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    B = np.concatenate(keep_b); S = np.concatenate(keep_s); C = np.concatenate(keep_c)
    # 反 letterbox：去灰边、除缩放
    r, pt, pl = geo
    H, W = imsz
    B[:, [0, 2]] = (B[:, [0, 2]] - pl) / r
    B[:, [1, 3]] = (B[:, [1, 3]] - pt) / r
    B[:, [0, 2]] = B[:, [0, 2]].clip(0, W)
    B[:, [1, 3]] = B[:, [1, 3]].clip(0, H)
    return B, S, C


def load_gt(label_path, imsz):
    H, W = imsz
    if not label_path.exists():
        return np.zeros((0, 5), np.float32)
    a = np.loadtxt(label_path, ndmin=2)
    if a.size == 0:
        return np.zeros((0, 5), np.float32)
    cls = a[:, 0:1]
    cx, cy, w, h = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
    box = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return np.concatenate([cls, box], 1).astype(np.float32)


def iou_1toN(box, boxes):
    if len(boxes) == 0:
        return np.zeros(0, np.float32)
    xx1 = np.maximum(box[0], boxes[:, 0]); yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2]); yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    a1 = (box[2] - box[0]) * (box[3] - box[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-9)


def ap50(preds, gts):
    """preds: list[(boxes,scores,classes)]；gts: list[(n,5) cls+xyxy]。
    返回逐类 AP 与 mAP。**预测必须按类过滤**——不过滤就会把别的类的框
    当成这一类的误检/命中，AP 会虚高（自测：不过滤时「全错」样例报 1.0）。"""
    n_gt = {k: 0 for k in range(3)}
    for g in gts:
        for c in range(3):
            n_gt[c] += int((g[:, 0] == c).sum())
    aps = {}
    for c in range(3):
        if n_gt[c] == 0:
            continue
        all_s, all_tp = [], []
        for (boxes, scores, classes), g in zip(preds, gts):
            sel = classes == c
            pb, ps = boxes[sel], scores[sel]
            gb = g[g[:, 0] == c][:, 1:5]
            order = ps.argsort()[::-1]
            used = np.zeros(len(gb), bool)
            for j in order:
                io = iou_1toN(pb[j], gb)
                if len(io) == 0:
                    all_s.append(ps[j]); all_tp.append(0)
                    continue
                io = np.where(used, -1, io)
                k = int(io.argmax())
                if io[k] >= 0.5:
                    used[k] = True
                    all_s.append(ps[j]); all_tp.append(1)
                else:
                    all_s.append(ps[j]); all_tp.append(0)
        if not all_s:
            aps[c] = 0.0
            continue
        all_s = np.array(all_s); all_tp = np.array(all_tp)
        o = all_s.argsort()[::-1]
        tp = all_tp[o]
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(1 - tp)
        rc = cum_tp / max(n_gt[c], 1)
        pr = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        mrec = np.linspace(0, 1, 101)
        pr_i = np.array([pr[rc >= t].max() if (rc >= t).any() else 0.0 for t in mrec])
        aps[c] = float(pr_i.mean())
    return aps, (float(np.mean(list(aps.values()))) if aps else 0.0), n_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kmodel', default='D:/2/deploy/yolo11s_640_u8r.kmodel')
    ap.add_argument('--onnx', default='D:/2/deploy/yolo11s_640_single.onnx')
    ap.add_argument('--imgs', default='D:/2/dataset_noleak/images/test')
    ap.add_argument('--labels', default='D:/2/dataset_noleak/labels/test')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--u8', action='store_true')
    ap.add_argument('--cache', default='D:/2/deploy/map_cache.npz')
    ap.add_argument('--onnx-only', action='store_true',
                    help='只跑 ONNX，跳过 nncase 模拟器（用来快速核对口径；Δ 恒为 0）')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    if args.onnx_only:
        sim = None          # 不加载 kmodel，省钱也省 6 倍时间
    else:
        sim = nncase.Simulator()
        sim.load_model(Path(args.kmodel).read_bytes())
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])

    files = sorted([p for p in Path(args.imgs).iterdir()
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    step = max(1, len(files) // args.n)
    picked = files[::step][:args.n]
    print(f'子集 {len(picked)} 张（每 {step} 张取 1），kmodel={Path(args.kmodel).name} u8={args.u8}')

    cache = {}
    if Path(args.cache).exists():
        z = np.load(args.cache)
        cache = {k: z[k] for k in z.files}

    po, pk, gts = [], [], []
    for i, p in enumerate(picked):
        canvas, geo, imsz = load_img(p)
        if canvas is None:
            continue
        gt = load_gt(Path(args.labels) / (p.stem + '.txt'), imsz)
        gts.append(gt)
        xo = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None]
                                  ).astype(np.float32) / 255.0
        xk = np.ascontiguousarray(canvas[None]) if args.u8 else xo

        ko, kk = 'o_' + p.name, 'k_' + p.name
        if args.onnx_only:
            # 只核对 ONNX 口径：一律 yk = yo，Δ 恒为 0，也不写缓存
            yo = cache[ko] if ko in cache else sess.run(None, {'images': xo})[0]
            yk = yo
        elif ko in cache and kk in cache:
            yo, yk = cache[ko], cache[kk]
        else:
            yo = sess.run(None, {'images': xo})[0]
            sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(xk))
            sim.run()
            yk = sim.get_output_tensor(0).to_numpy().reshape(yo.shape)
            np.savez(args.cache, **{**cache, ko: yo, kk: yk})
            cache[ko], cache[kk] = yo, yk

        po.append(decode(yo, geo, imsz))
        pk.append(decode(yk, geo, imsz))
        if (i + 1) % 5 == 0:
            print(f'  {i+1}/{len(picked)} 完成', flush=True)

    aps_o, map_o, ngt = ap50(po, gts)
    aps_k, map_k, _ = ap50(pk, gts)
    print(f'\nGT 框数 {ngt}')
    print(f'{"":<10}{"break":>9}{"thunder":>9}{"wear":>9}{"mAP50":>9}')
    print(f'{"ONNX":<10}{aps_o.get(0,0):>9.4f}{aps_o.get(1,0):>9.4f}'
          f'{aps_o.get(2,0):>9.4f}{map_o:>9.4f}')
    print(f'{"kmodel":<10}{aps_k.get(0,0):>9.4f}{aps_k.get(1,0):>9.4f}'
          f'{aps_k.get(2,0):>9.4f}{map_k:>9.4f}')
    print(f'{"Δ":<10}{aps_k.get(0,0)-aps_o.get(0,0):>+9.4f}'
          f'{aps_k.get(1,0)-aps_o.get(1,0):>+9.4f}'
          f'{aps_k.get(2,0)-aps_o.get(2,0):>+9.4f}{map_k-map_o:>+9.4f}')
    out = ('D:/2/deploy/map_full_onnx.json' if args.onnx_only
           else 'D:/2/deploy/map_subset.json')
    Path(out).write_text(json.dumps(
        {'n': len(picked), 'onnx': aps_o, 'kmodel': aps_k,
         'map50_onnx': map_o, 'map50_kmodel': map_k,
         'n_gt': ngt, 'files': [p.name for p in picked]},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n写出 {out}')


if __name__ == '__main__':
    main()
