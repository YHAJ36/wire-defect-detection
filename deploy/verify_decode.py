# -*- coding: utf-8 -*-
"""kmodel vs ONNX 的**解码后**一致性（这才是上板前该看的数）

为什么不能只看张量误差：
  verify_kmodel.py 报的 max|Δ| 是整张张量（7×8400）的最大绝对差，
  分母取的是整张量最大值（≈637，因为框坐标是像素尺度）。
  这样算出来的 10% 会被**背景 anchor** 的回归噪声主导——
  背景 anchor 的框坐标错了 70 px 完全不重要，因为它根本不会过置信度阈值。
  真正决定 mAP 的是：过阈值的那几十个框，位置偏了多少。

本脚本的做法：
  · 以 ONNX 的解码结果当参考（同一张图、同一套后处理）
  · 分别对 kmodel 输出和 ONNX 输出做 conf@0.25 + 逐类 NMS
  · 按类别匹配框，报 IoU、召回、类别分数差
  两侧用完全相同的解码代码，所以差异只来自量化本身。

用法
    set DOTNET_ROLL_FORWARD=LatestMajor
    D:/yolo-env/Scripts/python.exe D:/2/deploy/verify_decode.py --n 8
"""
import argparse
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


def letterbox(im, new=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new - nh) // 2, (new - nw) // 2
    return cv2.copyMakeBorder(im, top, new - nh - top, left, new - nw - left,
                              cv2.BORDER_CONSTANT, value=color)


def prep(path, size=640):
    im = cv2.imread(str(path))
    if im is None:
        return None
    im = letterbox(im, size)[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(im.transpose(2, 0, 1)[None])


def prep_u8(path, size=640):
    """uint8 NHWC RGB，给「归一化写进图里」的那版 kmodel 用。"""
    im = cv2.imread(str(path))
    if im is None:
        return None
    im = letterbox(im, size)[:, :, ::-1]
    return np.ascontiguousarray(im)[None]


def nms(boxes, scores, thr=0.45):
    """boxes: (N,4) xyxy；返回保留索引。纯 numpy，不依赖 torch。"""
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
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / (a1 + a2 - inter + 1e-9)
        order = order[1:][iou < thr]
    return np.array(keep, dtype=int)


def decode(y, conf=0.25, iou_thr=0.45):
    """y: (7, 8400) -> list of (xyxy, score, cls)。框已是 640 像素尺度。"""
    y = np.asarray(y).reshape(7, -1)
    cx, cy, w, h = y[0], y[1], y[2], y[3]
    cls = y[4:]                       # (3, 8400)
    best = cls.max(0)
    m = best > conf
    if not m.any():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    b = np.stack([cx[m] - w[m] / 2, cy[m] - h[m] / 2,
                  cx[m] + w[m] / 2, cy[m] + h[m] / 2], 1).astype(np.float32)
    s = best[m].astype(np.float32)
    c = cls[:, m].argmax(0).astype(int)
    out_b, out_s, out_c = [], [], []
    for k in range(3):
        idx = np.where(c == k)[0]
        if not idx.size:
            continue
        keep = idx[nms(b[idx], s[idx], iou_thr)]
        out_b.append(b[keep]); out_s.append(s[keep]); out_c.append(np.full(keep.size, k))
    if not out_b:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    return np.concatenate(out_b), np.concatenate(out_s), np.concatenate(out_c)


def iou_mat(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    a1 = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    a2 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kmodel', default='D:/2/deploy/yolo11s_640.kmodel')
    ap.add_argument('--onnx', default='D:/2/deploy/yolo11s_640_single.onnx')
    ap.add_argument('--imgs', default='D:/2/dataset_noleak/images/test')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--dump', default='D:/2/deploy/verify_outputs.npz',
                    help='把两侧原始输出存下来，下次不用再跑模拟器')
    ap.add_argument('--u8', action='store_true',
                    help='kmodel 吃 uint8 NHWC（归一化写进图里的那一版）')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    sim = nncase.Simulator()
    sim.load_model(Path(args.kmodel).read_bytes())
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])

    files = sorted([p for p in Path(args.imgs).iterdir()
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    step = max(1, len(files) // args.n)
    picked = files[::step][:args.n]

    cache = {}
    if Path(args.dump).exists():
        z = np.load(args.dump)
        cache = {k: z[k] for k in z.files}
        print(f'复用缓存 {args.dump}（{len(cache)//2} 张）')

    print(f'对比 {len(picked)} 张，conf={args.conf}  逐类 NMS=0.45\n')
    print(f'{"图":<40}{"ONNX框":>7}{"KM框":>6}{"召回":>7}{"平均IoU":>8}'
          f'{"分数差":>8}')
    recs, ious, sds = [], [], []
    for p in picked:
        x = prep(p)
        if x is None:
            continue
        x8 = prep_u8(p) if args.u8 else None
        ko, kk = f'on_{p.name}', f'km_{p.name}'
        if ko in cache:
            yo, yk = cache[ko], cache[kk]
        else:
            yo = sess.run(None, {'images': x})[0]
            sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(x8 if args.u8 else x))
            sim.run()
            yk = sim.get_output_tensor(0).to_numpy()
            yk = yk.reshape(yo.shape)
            np.savez(args.dump, **{**cache, ko: yo, kk: yk})
            cache[ko], cache[kk] = yo, yk

        bo, so, co = decode(yo, args.conf)
        bk, sk, ck = decode(yk, args.conf)

        # 逐类匹配：ONNX 的框当参考，找 kmodel 里同类且 IoU 最大的
        m_iou, m_sd, hit = [], [], 0
        for k in range(3):
            io = np.where(co == k)[0]
            ik = np.where(ck == k)[0]
            if not io.size:
                continue
            if not ik.size:
                m_iou.extend([0.0] * io.size)
                continue
            M = iou_mat(bo[io], bk[ik])
            best_iou = M.max(1)
            best_j = M.argmax(1)
            keep = best_iou > 0.5
            hit += int(keep.sum())
            m_iou.extend(best_iou.tolist())
            m_sd.extend(np.abs(so[io][keep] - sk[ik][best_j][keep]).tolist())

        n_ref = len(bo)
        rec = hit / n_ref if n_ref else 1.0
        mi = float(np.mean(m_iou)) if m_iou else 1.0
        sd = float(np.mean(m_sd)) if m_sd else 0.0
        recs.append(rec); ious.append(mi); sds.append(sd)
        print(f'{p.name[:38]:<40}{n_ref:>7}{len(bk):>6}{rec*100:>6.0f}%'
              f'{mi:>8.3f}{sd:>8.3f}')

    print(f'\n汇总（{len(recs)} 张）')
    print(f'  框召回  均值 {np.mean(recs)*100:.1f}%   最低 {np.min(recs)*100:.1f}%')
    print(f'  匹配框 IoU 均值 {np.mean(ious):.3f}   最低 {np.min(ious):.3f}')
    print(f'  类别分数平均差 {np.mean(sds):.3f}')
    print('\n判读：召回 ≥95% 且 IoU ≥0.90 → 量化可直接用；'
          'IoU 掉到 0.8 以下说明回归通道被量化压坏了，要拆输出头重编。')


if __name__ == '__main__':
    main()
