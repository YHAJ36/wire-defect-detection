# -*- coding: utf-8 -*-
"""混合族 WBF：base(3) + Triplet(3) = 6 个模型一起融合，和老 / 新两张考卷各跑一遍。

复用 eval_wbf_tta.py 的注入框架（同一 update_metrics 注入点、同一 val 口径），
只把 WBF 换成 6 模型版本，并额外跑一个"关掉一致性惩罚"的变体：
    标准 Solovyev/ZFTurbo: score = Σs/N · min(n,N)/N      N=6 时，
        只被一族（3 票）找到的框得分被压到 1/2，这是混合族掉分的可能来源。
    无惩罚变体:          score = Σs/N                    只保留平均分，不加票数折减。

用法：D:/yolo-env/Scripts/python.exe code/tools/eval/eval_wbf_mixed.py
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, 'D:/2/code/tools/eval')
import eval_wbf_tta as E  # noqa: E402  导入即完成 update_metrics 打桩

BASE = ['nf_v11_896_s42', 'nf_v11_896_s43', 'nf_v11_896_s44']
TRIP = ['att_v11_896_triplet_s42', 'att_v11_896_triplet_s43', 'att_v11_896_triplet_s44']
ALL6 = BASE + TRIP

DATASETS = [
    ('old179', 'D:/2/dataset_final_bbox/data.yaml', Path('D:/2/exp_upscale/logs/wbf_cache179')),
    ('clean100', 'D:/2/dataset_noleak_clean/data.yaml', Path('D:/2/exp_upscale/logs/wbf_cache')),
]
OUT_TSV = Path('D:/2/exp_upscale/logs/eval_wbf_mixed.tsv')


# ------------------------------------------------------------------ WBF
def _iou_one(a, b):
    if len(b) == 0:
        return np.zeros((0,), np.float32)
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(aa + ab - inter, 1e-9)


def wbf(boxes_list, scores_list, labels_list, n_models, iou_thr=0.55, penalty=True):
    all_b = np.concatenate(boxes_list, 0)
    all_s = np.concatenate(scores_list, 0)
    all_c = np.concatenate(labels_list, 0)
    if len(all_s) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)
    ob, os_, oc = [], [], []
    for c in np.unique(all_c):
        m = all_c == c
        b = all_b[m].astype(np.float32); s = all_s[m].astype(np.float32)
        order = np.argsort(-s, kind='stable'); b, s = b[order], s[order]
        alive = np.ones(len(s), dtype=bool)
        while alive.any():
            idx = np.flatnonzero(alive)
            sel = idx[_iou_one(b[idx[0]], b[idx]) > iou_thr]
            w = s[sel]
            ob.append((b[sel] * w[:, None]).sum(0) / max(w.sum(), 1e-9))
            base = w.sum() / n_models
            if penalty:
                base *= min(len(sel), n_models) / n_models
            os_.append(base)
            oc.append(c)
            alive[sel] = False
    return np.asarray(ob, np.float32), np.asarray(os_, np.float32), np.asarray(oc, np.int64)


def provider(caches, iou_thr, n_models, penalty=True):
    def f(preds, batch, validator):
        hs, ws = batch['img'].shape[2], batch['img'].shape[3]
        dev = batch['bboxes'].device
        out = []
        for si in range(len(batch['im_file'])):
            stem = Path(batch['im_file'][si]).stem
            fb, fs, fc = wbf([c[stem]['boxes'] for c in caches],
                             [c[stem]['conf'] for c in caches],
                             [c[stem]['cls'] for c in caches],
                             n_models, iou_thr, penalty)
            if len(fs):
                fb = fb * np.array([ws, hs, ws, hs], np.float32)
            out.append({'bboxes': torch.from_numpy(fb).to(dev),
                        'conf': torch.from_numpy(fs).to(dev),
                        'cls': torch.from_numpy(fc.astype(np.float32)).to(dev),
                        'extra': torch.zeros((len(fs), 0), device=dev)})
        return out
    return f


def load_npz(p):
    z = np.load(p)
    keys = [str(k) for k in z['__keys__']]
    return {k: {'boxes': z[k][:, :4], 'conf': z[k][:, 4], 'cls': z[k][:, 5].astype(np.int64)}
            for k in keys}


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    rows = []
    for tag, yaml, cdir in DATASETS:
        E.DATA = yaml
        E.CACHE = cdir
        print(f'\n########## {tag}   {yaml}')
        caches = {}
        for run in ALL6:
            p = cdir / f'{run}_raw.npz'
            if p.exists():
                caches[run] = load_npz(p)
            else:
                c = {}
                E.run_val(run, augment=False, collect=c, tag=f'collect179_{run}')
                E.save_cache(p, c)
                caches[run] = c
            print(f'  cache {run:<26} imgs={len(caches[run])}')

        cases = [
            ('base 三颗 WBF', [caches[r] for r in BASE], 3, True),
            ('Triplet 三颗 WBF', [caches[r] for r in TRIP], 3, True),
            ('混合六颗 WBF（标准惩罚）', [caches[r] for r in ALL6], 6, True),
            ('混合六颗 WBF（无票数惩罚）', [caches[r] for r in ALL6], 6, False),
        ]
        for name, cs, n, pen in cases:
            met = E.run_val(BASE[0], augment=False, provider=provider(cs, 0.55, n, pen),
                            tag=f'{tag}_mix_{n}_{int(pen)}')
            pc = met['per_class']
            rows.append({'dataset': tag, 'case': name, 'n_models': n, 'penalty': int(pen),
                         'mAP50': met['mAP50'], 'mAP50-95': met['mAP50-95'],
                         'P': met['P'], 'R': met['R'],
                         'break': pc.get('break', {}).get('ap50', float('nan')),
                         'thunder': pc.get('thunderbolt', {}).get('ap50', float('nan')),
                         'wear': pc.get('wear', {}).get('ap50', float('nan'))})
            print(f'  [{name:<26}] mAP50={met["mAP50"]:.4f}  mAP50-95={met["mAP50-95"]:.4f}  '
                  f'P={met["P"]:.4f} R={met["R"]:.4f}  '
                  + '  '.join(f'{k}={v["ap50"]:.3f}' for k, v in pc.items()))

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    hdr = ['dataset', 'case', 'n_models', 'penalty', 'mAP50', 'mAP50-95', 'P', 'R',
           'break', 'thunder', 'wear']
    with OUT_TSV.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(hdr)
        for r in rows:
            w.writerow([r[k] if isinstance(r[k], (str, int)) else f'{r[k]:.4f}' for k in hdr])
    print(f'\n已写 {OUT_TSV}')

    print('\n' + '=' * 96)
    print(f'{"数据集":<10}{"配置":<28}{"mAP50":>9}{"mAP50-95":>10}{"break":>9}{"thunder":>9}{"wear":>9}')
    print('-' * 96)
    for r in rows:
        print(f'{r["dataset"]:<10}{r["case"]:<28}{r["mAP50"]*100:>9.2f}{r["mAP50-95"]*100:>10.2f}'
              f'{r["break"]:>9.4f}{r["thunder"]:>9.4f}{r["wear"]:>9.4f}')
    print('=' * 96)


if __name__ == '__main__':
    main()
