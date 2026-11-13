# -*- coding: utf-8 -*-
"""段级无泄漏 test 203 图上的「积木组合」对照：同族 / 跨族 / 3:1 / 2:2。

目的
----
回答一个具体问题：把 base 族（无注意力）和 Triplet 族（注意力）当成两块积木，
合起来会不会比各用一族更好？也就是"组合增益是否超加"。

为什么在 203 图段级卷上做
------------------------
老的 179 图卷和 clean 100 图卷已经跑过同一组对照（eval_wbf_mixed.tsv）：
  old179：base 93.21 / Triplet 93.29 / 六颗 93.29  → 混合 = 取最大值，零超加
  clean100：base 90.63 / Triplet 92.04 / 六颗 91.82 → 混合反而掉 0.22
但那两张卷一张带泄漏、一张只有 100 图。段级 203 图是现行 v3 口径，
且噪声地板已经标定（1.60 pt），所以只有在这张卷上才能判"增益是否真实"。

口径
----
完全复用 eval_wbf_tta.py 的注入点（DetectionValidator.update_metrics），
imgsz=896、NMS iou=0.5、conf=0.001、max_det=300，与 eval_noleak_final.tsv 逐位一致。
单模型 raw 值不在这里重算，直接读 eval_noleak_final.tsv（75.91 / 74.93 / 73.17 与
77.23 / 73.64 / 71.28）。

只读：不改权重、不动数据集、不写报告。

用法
    D:/yolo-env/Scripts/python.exe code/tools/eval/eval_wbf_mixed_noleak.py
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ['NOLEAK_YAML'] = 'D:/2/dataset_noleak/data.yaml'
os.environ['NOLEAK_CACHE'] = 'D:/2/exp_upscale/logs/wbf_cache_noleak'
os.environ['NOLEAK_SKIP_TTA'] = '1'
sys.path.insert(0, 'D:/2/code/tools/eval')
import eval_wbf_tta as E  # noqa: E402  导入即完成 update_metrics 打桩

BASE = ['nf_v11_noleak_s42', 'nf_v11_noleak_s43', 'nf_v11_noleak_s44']
TRIP = ['att_v11_noleak_triplet_s42', 'att_v11_noleak_triplet_s43',
        'att_v11_noleak_triplet_s44']
CACHE_DIR = Path('D:/2/exp_upscale/logs/wbf_cache_noleak')
OUT_TSV = Path('D:/2/exp_upscale/logs/eval_wbf_mixed_noleak.tsv')


# ---------------------------------------------------------------- WBF（与 eval_wbf_mixed.py 同一实现）
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
    """score = Σs/N · min(|cluster|,N)/N（penalty=True，ZFTurbo 语义）或 Σs/N。"""
    all_b = np.concatenate(boxes_list, 0)
    all_s = np.concatenate(scores_list, 0)
    all_c = np.concatenate(labels_list, 0)
    if len(all_s) == 0:
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                np.zeros((0,), np.int64))
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
            bw = (b[sel] * w[:, None]).sum(0) / max(w.sum(), 1e-9)
            sc = w.sum() / n_models
            if penalty:
                sc = sc * (min(len(sel), n_models) / n_models)
            ob.append(bw); os_.append(sc); oc.append(c)
            alive[sel] = False
    return (np.array(ob, np.float32), np.array(os_, np.float32),
            np.array(oc, np.int64))


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
    print(f'data={E.DATA}  split={E.SPLIT}  imgsz={E.IMGSZ}  iou={E.NMS_IOU}  conf={E.CONF}')

    caches = {}
    for run in BASE + TRIP:
        p = CACHE_DIR / f'{run}_raw.npz'
        assert p.exists(), f'缺缓存 {p}'
        caches[run] = load_npz(p)
        print(f'  cache {run:<28} imgs={len(caches[run])}')

    cases = [
        ('同族 base 三颗', [caches[r] for r in BASE], 3, True),
        ('同族 Triplet 三颗', [caches[r] for r in TRIP], 3, True),
        ('跨族 六颗（标准惩罚）', [caches[r] for r in BASE + TRIP], 6, True),
        ('跨族 六颗（无票数惩罚）', [caches[r] for r in BASE + TRIP], 6, False),
        ('跨族 3:1 base_s42+Triplet×3',
         [caches[BASE[0]]] + [caches[r] for r in TRIP], 4, True),
        ('跨族 2:2 base_s42/43+Triplet_s42/43',
         [caches[BASE[0]], caches[BASE[1]], caches[TRIP[0]], caches[TRIP[1]]], 4, True),
    ]

    rows = []
    # carrier 用 base s42 的权重跑 val，预测值由 provider 整体替换，权重本身不影响结果
    for name, cs, n, pen in cases:
        met = E.run_val(BASE[0], augment=False, provider=provider(cs, 0.55, n, pen),
                        tag=f'noleak203_mix_{n}_{int(pen)}_{abs(hash(name)) % 9999}')
        pc = met['per_class']
        row = {'case': name, 'n_models': n, 'penalty': int(pen),
               'mAP50': met['mAP50'], 'mAP50-95': met['mAP50-95'],
               'P': met['P'], 'R': met['R'],
               'break': pc.get('break', {}).get('ap50', float('nan')),
               'thunder': pc.get('thunderbolt', {}).get('ap50', float('nan')),
               'wear': pc.get('wear', {}).get('ap50', float('nan'))}
        rows.append(row)
        print(f'  [{name:<32}] mAP50={row["mAP50"] * 100:6.2f}  mAP50-95={row["mAP50-95"] * 100:6.2f}  '
              f'P={row["P"]:6.4f} R={row["R"]:6.4f}  '
              f'break={row["break"]:.4f} thunder={row["thunder"]:.4f} wear={row["wear"]:.4f}')

    hdr = ['case', 'n_models', 'penalty', 'mAP50', 'mAP50-95', 'P', 'R',
           'break', 'thunder', 'wear']
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(hdr)
        for r in rows:
            w.writerow([r[k] if isinstance(r[k], (str, int)) else f'{r[k]:.6f}' for k in hdr])
    print(f'\n已写 {OUT_TSV}')

    print('\n' + '=' * 92)
    print(f'{"配置":<34}{"mAP50":>9}{"Δ50":>8}{"mAP50-95":>11}{"Δ50-95":>9}')
    print('-' * 92)
    best = max(r['mAP50'] for r in rows if r['n_models'] == 3)
    for r in rows:
        print(f'{r["case"]:<34}{r["mAP50"] * 100:>9.2f}'
              f'{(r["mAP50"] - best) * 100:>+8.2f}{r["mAP50-95"] * 100:>11.2f}'
              f'{(r["mAP50-95"] - max(x["mAP50-95"] for x in rows if x["n_models"] == 3)) * 100:>+9.2f}')
    print('=' * 92)
    print(f'参照：同族三颗的最好成绩 = {best * 100:.2f}；噪声地板 1.60 pt。')


if __name__ == '__main__':
    main()
