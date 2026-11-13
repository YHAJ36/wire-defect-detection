# -*- coding: utf-8 -*-
"""同一组权重、只改**推理分辨率**，在 203 图段级 test 上做配对比较。

动机（2026-10-29 深夜）：C 组 multi_scale 在第 21 轮 CUDA OOM 死掉，整条数据侧扰动
路线断掉。转向"推理侧能不能救"时发现一个反直觉结果：**896 训练的模型，用 640 推理
反而比 896 推理高**。本脚本把它量清楚，并给出配对置信区间。

为什么这个比较特别干净：权重完全相同、训练完全相同，唯一变量是推理时的 imgsz。
训练侧的种子方差完全不存在，所以配对 bootstrap 里剩下的只有"图像抽样"这一项。

用法
    # 先生成高/低分辨率的决策缓存（cache_perturb_tp.py 支持 --imgsz/--suffix）
    D:/yolo-env/Scripts/python.exe code/tools/eval/cache_perturb_tp.py \
        --imgsz 640 --suffix _r640 nf_v11_noleak_s42 nf_v11_noleak_s43 nf_v11_noleak_s44
    # 再配对
    D:/yolo-env/Scripts/python.exe code/tools/eval/bootstrap_paired_res.py --suffix-b _r640 --tag 640
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from ultralytics.utils.metrics import ap_per_class  # noqa: E402

ROOT = Path('D:/2')
CACHE = ROOT / 'exp_upscale/logs/tp_cache_noleak'
ASSIGN = ROOT / 'dataset_noleak/assign.json'
NAMES = ['break', 'thunderbolt', 'wear']


def runs_of(tag, seeds):
    return [f'nf_v11_noleak_{tag + "_" if tag else ""}s{s}' for s in seeds]


def load_tp(path):
    z = np.load(path, allow_pickle=True)
    keys = [str(k) for k in z['__keys__']]
    return {k: (z['tp50'][i], z['dconf'][i], z['pcls'][i], z['gcls'][i])
            for i, k in enumerate(keys)}


def ap50(d, idx, cls=None):
    if not idx:
        return float('nan')
    tp = np.concatenate([d[i][0] for i in idx])[:, None]
    cf = np.concatenate([d[i][1] for i in idx])
    pc = np.concatenate([d[i][2] for i in idx])
    gc = np.concatenate([d[i][3] for i in idx])
    if len(gc) == 0:
        return float('nan')
    res = ap_per_class(tp, cf, pc, gc)
    a = np.asarray(res[5])
    if a.ndim == 2:
        a = a[:, 0]
    return float(a[cls]) if cls is not None and cls < len(a) else float(np.nanmean(a))


def ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v)
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='', help='训练组标签，base 留空')
    ap.add_argument('--suffix-a', default='', help='基准分辨率缓存后缀（默认 = 896）')
    ap.add_argument('--suffix-b', default='_r640', help='对照分辨率缓存后缀')
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--B', type=int, default=3000)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    seeds = tuple(int(x) for x in args.seeds.split(',') if x.strip())
    runs = runs_of(args.tag, seeds)
    feats, stems0 = {}, None
    for r in runs:
        for sfx, key in ((args.suffix_a, 'a'), (args.suffix_b, 'b')):
            p = CACHE / f'{r}{sfx}.npz'
            if not p.exists():
                raise SystemExit(f'缺缓存 {p}')
            d = load_tp(p)
            ks = sorted(d)
            if stems0 is None:
                stems0 = ks
            elif ks != stems0:
                raise SystemExit(f'{p.name} 图列表不一致')
            feats[(key, r)] = [d[s] for s in stems0]
    stems = stems0
    n = len(stems)
    ALL = list(range(n))

    seg_of = json.load(open(ASSIGN, encoding='utf-8'))['seg_of']
    segs = {}
    for i, s in enumerate(stems):
        segs.setdefault(seg_of.get(s + '.jpg', seg_of.get(s, -1)), []).append(i)
    seg_list = list(segs.values())

    print(f'test 图 {n}  GT {sum(len(f[3]) for f in feats[("a", runs[0])])}  段 {len(seg_list)}')
    pts, deltas = {}, []
    for r in runs:
        a, b = ap50(feats[('a', r)], ALL), ap50(feats[('b', r)], ALL)
        pts[r] = (a, b)
        deltas.append(b - a)
        print(f'  {r[-3:]}  {args.suffix_a or "_896"}: {a:.4f}   {args.suffix_b}: {b:.4f}   '
              f'Δ={(b - a) * 100:+.2f} pt')
    ma = np.mean([pts[r][0] for r in runs]) * 100
    mb = np.mean([pts[r][1] for r in runs]) * 100
    print(f'  均值  {ma:.2f} → {mb:.2f}   Δ={mb - ma:+.2f} pt   '
          f'（三种子 Δ: {", ".join(f"{d*100:+.2f}" for d in deltas)}）')

    rng = np.random.default_rng(0)
    obs = np.mean(deltas) * 100
    rows = []
    for name, sampler in (('图级配对', 'img'), ('段级 cluster 配对', 'seg')):
        ds = []
        for _ in range(args.B):
            if sampler == 'img':
                idx = list(rng.integers(0, n, n))
            else:
                pick = rng.integers(0, len(seg_list), len(seg_list))
                idx = [i for k in pick for i in seg_list[k]]
                if not idx:
                    continue
            ds.append(np.mean([ap50(feats[('b', r)], idx) - ap50(feats[('a', r)], idx)
                               for r in runs]))
        lo, hi = ci(ds)
        verdict = '跨零 → 不可判' if lo <= 0 <= hi else ('显著为正' if lo > 0 else '显著为负')
        print(f'  {name:<18} Δ={obs:+.2f}  95%CI [{lo*100:+.2f}, {hi*100:+.2f}]  {verdict}')
        rows.append({'scope': name, 'delta_pt': round(obs, 3), 'ci_lo_pt': round(lo * 100, 3),
                     'ci_hi_pt': round(hi * 100, 3), 'verdict': verdict})

    print('\n[分类别 · 图级配对]')
    for c, nm in enumerate(NAMES):
        dv = np.array([ap50(feats[('b', r)], ALL, c) - ap50(feats[('a', r)], ALL, c)
                       for r in runs]) * 100
        ds = []
        for _ in range(min(args.B, 2000)):
            idx = list(rng.integers(0, n, n))
            ds.append(np.mean([ap50(feats[('b', r)], idx, c) - ap50(feats[('a', r)], idx, c)
                               for r in runs]) * 100)
        lo, hi = ci(ds)
        verdict = '跨零' if lo <= 0 <= hi else ('显著为正' if lo > 0 else '显著为负')
        print(f'  {nm:<12} Δ={dv.mean():+6.2f}  95%CI [{lo:+.2f}, {hi:+.2f}]  {verdict}')
        rows.append({'scope': f'图级配对/{nm}', 'delta_pt': round(dv.mean(), 3),
                     'ci_lo_pt': round(lo, 3), 'ci_hi_pt': round(hi, 3), 'verdict': verdict})

    out = Path(args.out) if args.out else ROOT / f'exp_upscale/logs/bootstrap_paired_res{args.suffix_b}.tsv'
    with out.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter='\t')
        wr.writeheader()
        wr.writerows(rows)
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
