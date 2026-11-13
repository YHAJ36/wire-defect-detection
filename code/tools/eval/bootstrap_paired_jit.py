# -*- coding: utf-8 -*-
"""A 组（GT 抖动）vs 基线：203 图段级 test 上的**配对** bootstrap。

为什么必须配对：jit08 与 base 是两组独立训练，但评估在**同一张卷**上，
同一张图的难易程度对两者是同一个随机因素。配对重采样（同一批图同时喂给
两个配置）把这个共同方差消掉，CI 从单侧的 10.4 pt 压到约 3.3 pt
（见 bootstrap_ci_noleak.tsv）。

三层判读
    1. 每种子配对 Δ        —— 同种子、同图，看方向是否一致
    2. 图级配对 Δ（三种子） —— 重采样图，三种子的 Δ 一起算，看均值是否跨零
    3. 段级 cluster 配对 Δ  —— 按视觉段整段重采样，同段相邻帧相关，图级会低估

决策量（tp/conf/pred_cls/gt_cls）全部取自 ultralytics 的
`DetectionValidator._process_batch`，缓存由 cache_perturb_tp.py 生成，
与 base 的缓存格式逐字段一致。

用法
    D:/yolo-env/Scripts/python.exe code/tools/eval/bootstrap_paired_jit.py --B 4000
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


def runs_of(tag, seeds=(42, 43, 44)):
    """tag='' → 基线；tag='jit08' / 'ms' → nf_v11_noleak_{tag}_s{seed}。"""
    return [f'nf_v11_noleak_{tag + "_" if tag else ""}s{s}' for s in seeds]


def load_tp(path):
    z = np.load(path, allow_pickle=True)
    keys = [str(k) for k in z['__keys__']]
    return {k: (z['tp50'][i], z['dconf'][i], z['pcls'][i], z['gcls'][i])
            for i, k in enumerate(keys)}


def ap50(d, idx, cls=None):
    """宏平均 AP@0.5（cls=None 时三类平均）。idx 可含重复下标。"""
    if not idx:
        return float('nan')
    tp = np.concatenate([d[i][0] for i in idx])[:, None]
    cf = np.concatenate([d[i][1] for i in idx])
    pc = np.concatenate([d[i][2] for i in idx])
    gc = np.concatenate([d[i][3] for i in idx])
    if len(gc) == 0:
        return float('nan')
    res = ap_per_class(tp, cf, pc, gc)
    ap = np.asarray(res[5])          # 8.3.180：ap 在 index 5
    if ap.ndim == 2:
        ap = ap[:, 0]
    if cls is not None:
        return float(ap[cls]) if cls < len(ap) else float('nan')
    return float(np.nanmean(ap))


def ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v)
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--target', default='jit08',
                    help='被检验的扰动组标签：jit08 / ms / drop（拼成 nf_v11_noleak_<tag>_s<seed>）')
    ap.add_argument('--seeds', default='42,43,44', help='逗号分隔的种子，缺哪个就先跳过')
    ap.add_argument('--out', default='', help='输出 TSV，默认 bootstrap_paired_<target>.tsv')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    seeds = tuple(int(x) for x in args.seeds.split(',') if x.strip())
    target_runs = runs_of(args.target, seeds)
    base_runs = runs_of('', seeds)
    # 只保留缓存齐全的种子（训练还没跑完时先算已完成的）
    keep = [(b, t) for b, t in zip(base_runs, target_runs)
            if (CACHE / f'{b}.npz').exists() and (CACHE / f'{t}.npz').exists()]
    if not keep:
        raise SystemExit(f'没有任何一对缓存齐全：{target_runs}')
    if len(keep) < len(seeds):
        miss = [t for b, t in zip(base_runs, target_runs) if (b, t) not in keep]
        print(f'⚠ 只算 {len(keep)} 个种子（缺缓存：{", ".join(miss)}）')
    BASE_ = [b for b, _ in keep]
    JIT_ = [t for _, t in keep]
    out_path = Path(args.out) if args.out else ROOT / f'exp_upscale/logs/bootstrap_paired_{args.target}.tsv'

    feats, stems0 = {}, None
    for grp, runs in (('base', BASE_), ('jit', JIT_)):
        for r in runs:
            p = CACHE / f'{r}.npz'
            if not p.exists():
                raise SystemExit(f'缺缓存 {p}')
            d = load_tp(p)
            keys = sorted(d)
            if stems0 is None:
                stems0 = keys
            elif keys != stems0:
                raise SystemExit(f'{p.name} 的图列表与基准不一致')
            feats[(grp, r)] = [d[s] for s in stems0]
    stems = stems0
    n = len(stems)
    ALL = list(range(n))

    # 段 id
    seg_of = json.load(open(ASSIGN, encoding='utf-8'))['seg_of']
    seg_ids = [seg_of.get(s + '.jpg', seg_of.get(s, -1)) for s in stems]
    segs = {}
    for i, g in enumerate(seg_ids):
        segs.setdefault(g, []).append(i)
    seg_list = [v for v in segs.values()]
    print(f'test 图 {n}   GT 框 {sum(len(f[3]) for f in feats[("base", BASE_[0])])}   '
          f'视觉段 {len(seg_list)}（最大段 {max(len(v) for v in seg_list)} 图）')

    # 1) 点估计
    print('\n[点估计]')
    point = {}
    for grp, runs in (('base', BASE_), ('jit', JIT_)):
        vals = []
        for r in runs:
            v = ap50(feats[(grp, r)], ALL)
            point[(grp, r)] = v
            vals.append(v)
            print(f'  {grp:<5}{r[-3:]}  mAP50={v:.4f}')
        print(f'  {grp:<5}均值  {np.mean(vals):.4f}   种子 span={max(vals) - min(vals):.4f}')

    deltas = [point[('jit', j)] - point[('base', b)] for b, j in zip(BASE_, JIT_)]
    print('\n[每种子配对 Δ（jit − base，同一张卷）]')
    for b, j, dv in zip(BASE_, JIT_, deltas):
        print(f'  {b[-3:]} vs {j[-3:]}   Δ={dv * 100:+.2f} pt')
    print(f'  均值 Δ = {np.mean(deltas) * 100:+.2f} pt  '
          f'(base {np.mean([point[("base", b)] for b in BASE_]) * 100:.2f} → '
          f'jit {np.mean([point[("jit", j)] for j in JIT_]) * 100:.2f})')

    # 2) 配对 bootstrap
    rng = np.random.default_rng(args.seed)
    d_img, d_seg = [], []
    for _ in range(args.B):
        idx = list(rng.integers(0, n, n))
        dv = np.mean([ap50(feats[('jit', j)], idx) - ap50(feats[('base', b)], idx)
                      for b, j in zip(BASE_, JIT_)])
        d_img.append(dv)
    for _ in range(args.B):
        pick = rng.integers(0, len(seg_list), len(seg_list))
        idx = [i for k in pick for i in seg_list[k]]
        if not idx:
            continue
        dv = np.mean([ap50(feats[('jit', j)], idx) - ap50(feats[('base', b)], idx)
                      for b, j in zip(BASE_, JIT_)])
        d_seg.append(dv)

    print('\n[配对 bootstrap  Δ = jit − base，单位 pt]')
    rows = []
    obs = np.mean(deltas) * 100
    for tag, v in (('图级配对', d_img), ('段级 cluster 配对', d_seg)):
        lo, hi = ci(v)
        cross = '跨零 → 不可判' if lo <= 0 <= hi else ('显著为负' if hi < 0 else '显著为正')
        print(f'  {tag:<18} Δ={obs:+.2f}  95%CI [{lo * 100:+.2f}, {hi * 100:+.2f}]  {cross}')
        rows.append({'scope': tag, 'delta_pt': round(obs, 3),
                     'ci_lo_pt': round(lo * 100, 3), 'ci_hi_pt': round(hi * 100, 3),
                     'verdict': cross})

    # 3) 分类别（图级配对）
    print('\n[分类别：图级配对 Δ，单位 pt]')
    for ci_, nm in enumerate(NAMES):
        dv = np.array([ap50(feats[('jit', j)], ALL, ci_) - ap50(feats[('base', b)], ALL, ci_)
                       for b, j in zip(BASE_, JIT_)]) * 100
        bs = []
        for _ in range(min(args.B, 2000)):
            idx = list(rng.integers(0, n, n))
            bs.append(np.mean([ap50(feats[('jit', j)], idx, ci_) - ap50(feats[('base', b)], idx, ci_)
                               for b, j in zip(BASE_, JIT_)]) * 100)
        lo, hi = ci(bs)
        cross = '跨零' if lo <= 0 <= hi else ('显著为负' if hi < 0 else '显著为正')
        print(f'  {nm:<12} Δ={dv.mean():+6.2f}  95%CI [{lo:+.2f}, {hi:+.2f}]  {cross}')
        rows.append({'scope': f'图级配对/{nm}', 'delta_pt': round(dv.mean(), 3),
                     'ci_lo_pt': round(lo, 3), 'ci_hi_pt': round(hi, 3), 'verdict': cross})

    with out_path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['scope', 'delta_pt', 'ci_lo_pt', 'ci_hi_pt', 'verdict'],
                            delimiter='\t')
        wr.writeheader()
        wr.writerows(rows)
    print(f'\n→ {out_path}')


if __name__ == '__main__':
    main()
