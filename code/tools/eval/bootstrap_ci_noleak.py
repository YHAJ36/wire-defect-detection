# -*- coding: utf-8 -*-
"""给段级 test 卷（203 图）上的 mAP50 算 bootstrap 置信区间。

为什么需要：v3 口径的主结论是「跨段泛化 74.7% ~ 89.2% 区间」，但 74.67 这个数
本身只来自 203 张图 / 86 个视觉段。**在谈论任何 <1.6 pt 的差异之前，必须先知道
这把尺子本身的刻度**。本脚本只读磁盘上已有的决策缓存，不跑训练、不改任何权重。

三个置信区间，回答三个不同问题
    1. 图级 bootstrap      ：203 张图若换一批同等图，mAP50 会落在哪  —— 测试集抽样误差
    2. 段级 cluster bootstrap：按视觉段整段重采样                  —— 同段内相关的真实误差
       （图级会低估，因为同段相邻帧高度相关，等效样本数远小于 203）
    3. 配对 bootstrap      ：两种配置在同一张卷上的 Δ 是否跨零      —— 增益是否可判

口径：**决策量（tp / conf / pred_cls / gt_cls）直接取自 ultralytics 自己的
`DetectionValidator._process_batch`**，不是自己实现 IoU 匹配。原因是自己实现时
在 IoU≈0.5 的边界上会翻转，实测与 TSV 差 0.15~2.35 pt 且方向不定 —— 那种数字
算出来的 CI 没有意义。AP 也用框架的 `ap_per_class`。脚本开头先复算一遍并与
`eval_noleak_final.tsv` 对账，对不上直接退出。

用法
    D:/yolo-env/Scripts/python.exe code/tools/eval/bootstrap_ci_noleak.py
    ... --B 4000 --seed 7
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from ultralytics.utils.metrics import ap_per_class  # noqa: E402

ROOT = Path('D:/2')
CACHE = ROOT / 'exp_upscale/logs/tp_cache_noleak'
TSV = ROOT / 'exp_upscale/logs/eval_noleak_final.tsv'
ASSIGN = ROOT / 'dataset_noleak/assign.json'
OUT = ROOT / 'exp_upscale/logs/bootstrap_ci_noleak.tsv'

SEEDS = ('s42', 's43', 's44')
FAMILIES = {
    'base':    'nf_v11_noleak_{s}.npz',
    'triplet': 'att_v11_noleak_triplet_{s}.npz',
}
# WBF 融合结果（由 regen_tp_wbf_noleak.py 生成），单配置、无种子维度
WBF = {'wbf_base': 'wbf_base.npz', 'wbf_triplet': 'wbf_triplet.npz',
       'wbf_mix6': 'wbf_mix6.npz'}
# TSV 里 单模型 s42/s43/s44 的 mAP50，用于对账
EXPECT = {'base': [0.7591, 0.7493, 0.7317], 'triplet': [0.7723, 0.7364, 0.7128]}
EXPECT_WBF = {'wbf_base': 0.7776, 'wbf_triplet': 0.7724, 'wbf_mix6': 0.7736}


def load_tp(path):
    """返回 (stems, {stem: (tp50, conf, pred_cls, gt_cls)})。"""
    z = np.load(path, allow_pickle=True)
    keys = [str(k) for k in z['__keys__']]
    d = {k: (z['tp50'][i], z['dconf'][i], z['pcls'][i], z['gcls'][i])
         for i, k in enumerate(keys)}
    return keys, d


def map50(d, idx):
    """3 类宏平均 AP@0.5。idx 为图片下标序列（可重复），直接交给框架的 ap_per_class。"""
    tp = np.concatenate([d[i][0] for i in idx])[:, None]
    cf = np.concatenate([d[i][1] for i in idx])
    pc = np.concatenate([d[i][2] for i in idx])
    gc = np.concatenate([d[i][3] for i in idx])
    if len(gc) == 0:
        return float('nan')
    res = ap_per_class(tp, cf, pc, gc)
    # 8.3.180 返回 12 项：(tp, fp, p, r, f1, ap, unique_classes, p_curve, r_curve,
    # f1_curve, x, prec_values) —— ap 在 index 5，不是 index 3。
    ap = np.asarray(res[5])
    if ap.ndim == 2:
        ap = ap[:, 0]
    return float(ap.mean())


def ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v)
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def read_tsv_expect():
    got = {}
    with TSV.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['group'] in ('单模型 s42', '单模型 s43', '单模型 s44'):
                got[(r['family'], r['group'][-3:])] = float(r['mAP50'])
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    # 1) 装载（六份缓存的图列表必须完全一致）
    feats, stems0 = {}, None
    for fam, tmpl in FAMILIES.items():
        for s in SEEDS:
            p = CACHE / tmpl.format(s=s)
            if not p.exists():
                raise SystemExit(f'缺缓存 {p}（先跑 regen_cache_orig.py）')
            keys, d = load_tp(p)
            if stems0 is None:
                stems0 = sorted(keys)
            elif sorted(keys) != stems0:
                raise SystemExit(f'{p.name} 的图列表与 base s42 不一致')
            feats[(fam, s)] = d
    for name, fn in WBF.items():
        p = CACHE / fn
        if not p.exists():
            print(f'  （跳过 {name}：缺 {p.name}）')
            continue
        keys, d = load_tp(p)
        if sorted(keys) != stems0:
            raise SystemExit(f'{p.name} 的图列表与 base s42 不一致')
        feats[(name, None)] = d
    stems = stems0
    # 统一成「按 stems 顺序的下标列表」，bootstrap 时用下标访问
    feats = {k: [v[s] for s in stems] for k, v in feats.items()}
    ALL = list(range(len(stems)))
    print(f'test 图 {len(stems)}   GT 框 {sum(len(f[3]) for f in feats[("base","s42")])}')

    point = {}
    for k in feats:
        point[k] = map50(feats[k], ALL)
        fam, s = k
        print(f'  {fam:<12}{(s or ""):<5} mAP50={point[k]:.4f}')

    # 2) 对账
    exp = read_tsv_expect()
    bad = []
    for fam in FAMILIES:
        for s in SEEDS:
            e = exp.get((fam, s))
            if e is None:
                bad.append((fam, s, 'TSV 无此行'))
            elif abs(point[(fam, s)] - e) > 0.00015:
                bad.append((fam, s, f'复算 {point[(fam,s)]:.4f} vs TSV {e:.4f}'))
    for name, e in EXPECT_WBF.items():
        k = (name, None)
        if k not in point:
            continue
        if abs(point[k] - e) > 0.00015:
            bad.append((name, f'复算 {point[k]:.4f} vs TSV {e:.4f}'))
    if bad:
        print('\n[对账失败] 复算与 eval_noleak_final.tsv 不一致，拒绝输出 CI：')
        for b in bad:
            print('  ', b)
        raise SystemExit(2)
    print('  对账通过：复算与 eval_noleak_final.tsv 逐位一致\n')

    # 3) 段分组
    seg_of = json.load(open(ASSIGN, encoding='utf-8'))['seg_of']
    seg2idx = {}
    for i, s in enumerate(stems):
        seg2idx.setdefault(seg_of[s + '.jpg'], []).append(i)
    groups = list(seg2idx.values())
    print(f'段级分组：{len(groups)} 段，最大 {max(len(g) for g in groups)} 图，'
          f'中位 {int(np.median([len(g) for g in groups]))} 图\n')

    rng = np.random.default_rng(args.seed)
    n = len(stems)
    B = args.B

    def boot_image(fn):
        return np.array([fn(rng.integers(0, n, n)) for _ in range(B)])

    def boot_cluster(fn):
        g = len(groups)
        return np.array([fn(np.concatenate([groups[i] for i in rng.integers(0, g, g)]))
                         for _ in range(B)])

    # 4) 单配置 CI（三种子均值 / WBF 融合）
    rows = []

    def mean_stat(keys):
        ds = [feats[k] for k in keys]
        return lambda idx: float(np.nanmean([map50(d, idx) for d in ds]))

    BASE_KEYS = [('base', s) for s in SEEDS]
    TRIP_KEYS = [('triplet', s) for s in SEEDS]
    groups_of_keys = [
        ('base 三种子均值', BASE_KEYS),
        ('triplet 三种子均值', TRIP_KEYS),
    ] + [(name, [(name, None)]) for name in WBF if (name, None) in feats]

    for name, keys in groups_of_keys:
        fn = mean_stat(keys)
        pt = float(np.nanmean([point[k] for k in keys]))
        bi, bc = boot_image(fn), boot_cluster(fn)
        li, hi = ci(bi)
        lc, hc = ci(bc)
        rows.append([name, pt, li, hi, lc, hc, hi - li, hc - lc])
        print(f'{name:<22} {pt:.4f}  图级 [{li:.4f},{hi:.4f}] w={hi-li:.4f}    '
              f'段级 [{lc:.4f},{hc:.4f}] w={hc-lc:.4f}')

    # 5) 配对差（同一批图，配对重采样；共同项被抵掉，通常比单侧 CI 窄得多）
    def pair_diff(name, keys_a, label_a, keys_b, label_b):
        fa = [feats[k] for k in keys_a]
        fb = [feats[k] for k in keys_b]
        pt = float(np.nanmean([point[k] for k in keys_a])
                   - np.nanmean([point[k] for k in keys_b]))

        def fn(idx):
            return (np.nanmean([map50(d, idx) for d in fa])
                    - np.nanmean([map50(d, idx) for d in fb]))

        bi, bc = boot_image(fn), boot_cluster(fn)
        li, hi = ci(bi)
        lc, hc = ci(bc)
        rows.append([name, pt, li, hi, lc, hc, hi - li, hc - lc])
        sig = '可判' if (li > 0 or hi < 0) else '跨零→不可判'
        print(f'{name:<22} {pt:+.4f}  图级 [{li:+.4f},{hi:+.4f}] w={hi-li:.4f}    '
              f'段级 [{lc:+.4f},{hc:+.4f}]    {sig}(图级)')

    pair_diff('Δ base−triplet', BASE_KEYS, 'base', TRIP_KEYS, 'triplet')
    for name in ('wbf_base', 'wbf_triplet', 'wbf_mix6'):
        if (name, None) in feats:
            pair_diff(f'Δ {name}−base均值', [(name, None)], name, BASE_KEYS, 'base均值')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['stat', 'point', 'lo_image', 'hi_image', 'lo_cluster', 'hi_cluster',
                    'width_image', 'width_cluster', 'B', 'seed'])
        for r in rows:
            w.writerow([r[0]] + [f'{x:.4f}' for x in r[1:]] + [B, args.seed])
    print(f'\n已写 {OUT}')


if __name__ == '__main__':
    main()
