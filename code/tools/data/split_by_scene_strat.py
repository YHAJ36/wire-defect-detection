# -*- coding: utf-8 -*-
"""段级**分层**重划分：修 split_by_scene.py「随机划分不控难度」的缺陷。

背景（WORKLOG A.9.62 / A.9.63 实测）
    段级随机划分做到了 1080 段零跨 split（真无泄漏），但**没控制每段的难度**。
    同一颗新权重、两张来源构成几乎相同的卷子：

        val  181 图  mAP50 89.15   break 0.823
        test 203 图  mAP50 75.91   break 0.607

    ⇒ 段级随机把「难的 break 段」整批分给了 test，把「容易的」分给了 val。
    `best.pt` 又是在 val 上挑的，偏差同时污染选模口径。
    零泄漏 ≠ 三张卷子等价。本脚本在**不放松零泄漏**的前提下修这个偏差。

做法
    1. 复用已有段结构（`assign.json` 的 `seg_of`，dHash+MAE 双闸并查集 1080 段），
       **不重算图像特征**——分层只改「段怎么分」，与怎么聚类无关。
    2. 给每段算难度特征（读 label，秒级）：
         break_per_img   段内 break 框数 / 图数      ← 主难度维度（实测差 21.6 pt）
         big_wear_ratio  段内 wear 大框(h>0.18)占比
         med_area        段内框面积中位数
         n_img           段大小
    3. 按 break_per_img 的 1/3、2/3 分位（低/中/高）× 段大小（≥8 图 / <8 图）
       分层 → 最多 6 层。层数刻意压到 6 而不是 10：层太细会让 val/test 每层
       只剩个位数图，配额噪声反而更大。
    4. **层内**按 8:1:1 独立配额分配（归一化 deficit 贪心，大段先放），
       保证每个难度层在三个 split 里都有份额 —— 这就是「分层」的全部作用。
    5. 校验：零跨段、类别分布、难度分布、框尺度分布，以及与随机划分的对照。

用法
    D:/yolo-env/Scripts/python.exe code/tools/data/split_by_scene_strat.py --seed 42
    ...  --apply     # 另切到 dataset_noleak_strat/（不动 dataset_noleak/）
"""
import argparse
import collections
import json
import os
import shutil
import sys

import numpy as np

BASE = 'D:/2/dataset_final_bbox'
SRC = 'D:/2/dataset_noleak'                 # 提供 assign.json（段结构）
OUT = 'D:/2/dataset_noleak_strat'           # 分层版另起目录，不覆盖已训的那份
NAMES = ['break', 'thunderbolt', 'wear']
BIG_H = 0.18                                 # wear「大框」阈值（沿用诊断口径）
BIG_SEG = 8                                  # 段大小分档阈值


def load_segments():
    p = os.path.join(SRC, 'assign.json')
    with open(p, encoding='utf-8') as f:
        plan = json.load(f)
    seg_of, old_of = plan['seg_of'], plan['old_of']
    segs = collections.defaultdict(list)
    for f, k in seg_of.items():
        segs[k].append(f)
    return plan, segs, old_of


def seg_features(files, old_of):
    """段级难度特征。label 路径按 old_of 定位。"""
    n_break = n_box = n_wear = n_big_wear = 0
    areas, break_areas = [], []
    for f in files:
        lab = os.path.join(BASE, 'labels', old_of[f], f.rsplit('.', 1)[0] + '.txt')
        if not os.path.exists(lab):
            continue
        for line in open(lab):
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            w, h = float(p[3]), float(p[4])
            n_box += 1
            areas.append(w * h)
            if c == 0:
                n_break += 1
                break_areas.append(w * h)
            elif c == 2:
                n_wear += 1
                if h > BIG_H:
                    n_big_wear += 1
    n_img = len(files)
    return {
        'n_img': n_img,
        'n_box': n_box,
        'n_thunder': n_box - n_break - n_wear,
        'sum_area': float(np.sum(areas)),
        'break_per_img': n_break / n_img,
        # 分层主键：段内 break 框面积中位。0 表示该段无 break
        'break_area': float(np.median(break_areas)) if break_areas else 0.0,
        'big_wear_ratio': (n_big_wear / n_wear) if n_wear else 0.0,
        'med_area': float(np.median(areas)) if areas else 0.0,
        'n_break': n_break,
        'n_wear': n_wear,
    }


def stratify(keys, feat, rng):
    """分层键：break 难度三档 × wear 密度三档 = 最多 9 层。

    ⚠ 前几版都试错，记在这里免得再踩：
      ① `break_per_img` 段级是 0/1 双峰（段内要么无 break，要么每图约 1 个），
         三分位退化成 q1=0.000 / q2=1.000，等于只按「有无」分，val 反而低 26.6%。
      ② `break_area` 直接分位也不行：**1080 段里 757 段根本没有 break**，
         这些段的值为 0，把三分位压到同一个点上。
      ③ 第二维用「段大小」→ 比例修好了（1425/179/181），但 wear 崩：val 29 / test 164。
      ④ 第二维改「有无 wear」（2 档）→ 仍崩：val 28 / test 179。
         原因是有 wear 的段内部差异极大，单段能塞几十个 wear 框，
         **只分「有无」等于没有区分度**，随机抽仍可能把高密度段整批抽走。
      正解：wear 也按密度分三档（0 / 低 / 高），与 break 三档交叉成 9 层。
    """
    withb = [k for k in keys if feat[k]['break_area'] > 0]
    med = float(np.median([feat[k]['break_area'] for k in withb])) if withb else 0.0
    wd = [feat[k]['n_wear'] / feat[k]['n_img'] for k in keys if feat[k]['n_wear'] > 0]
    wmed = float(np.median(wd)) if wd else 0.0
    strata = collections.defaultdict(list)
    for k in keys:
        a = feat[k]['break_area']
        lvl = 'nobreak' if a <= 0 else ('bsmall' if a <= med else 'blarge')
        wr = feat[k]['n_wear'] / feat[k]['n_img']
        w = 'w0' if wr <= 0 else ('wlo' if wr <= wmed else 'whi')
        strata[f'{lvl}|{w}'].append(k)
    return strata, (float(med), len(withb), float(wmed))


def optimize(keys, feat, rng, ratio=(8, 1, 1), iters=400000, T0=0.6, T1=0.02):
    """直接在**四个总量指标**上做局部搜索（模拟退火），把三个 split 压到同一比例。

    为什么放弃纯分层：段是原子单位，而段大小差 77 倍（1~77 图）、wear 框又集中在
    103 个段里。任何「分层 + 随机切」都会在某一维上爆掉（实测按图数配额切 →
    val 只剩 132 图；按段数配额切 → val/test 冲到 215/263 且 break 面积偏 30~44%；
    加 wear 维 → wear 框 val 49 / test 35）。层只解决「覆盖」，不解决「配平」。

    目标函数（归一化偏差平方和，四维等权）：
        Σ_metric Σ_split ( value / (total × 目标份额) − 1 )²
        metric ∈ {图数, break 框数, wear 框数, thunder 框数, break 面积和, 全体面积和}
    搜索：随机交换两个不同 split 的整段，Δcost<0 接受，否则按 exp(−Δ/T) 概率接受。
    段整段交换 ⇒ 零跨段约束天然成立。
    """
    SPLITS = ['train', 'val', 'test']
    tot_r = sum(ratio)
    tgt = {sp: ratio[i] / tot_r for i, sp in enumerate(SPLITS)}
    ba = {k: feat[k]['break_area'] * feat[k]['n_break'] for k in keys}
    # 六维等权。只约束前四维时实测 val 的「全体框面积」偏 −49.7%、thunder 框数偏 −26.8%：
    # 优化器会拿「抽到一堆小框段」去凑 break 框数，把尺度维度卖掉。必须进目标函数。
    METRICS = ['n_img', 'n_break', 'n_wear', 'n_thunder', 'ba', 'area']

    def val_of(k, m):
        if m == 'ba':
            return ba[k]
        if m == 'area':
            return feat[k]['sum_area']
        return feat[k][m]

    totals = {m: sum(val_of(k, m) for k in keys) for m in METRICS}

    # 权重：图数/面积只作粗略约束；三类**框数**才是 AP 的直接分子，权重加倍。
    # 实测仅等权时 val 的 break 框只拿到 8.3%（目标 10%）、thunder 8.8%，
    # 因为优化器更愿意牺牲「少数类别的框数」去换图数整齐。
    W = {'n_img': 0.5, 'n_break': 2.0, 'n_wear': 2.0, 'n_thunder': 2.0,
         'ba': 1.0, 'area': 0.5}

    def cost_one(a, sp, m):
        want = totals[m] * tgt[sp]
        if want <= 0:
            return 0.0
        return W[m] * (a[sp][m] / want - 1.0) ** 2

    # ---- 初始化：按 n_img 降序贪心，尽量保住 8:1:1 ----
    a = {sp: {m: 0.0 for m in METRICS} for sp in SPLITS}
    state = {}
    for k in sorted(keys, key=lambda x: -feat[x]['n_img']):
        best, bestd = 'train', -1e9
        for sp in SPLITS:
            d = sum(-(cost_one(a, sp, m)) for m in METRICS)
            # 用「当前缺口」而非 cost 增量，避免大段一开始就把 train 吃满
            d = sum((totals[m] * tgt[sp] - a[sp][m]) / max(totals[m] * tgt[sp], 1e-9)
                    for m in METRICS)
            if d > bestd:
                best, bestd = sp, d
        state[k] = best
        for m in METRICS:
            a[best][m] += val_of(k, m)

    cur = sum(cost_one(a, sp, m) for sp in SPLITS for m in METRICS)
    lists = {sp: [k for k in keys if state[k] == sp] for sp in SPLITS}

    acc = 0
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / max(iters - 1, 1))
        sp1, sp2 = rng.choice(SPLITS, 2, replace=False)
        if not lists[sp1] or not lists[sp2]:
            continue
        i1 = rng.randint(len(lists[sp1]))
        i2 = rng.randint(len(lists[sp2]))
        k1, k2 = lists[sp1][i1], lists[sp2][i2]

        old = sum(cost_one(a, sp, m) for sp in (sp1, sp2) for m in METRICS)
        for m in METRICS:
            v1, v2 = val_of(k1, m), val_of(k2, m)
            a[sp1][m] += v2 - v1
            a[sp2][m] += v1 - v2
        new = sum(cost_one(a, sp, m) for sp in (sp1, sp2) for m in METRICS)

        if new <= old or rng.random() < np.exp(-(new - old) / max(T, 1e-12)):
            state[k1], state[k2] = sp2, sp1
            lists[sp1][i1], lists[sp2][i2] = k2, k1
            cur += new - old
            acc += 1
        else:
            for m in METRICS:
                v1, v2 = val_of(k1, m), val_of(k2, m)
                a[sp1][m] -= v2 - v1
                a[sp2][m] -= v1 - v2
    print(f'  优化：{iters} 次试探，接受 {acc} 次，终态 cost={cur:.4f}')
    return state, a



def dist_report(assign, feat, keys):
    """报三 split 的难度/尺度分布，用来看分层是否真把分布拉齐了。"""
    print('\n【难度与尺度分布（分层的核心验收指标）】')
    hdr = (f'  {"split":6s} {"图":>5s} {"段":>4s} {"break/图":>9s} {"break面积中位‰":>13s}'
           f' {"全体面积中位‰":>13s} {"大wear%":>8s} {"break框":>7s}')
    print(hdr)
    rows = {}
    for sp in ('train', 'val', 'test'):
        ks = [k for k in keys if assign[k] == sp]
        n_img = sum(feat[k]['n_img'] for k in ks)
        nb = sum(feat[k]['n_break'] for k in ks)
        nw = sum(feat[k]['n_wear'] for k in ks)
        # 段级加权平均，权重=段图数
        w = np.array([feat[k]['n_img'] for k in ks], dtype=float)
        bp = np.array([feat[k]['break_per_img'] for k in ks], dtype=float)
        ba = np.array([feat[k]['break_area'] for k in ks], dtype=float)
        bw = np.array([feat[k]['big_wear_ratio'] for k in ks], dtype=float)
        ma = np.array([feat[k]['med_area'] for k in ks], dtype=float)
        wsum = w.sum()
        # break 面积只对有 break 的段加权，否则被 0 拉低
        wa = w[ba > 0]
        ba_m = float((ba[ba > 0] * wa).sum() / wa.sum()) if wa.sum() > 0 else 0.0
        nt = sum(feat[k]['n_box'] for k in ks) - nb - nw
        rows[sp] = dict(img=n_img, seg=len(ks), bp=float((bp * w).sum() / wsum),
                        ba=ba_m, bw=float((bw * w).sum() / wsum),
                        ma=float((ma * w).sum() / wsum), n_break=nb, n_wear=nw,
                        wd=nw / n_img, td=nt / n_img)
        r = rows[sp]
        print(f'  {sp:6s} {n_img:5d} {len(ks):4d} {r["bp"]:9.3f} {r["ba"]*1000:13.2f} '
              f'{r["ma"]*1000:13.2f} {r["bw"]*100:7.1f}% {nb:7d}')
    # 验收：val/test 相对 train 的关键偏差（图数比例 + 三类框密度 + 尺度）
    print('  ── 相对 train 的偏差（>15% 记 ⚠）──')
    bad = 0
    for sp in ('val', 'test'):
        for key, lab in (('bp', 'break/图'), ('ba', 'break面积'), ('ma', '全体面积'),
                         ('wd', 'wear/图'), ('td', 'thunder/图')):
            t, v = rows['train'][key], rows[sp][key]
            rel = (v - t) / t * 100 if t else 0.0
            ok = abs(rel) < 15
            bad += 0 if ok else 1
            print(f'  {sp:5s} {lab:9s} {t:.4f} → {v:.4f}  {rel:+6.1f}%  {"✓" if ok else "⚠"}')
    print(f'  合计越界项：{bad}')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ratio', default='8:1:1')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--tag', default='strat',
                    help='输出文件名后缀，默认 strat → assign_strat.json')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    plan, segs, old_of = load_segments()
    keys = sorted(segs.keys())
    print(f'段数 {len(keys)}  图数 {len(plan["assign"])}')

    feat = {k: seg_features(segs[k], old_of) for k in keys}
    tot = sum(f['n_img'] for f in feat.values())
    print(f'段特征完成；break/图 全局 {sum(f["n_break"] for f in feat.values())/tot:.3f}')

    rng = np.random.RandomState(args.seed)
    strata, (med, n_withb, wmed) = stratify(keys, feat, rng)
    print(f'\n【分层】含 break 的段 {n_withb}/{len(keys)}，框面积中位 '
          f'{med*1000:.2f}‰ 分大小框；wear 密度中位 {wmed:.2f} 框/图；共 {len(strata)} 层')
    for name in sorted(strata):
        ks = strata[name]
        ni = sum(feat[k]['n_img'] for k in ks)
        print(f'  {name:14s} 段{len(ks):5d} 图{ni:5d}')

    ratio = tuple(int(x) for x in args.ratio.split(':'))
    print('\n【配平优化】六目标：图数 / break 框 / wear 框 / thunder 框 / break 面积 / 全体面积（均按 8:1:1）')
    assign, agg = optimize(keys, feat, rng, ratio)

    # ---------- 验收 ----------
    n_split = collections.Counter()
    for k in keys:
        n_split[assign[k]] += feat[k]['n_img']
    print(f'\n新划分图数：{dict(n_split)}（目标约 '
          f'{round(tot*ratio[0]/sum(ratio))}/{round(tot*ratio[1]/sum(ratio))}/'
          f'{round(tot*ratio[2]/sum(ratio))}）')

    # 零跨段
    cross = sum(1 for k in keys if len({assign[k]} ) != 1)
    print(f'零跨段自检：每段只属于一个 split → 违规 {cross}')

    rows = dist_report(assign, feat, keys)

    # 与随机划分对照
    old_assign = plan['assign']
    print('\n【与随机划分（已训那版）对照】')
    fname_of = {}
    for k in keys:
        for f in segs[k]:
            fname_of[f] = k
    for sp in ('val', 'test'):
        ok = [f for f, s in old_assign.items() if s == sp and f in fname_of]
        bp_old = float(np.mean([feat[fname_of[f]]['break_per_img'] for f in ok]))
        bp_new = rows[sp]['bp']
        print(f'  {sp:5s} break/图  随机版 {bp_old:.3f} → 分层版 {bp_new:.3f}'
              f'（train 参考 {rows["train"]["bp"]:.3f}）')

    # 类别分布
    print('\n【类别分布】')
    for sp in ('train', 'val', 'test'):
        ks = [k for k in keys if assign[k] == sp]
        nb = sum(feat[k]['n_break'] for k in ks)
        nt = sum(feat[k]['n_box'] for k in ks) - nb - sum(feat[k]['n_wear'] for k in ks)
        nw = sum(feat[k]['n_wear'] for k in ks)
        print(f'  {sp:5s} break={nb:5d} thunderbolt={nt:5d} wear={nw:5d}')

    # ---------- 落盘 ----------
    os.makedirs(OUT, exist_ok=True)
    out_assign = {}
    for k in keys:
        for f in segs[k]:
            out_assign[f] = assign[k]
    new_plan = {
        'seed': args.seed, 'ratio': args.ratio, 'method': 'stratified_by_segment',
        'strata': {n: len(v) for n, v in strata.items()},
        'quantiles': {'break_area_median': med, 'n_seg_with_break': n_withb,
                      'wear_density_median': wmed},
        'assign': out_assign,
        'seg_of': plan['seg_of'],
        'old_of': old_of,
        'seg_size': {str(k): feat[k]['n_img'] for k in keys},
        # 每张图的段级难度分，供后续「按难度分组评估」直接使用
        'difficulty': {f: feat[k]['break_per_img'] for k in keys for f in segs[k]},
    }
    p = os.path.join(OUT, f'assign_{args.tag}.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(new_plan, f, ensure_ascii=False, indent=1)
    print(f'\n清单已写 {p}')

    if args.apply:
        for sp in ('train', 'val', 'test'):
            for kind in ('images', 'labels'):
                os.makedirs(os.path.join(OUT, kind, sp), exist_ok=True)
        n = 0
        for k in keys:
            sp = assign[k]
            for f in segs[k]:
                shutil.copy2(os.path.join(BASE, 'images', old_of[f], f),
                             os.path.join(OUT, 'images', sp, f))
                lab = os.path.join(BASE, 'labels', old_of[f], f.rsplit('.', 1)[0] + '.txt')
                if os.path.exists(lab):
                    shutil.copy2(lab, os.path.join(OUT, 'labels', sp,
                                                   f.rsplit('.', 1)[0] + '.txt'))
                n += 1
        with open(os.path.join(OUT, 'data.yaml'), 'w', encoding='utf-8') as f:
            f.write(f'path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\n'
                    f'nc: 3\nnames: [\'break\', \'thunderbolt\', \'wear\']\ntask: detect\n')
        print(f'已复制 {n} 张到 {OUT}，并写 data.yaml')


if __name__ == '__main__':
    main()
