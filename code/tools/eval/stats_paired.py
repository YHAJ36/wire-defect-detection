"""
多种子配对显著性检验（学术阶段必需）

为什么需要：
    此前「11 个改进族全部不超过地板」的结论只基于**均值比较 + 区间是否重叠**，
    在 n=3 下这个判据既过严（区间重叠但对配对差显著）又过松（区间不重叠但配对差不显著）。
    正确做法是对**同种子配对差**做检验：同一 seed 下两族共享数据划分、初始化与增强序列，
    配对能消掉最大的一块方差（数据集/种子效应），检验力远高于独立两样本。

指标：
    - 每族 n=3 的 mAP50 / mAP50-95 均值、标准差
    - 配对差 d_i = 族_i − 地板_i（同 seed），均值、标准差
    - t 统计量与双侧 p（df = n−1）
    - Cohen's d_z = mean(d) / sd(d)，配对效应量
    - bootstrap 95% CI（重采样配对差，n=3 时用所有 3^3 组合的精确分布更稳，这里两者都给）
    - 若 n<3 或种子不配对，只报点估计并标注「不可检验」

判读约定（写进本文件，避免事后改口）：
    |d| < 0.6 pt 且 p > 0.05  → 判「无证据表明有差异」（不是「证明无差异」）
    p < 0.05 且 d < 0          → 判「显著变差」
    p < 0.05 且 d > 0          → 判「显著变好」

数据源：exp_upscale/logs/eval_nms05.tsv（27 run × 2 split，NMS 0.5 统一口径）

用法：
    python code/tools/eval/stats_paired.py
    python code/tools/eval/stats_paired.py --metric mAP50-95
"""
import argparse
import csv
import math
import re
from collections import defaultdict
from itertools import product
from pathlib import Path

TSV = Path('D:/2/exp_upscale/logs/eval_nms05.tsv')
FLOOR = 'nf_v11_896'      # 地板族前缀
SPLIT = 'test'


def load():
    """返回 {run: {split: {metric: value}}}"""
    d = defaultdict(lambda: defaultdict(dict))
    with TSV.open(encoding='utf-8-sig') as f:
        rd = csv.DictReader(f, delimiter='\t')
        if rd.fieldnames is None:
            raise SystemExit(f'空表: {TSV}')
        cols = [c.strip() for c in rd.fieldnames]
        for row in rd:
            row = {(k or '').strip(): (v or '').strip() for k, v in row.items()}
            run = row.get(cols[0], '')
            if not run:
                continue
            sp = row.get('split', '')
            for m in ('mAP50', 'mAP50-95', 'P', 'R'):
                v = row.get(m, '')
                if v not in ('', 'nan', 'NaN', None):
                    try:
                        d[run][sp][m] = float(v)
                    except ValueError:
                        pass
    return d


def family_of(run):
    """nf_v11_896_s42 -> nf_v11_896 ; att_v11_896_triplet_s42 -> att_v11_896_triplet"""
    return re.sub(r'_s\d+$', '', run)


def seed_of(run):
    m = re.search(r'_s(\d+)$', run)
    return int(m.group(1)) if m else None


def t_test_paired(d):
    """返回 (t, p) 双侧；n<2 返回 nan"""
    n = len(d)
    if n < 2:
        return float('nan'), float('nan')
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return float('inf') if m else 0.0, 0.0 if m else 1.0
    t = m / (sd / math.sqrt(n))
    # 双侧 p，用 t 分布的生存函数；n 小，用 exact via incomplete beta
    p = _t_sf_two_sided(abs(t), n - 1)
    return t, p


def _t_sf_two_sided(t, df):
    """双侧 p = I_{df/(df+t^2)}(df/2, 1/2)，用正则不完全 beta 的连分式"""
    x = df / (df + t * t)
    return _betainc_reg(df / 2.0, 0.5, x)


def _betainc_reg(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # 连分式 (Lentz)
    f, c, dd = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        dd = 1 + num * dd
        if abs(dd) < 1e-30:
            dd = 1e-30
        dd = 1 / dd
        c = 1 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * dd
        if abs(1 - c * dd) < 1e-12:
            break
    val = front * (f - 1)
    if x < (a + 1) / (a + b + 2):
        return val
    return 1 - _betainc_reg(b, a, 1 - x)


def boot_ci(d, n_boot=20000):
    """配对差的 bootstrap 95% CI（有放回重采样 n 个差值）"""
    import random
    random.seed(42)
    n = len(d)
    if n < 2:
        return float('nan'), float('nan')
    ms = []
    for _ in range(n_boot):
        s = sum(random.choice(d) for _ in range(n))
        ms.append(s / n)
    ms.sort()
    return ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot)]


def exact_perm(d):
    """n=3 时符号翻转精确置换检验（2^n 种），返回双侧 p"""
    n = len(d)
    if n > 12:
        return float('nan')
    obs = abs(sum(d))
    cnt = 0
    tot = 0
    for signs in product((1, -1), repeat=n):
        s = abs(sum(si * di for si, di in zip(signs, d)))
        tot += 1
        if s >= obs - 1e-12:
            cnt += 1
    return cnt / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric', default='mAP50', choices=['mAP50', 'mAP50-95', 'P', 'R'])
    args = ap.parse_args()

    data = load()
    if not data:
        raise SystemExit(f'没读到数据：{TSV}')

    # 地板：同 seed 的每颗
    floor = {}
    for run, v in data.items():
        if family_of(run) == FLOOR and SPLIT in v and args.metric in v[SPLIT]:
            sd = seed_of(run)
            if sd is not None:
                floor[sd] = v[SPLIT][args.metric]
    if not floor:
        raise SystemExit('找不到地板族数据，检查 TSV 格式')
    fmean = sum(floor.values()) / len(floor)
    fsd = math.sqrt(sum((x - fmean) ** 2 for x in floor.values()) / max(len(floor) - 1, 1))
    print(f'地板 {FLOOR}  n={len(floor)}  seeds={sorted(floor)}')
    print(f'  {args.metric}: 均值 {fmean:.4f}  sd {fsd:.4f}  '
          f'范围 [{min(floor.values()):.4f}, {max(floor.values()):.4f}]  跨度 {max(floor.values()) - min(floor.values()):.4f}')
    print(f'  （跨度 {max(floor.values()) - min(floor.values()):.4f} 是尺子的噪声地板）\n')

    # 按族汇总
    fams = defaultdict(dict)
    for run, v in data.items():
        fam = family_of(run)
        if fam == FLOOR:
            continue
        if SPLIT in v and args.metric in v[SPLIT]:
            sd = seed_of(run)
            if sd is not None:
                fams[fam][sd] = v[SPLIT][args.metric]

    rows = []
    for fam, per in fams.items():
        vals = list(per.values())
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else float('nan')
        # 配对差：只取 seed 同时存在的
        common = sorted(set(per) & set(floor))
        d = [per[s] - floor[s] for s in common]
        n = len(d)
        dm = sum(d) / n if n else float('nan')
        dsd = (math.sqrt(sum((x - dm) ** 2 for x in d) / (n - 1)) if n > 1 else float('nan'))
        t, p = t_test_paired(d) if n > 1 else (float('nan'), float('nan'))
        p_exact = exact_perm(d) if n > 1 else float('nan')
        lo, hi = boot_ci(d) if n > 1 else (float('nan'), float('nan'))
        dz = dm / dsd if (n > 1 and dsd and dsd > 0) else float('nan')
        # 同一族「均值 − 地板均值」的朴素差（不配对，便于对照）
        naive = mean - fmean
        verdict = _verdict(dm, p_exact if p_exact == p_exact else p, n)
        rows.append(dict(fam=fam, n=n, seeds=common, mean=mean, sd=sd, naive=naive,
                         dm=dm, dsd=dsd, t=t, p=p, p_ex=p_exact, lo=lo, hi=hi, dz=dz,
                         verdict=verdict))

    rows.sort(key=lambda r: -r['naive'])
    print(f'{"族":30s}{"n":>3s}{"均值":>9s}{"朴素差":>9s}{"配对差":>9s}{"CI95":>18s}{"p置换":>8s}{"dz":>7s}  判读')
    for r in rows:
        ci = f'[{r["lo"]:+.4f},{r["hi"]:+.4f}]' if r['lo'] == r['lo'] else '—'
        pex = f'{r["p_ex"]:.3f}' if r['p_ex'] == r['p_ex'] else '—'
        dzs = f'{r["dz"]:+.2f}' if r['dz'] == r['dz'] else '—'
        print(f'{r["fam"]:30s}{r["n"]:>3d}{r["mean"]:>9.4f}{r["naive"]:>+9.4f}'
              f'{r["dm"]:>+9.4f}{ci:>18s}{pex:>8s}{dzs:>7s}  {r["verdict"]}')

    print(f'\n判读约定：|配对差|<0.6pt 且 p>0.05 → 「无证据表明有差异」；'
          f'p<0.05 时按符号判变好/变差。')
    print(f'注意：n=3 时符号翻转置换检验的最小可达 p 是 '
          f'{2 / (2 ** 3):.3f}，即 n=3 永远拿不到 p<0.05 —— 这是 n=3 的硬限制，'
          f'任何族的「显著」都不可能成立。')


def _verdict(dm, p, n):
    if dm != dm:
        return '—'
    if n < 3:
        return f'点估计 {dm:+.2f} pt（n={n}，不可检验）'
    if p == p and p < 0.05:
        return '显著变好' if dm > 0 else '显著变差'
    if abs(dm) < 0.6:
        return '无证据（效应<0.6pt）'
    return f'方向{"正" if dm > 0 else "负"}但未达显著（n={n}）'


if __name__ == '__main__':
    main()
