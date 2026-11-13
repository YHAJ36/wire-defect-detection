# -*- coding: utf-8 -*-
"""训练预算诊断：用已有 results.csv 判断每个 run 是"欠训"还是"已收敛"。

判据（不训练、只读曲线）：
  tail_slope = 末 10 轮 val mAP50-95 的线性斜率（每轮 pt）
  peak_pos   = 峰值出现的轮次 / 总轮次
  gap        = 峰值 - 末轮（pt）

读法：
  tail_slope > 0 且 peak_pos > 0.85  → 欠训，加预算有收益
  tail_slope ≈ 0 且 peak_pos < 0.85  → 已收敛，加预算无益
  同一族不同种子的 tail_slope 差异 → 该族的方差主要来自未收敛

用法：
  D:/yolo-env/Scripts/python.exe code/tools/diag/budget_tail_slope.py
  ... --root D:/2/exp_upscale/runs --last 10
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_curve(path):
    """返回 (epochs, mAP50, mAP50-95)，缺列返回 None。"""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        key50 = next((k for k in rd.fieldnames if k.strip().endswith('mAP50(B)')), None)
        key95 = next((k for k in rd.fieldnames if k.strip().endswith('mAP50-95(B)')), None)
        if not key50 or not key95:
            return None
        for r in rd:
            try:
                rows.append((int(float(r['epoch'])), float(r[key50]), float(r[key95])))
            except (ValueError, TypeError, KeyError):
                continue
    if len(rows) < 12:
        return None
    rows.sort()
    ep = np.array([r[0] for r in rows], dtype=float)
    return ep, np.array([r[1] for r in rows]), np.array([r[2] for r in rows])


def slope(y):
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def family(name):
    """把 s42/s43/s44 之类的种子后缀去掉，便于按族聚合。"""
    return re.sub(r'_s\d+$', '', name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='D:/2/exp_upscale/runs')
    ap.add_argument('--last', type=int, default=10, help='取末 N 轮算斜率')
    ap.add_argument('--min-epochs', type=int, default=55, help='只看跑满的 run')
    args = ap.parse_args()

    records = []
    for p in sorted(Path(args.root).glob('*/results.csv')):
        got = read_curve(p)
        if got is None:
            continue
        ep, m50, m95 = got
        if ep[-1] + 1 < args.min_epochs:
            continue
        k = min(args.last, len(ep))
        records.append(dict(
            name=p.parent.name, n=len(ep),
            s50=slope(m50[-k:]) * 100, s95=slope(m95[-k:]) * 100,
            peak95=float(ep[int(np.argmax(m95))]) + 1,
            peak50=float(ep[int(np.argmax(m50))]) + 1,
            last50=float(m50[-1]) * 100, max50=float(m50.max()) * 100,
            last95=float(m95[-1]) * 100, max95=float(m95.max()) * 100,
        ))

    if not records:
        raise SystemExit('没找到可用 results.csv')

    print(f'\n=== 逐个 run（末 {args.last} 轮斜率，单位 pt/轮）===')
    hdr = f'{"run":32s} {"ep":>4s} {"s50":>7s} {"s95":>7s} {"peak95@":>8s} {"peak/总":>7s} {"末50":>7s} {"峰50":>7s} {"gap50":>6s}'
    print(hdr)
    print('-' * len(hdr))
    for r in sorted(records, key=lambda x: -x['s95']):
        print(f'{r["name"][:32]:32s} {r["n"]:4d} {r["s50"]:+7.3f} {r["s95"]:+7.3f} '
              f'{r["peak95"]:8.0f} {r["peak95"] / r["n"]:7.2f} '
              f'{r["last50"]:7.2f} {r["max50"]:7.2f} {r["max50"] - r["last50"]:+6.2f}')

    fam = defaultdict(list)
    for r in records:
        fam[family(r['name'])].append(r)

    print(f'\n=== 按族聚合（n≥2 的族才显示均值）===')
    hdr2 = f'{"family":34s} {"n":>2s} {"s95均值":>8s} {"s95跨度":>8s} {"末50均值":>9s}'
    print(hdr2)
    print('-' * len(hdr2))
    for f, rs in sorted(fam.items(), key=lambda kv: -np.mean([r['s95'] for r in kv[1]])):
        s95 = [r['s95'] for r in rs]
        print(f'{f[:34]:34s} {len(rs):2d} {np.mean(s95):+8.3f} {max(s95) - min(s95):8.3f} '
              f'{np.mean([r["last50"] for r in rs]):9.2f}')

    print('\n=== 读法 ===')
    print(' s95 > +0.050 pt/轮 且 peak/总 > 0.85 → 欠训，加预算大概率有收益')
    print(' s95 ≈ 0          或 peak/总 < 0.85      → 已收敛，加预算无益')
    print(' 同族 s95 跨度大                          → 该族的方差来自未收敛，不是结构')


if __name__ == '__main__':
    main()
