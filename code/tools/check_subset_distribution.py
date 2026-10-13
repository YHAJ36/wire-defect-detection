# -*- coding: utf-8 -*-
"""leak 子集 vs clean 子集的分布对比 —— 确认 leak 不是"某类场景的特例"。

要回答的问题（否则 leak − clean 的差可能是组成差，不是泄漏）：
  1. **类别分布**：两边的逐类实例数与"含该类的图数"是否可比。
     （mAP50(all) 是三类未加权均值，故某一类在一边特别多/特别少就会拉动总分。）
  2. **来源分布**：文件名前缀（拍摄批次/线路）两边是否可比 ——
     若 leak 全来自一个前缀，那它测的是那个批次，不是"泄漏"。
  3. **目标数量分布**：每图目标数是否可比。

另外给一个**分级泄漏剖面**（为什么需要）：
`check_index_continuity.py` 已证明数据集是**按帧抽的、带块状结构**的
（train 内部编号相邻的图 16.6% 是 dHash≤2 的近重复）。
块状结构意味着：val/test 里的图，其"同场景不同帧"的兄弟大概率落在 train 里，
但 dHash≤2 抓不到它们。故本脚本按 dHash 汉明距离**分档**统计
"每张评估图与 train 的最近距离"，看 dHash≤2 的"leak"之外还剩多少"疑似同场景"。
这直接支撑口径"clean = 无近复制品，**不等于**未见过的场景"。

只读，纯 CPU。
输出：logs/subset_distribution.txt
用法：D:/yolo-env/Scripts/python.exe D:/2/code/tools/check_subset_distribution.py
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path('D:/2/dataset_final_bbox')
SUB = Path('D:/2/dataset_leak_subsets')
OUT = Path('D:/2/logs/subset_distribution.txt')
NAMES = ['break', 'thunderbolt', 'wear']
SPLITS = ['test', 'val']
PRE = re.compile(r'^([A-Za-z_]+)')
NUM = re.compile(r'(\d+)(?!.*\d)')


def dhash(p, size=8):
    im = Image.open(p).convert('L').resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def read_lab(root, split, name):
    f = root / 'labels' / split / (Path(name).stem + '.txt')
    if not f.exists():
        return []
    return [ln.split() for ln in f.read_text(encoding='utf-8').splitlines() if ln.strip()]


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    say('=' * 100)
    say('leak vs clean 的分布对比（确认 leak 不是"某类场景特例"）')
    say('=' * 100)

    for sp in SPLITS:
        names = {sub: sorted(p.name for p in (SUB / f'{sp}_{sub}' / 'images').iterdir())
                 for sub in ('leak', 'clean')}
        say()
        say('#' * 100)
        say(f'# {sp}（leak {len(names["leak"])} 张 / clean {len(names["clean"])} 张）')
        say('#' * 100)

        # ---- 1. 类别分布 ----
        say()
        say('--- 类别分布 ---')
        say(f'{"子集":<8}{"逐类实例数(占全 split%)":>34}{"含该类的图数(占比)":>34}')
        tot = {sub: Counter() for sub in names}
        for sub, ns in names.items():
            c, ci = Counter(), Counter()
            for n in ns:
                cs = {int(float(t[0])) for t in read_lab(SRC, sp, n)}
                for t in read_lab(SRC, sp, n):
                    c[int(float(t[0]))] += 1
                for x in cs:
                    ci[x] += 1
            tot[sub] = (c, ci, len(ns))
        for sub in ('leak', 'clean'):
            c, ci, n = tot[sub]
            s1 = '  '.join(f'{NAMES[i]}:{c[i]}' for i in range(3))
            s2 = '  '.join(f'{NAMES[i]}:{ci[i]}({ci[i] / n * 100:.0f}%)' for i in range(3))
            say(f'{sub:<8}{s1:>34}{s2:>34}')
        cl, ccl, nl = tot['leak']
        cc, ccc, nc = tot['clean']
        say()
        say('  逐类实例数占比差（leak% − clean%，>±5 点即认为组成不可比）：')
        for i in range(3):
            fl = cl[i] / max(sum(cl.values()), 1) * 100
            fc = cc[i] / max(sum(cc.values()), 1) * 100
            flag = '  ⚠ 不可比' if abs(fl - fc) > 5 else ''
            say(f'    {NAMES[i]:<12} leak {fl:5.1f}% / clean {fc:5.1f}%   差 {fl - fc:+.1f} 点{flag}')

        # ---- 2. 来源（文件名前缀）分布 ----
        say()
        say('--- 来源分布（文件名前缀）---')
        pl = Counter(PRE.match(n).group(1) if PRE.match(n) else '?' for n in names['leak'])
        pc = Counter(PRE.match(n).group(1) if PRE.match(n) else '?' for n in names['clean'])
        keys = sorted(set(pl) | set(pc))
        say(f'{"前缀":<20}{"leak":>7}{"占比":>8}{"clean":>7}{"占比":>8}{"占比差(点)":>12}')
        for k in keys:
            a, b = pl[k], pc[k]
            fa = a / max(len(names['leak']), 1) * 100
            fb = b / max(len(names['clean']), 1) * 100
            flag = '  ⚠' if abs(fa - fb) > 10 else ''
            say(f'{k:<20}{a:>7}{fa:>7.1f}%{b:>7}{fb:>7.1f}%{fa - fb:>+12.1f}{flag}')

        # ---- 3. 每图目标数 ----
        say()
        say('--- 每图目标数 ---')
        for sub in ('leak', 'clean'):
            ns = [len(read_lab(SRC, sp, n)) for n in names[sub]]
            say(f'  {sub:<8} 均值 {np.mean(ns):.2f} / 中位 {np.median(ns):.0f} / 0 目标的图 '
                f'{sum(1 for x in ns if x == 0)} 张')

        # ---- 4. 分级泄漏剖面 ----
        say()
        say('--- 分级泄漏剖面：每张评估图到 **train** 的最近 dHash 汉明距离 ---')
        trn = sorted(p for p in (SRC / 'images' / 'train').iterdir() if p.is_file())
        H = np.stack([dhash(p) for p in trn])                      # (Nt, 64) bool
        say(f'  train 基准 {len(trn)} 张')
        say(f'{"档":<22}' + ''.join(f'{s:>10}' for s in ('leak', 'clean'))
            + f'{"clean 占比":>12}')
        D = {}
        for sub in ('leak', 'clean'):
            D[sub] = np.array([int((H != dhash(SRC / "images" / sp / n)).sum(1).min())
                               for n in names[sub]])
        bands = [(0, 0, '=0 完全相同'), (1, 2, '1~2 近重复(即 leak 档)'),
                 (3, 6, '3~6 疑似同场景'), (7, 12, '7~12 可能同场景'),
                 (13, 64, '≥13 视为不同场景')]
        for lo, hi, lab in bands:
            cells = ''
            for sub in ('leak', 'clean'):
                cells += f'{int(((D[sub] >= lo) & (D[sub] <= hi)).sum()):>10}'
            pct = ((D['clean'] >= lo) & (D['clean'] <= hi)).mean() * 100
            say(f'{lab:<22}{cells}{pct:>11.1f}%')
        say()
        say(f'  最近距离中位：leak {np.median(D["leak"]):.0f} / clean {np.median(D["clean"]):.0f}')
        say('  ⇒ 读法：clean 里落在"3~12 疑似同场景"档的图，其最近邻虽没近到 dHash≤2，')
        say('     但块状结构（见 logs/index_continuity.txt）说明它们很可能是**同一段视频的别帧**。')
        say('     故"clean"只能说**无近复制品**，不能说"未见过的新场景"——这正是一条口径。')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
