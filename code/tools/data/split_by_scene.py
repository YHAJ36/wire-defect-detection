# -*- coding: utf-8 -*-
"""按「视觉段」重划分，消除 train/val/test 之间的近重复泄漏。

背景
    数据是视频截帧，但原划分是按文件随机抽的 8:1:1，编号在三个集合里完全交错。
    双闸定案（dHash<=2 且 64x64 灰度 MAE<=20）：test 有 48/179（26.8%）、
    val 有 54/178（30.3%）在 train 里存在真近重复。

做法
    1. 全库 1785 张算特征：64x64 灰度（判 MAE）+ 8x8 dHash 的 64 bit（判汉明距离）
    2. 双闸判近重复 → 并查集 → 连通分量即「视觉段」
    3. 段级随机划分，**同段整段进同一个 split**，目标 8:1:1（按图数）
    4. 自检：新 split 之间不应再存在任何双闸近重复对

用法
    D:/yolo-env/Scripts/python.exe code/tools/data/split_by_scene.py --seed 42 --dry-run
    ...  --apply    # 真正建目录、复制图像与标签
"""
import argparse
import collections
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

BASE = 'D:/2/dataset_final_bbox'
OUT = 'D:/2/dataset_noleak'
NAMES = {0: 'break', 1: 'thunderbolt', 2: 'wear'}
DIST_MAX = 2      # dHash 汉明距离上限
MAE_MAX = 20.0    # 64x64 灰度平均绝对误差上限


def build_feats(items):
    n = len(items)
    A = np.zeros((n, 64 * 64), dtype=np.float32)
    B = np.zeros((n, 64), dtype=np.uint8)
    for i, it in enumerate(items):
        im = Image.open(it['path']).convert('L').resize((64, 64), Image.LANCZOS)
        a = np.asarray(im, dtype=np.float32)
        A[i] = a.ravel()
        g = a.reshape(8, 8, 8, 8).mean(axis=(1, 3))     # 8x8 灰度均值
        bits, k = np.zeros(64, dtype=np.uint8), 0
        for r in range(8):
            for c in range(7):
                bits[k] = 1 if g[r, c] > g[r, c + 1] else 0
                k += 1
        B[i] = bits
    return A, B


def hamming_matrix(B, chunk=256):
    n = len(B)
    d = np.zeros((n, n), dtype=np.int16)
    for s in range(0, n, chunk):
        d[s:s + chunk] = (B[s:s + chunk, None, :] != B[None, :, :]).sum(-1)
    return d


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--ratio', default='8:1:1')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    # ---------- 1. 收集 ----------
    items = []
    for split in ('train', 'val', 'test'):
        d = os.path.join(BASE, 'images', split)
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            items.append({'old': split, 'f': f, 'path': p})
    n = len(items)
    print(f'总图像 {n}')

    # ---------- 2. 特征与近重复暗查 ----------
    A, B = build_feats(items)
    print('特征完成，开始 dHash 距离矩阵...')
    D = hamming_matrix(B)
    ii, jj = np.where(D <= DIST_MAX)
    keep = ii < jj
    ii, jj = ii[keep], jj[keep]
    print(f'dHash<={DIST_MAX} 候选对 {len(ii)}')

    dsu = DSU(n)
    edges = 0
    for i, j in zip(ii, jj):
        if np.abs(A[i] - A[j]).mean() <= MAE_MAX:
            dsu.union(int(i), int(j))
            edges += 1
    print(f'过 MAD<={MAE_MAX:.0f} 闸的真近重复对 {edges}')

    # ---------- 3. 段 ----------
    comp = collections.defaultdict(list)
    for i in range(n):
        comp[dsu.find(i)].append(i)
    segs = list(comp.values())
    sizes = sorted((len(s) for s in segs), reverse=True)
    print(f'段数 {len(segs)}  最大 {sizes[:8]}  单图段 {sum(1 for x in sizes if x == 1)}')
    cross = sum(1 for s in segs if len({items[i]['old'] for i in s}) > 1)
    print(f'跨原 split 的段 {cross} / {len(segs)}')

    # ---------- 4. 段级划分 ----------
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(segs))
    r = [int(x) for x in args.ratio.split(':')]
    tot = sum(r)
    quota = {'train': r[0] / tot * n, 'val': r[1] / tot * n, 'test': r[2] / tot * n}
    got = {'train': 0, 'val': 0, 'test': 0}
    assign = {}
    for k in order:
        s = segs[k]
        # 放进「当前占比最低」的那个集合，平局按 优先级 train→val→test
        deficit = {sp: quota[sp] - got[sp] for sp in ('train', 'val', 'test')}
        best = max(('train', 'val', 'test'), key=lambda sp: (deficit[sp] / quota[sp], sp == 'train'))
        for i in s:
            assign[i] = best
        got[best] += len(s)
    print('\n新划分图数：', got, ' 目标：', {k: round(v) for k, v in quota.items()})

    # ---------- 5. 类别分布 ----------
    def count(idx_list):
        img_cls = collections.Counter()
        box_cls = collections.Counter()
        res = collections.Counter()
        for i in idx_list:
            it = items[i]
            W, H = Image.open(it['path']).size
            res[(W, H)] += 1
            stem = it['f'].rsplit('.', 1)[0]
            lab = os.path.join(BASE, 'labels', it['old'], stem + '.txt')
            cls = set()
            if os.path.exists(lab):
                for line in open(lab):
                    p = line.split()
                    if len(p) < 5:
                        continue
                    c = int(float(p[0]))
                    box_cls[NAMES[c]] += 1
                    cls.add(c)
            for c in cls:
                img_cls[NAMES[c]] += 1
        return img_cls, box_cls, res

    print('\n【新 split 类别分布】')
    for sp in ('train', 'val', 'test'):
        idx = [i for i in range(n) if assign[i] == sp]
        ic, bc, rc = count(idx)
        print(f'  {sp:5s} 图{len(idx):5d}  含类图数 ' +
              ' '.join(f'{k}={ic[k]}' for k in NAMES.values()) +
              f'  |  框数 ' + ' '.join(f'{k}={bc[k]}' for k in NAMES.values()))

    # ---------- 6. 残留泄漏自检 ----------
    print('\n【残留泄漏自检（双闸）】')
    for a, b in (('train', 'test'), ('train', 'val'), ('val', 'test')):
        ia = {i for i in range(n) if assign[i] == a}
        ib = {i for i in range(n) if assign[i] == b}
        hit = 0
        for i, j in zip(ii, jj):
            i, j = int(i), int(j)
            if (i in ia and j in ib) or (i in ib and j in ia):
                if np.abs(A[i] - A[j]).mean() <= MAE_MAX:
                    hit += 1
        print(f'  {a} ↔ {b}: {hit} 对')

    # ---------- 7. 落盘 ----------
    seg_of = {}
    for k, s_ in enumerate(segs):
        for i in s_:
            seg_of[items[i]['f']] = k
    plan = {'seed': args.seed, 'ratio': args.ratio,
            'segments': len(segs),
            'assign': {items[i]['f']: assign[i] for i in range(n)},
            'seg_of': seg_of,
            'old_of': {items[i]['f']: items[i]['old'] for i in range(n)},
            'seg_size': {str(k): len(v) for k, v in enumerate(segs)}}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'assign.json'), 'w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    print(f'\n清单已写 {OUT}/assign.json')

    if args.apply:
        for sp in ('train', 'val', 'test'):
            for kind in ('images', 'labels'):
                os.makedirs(os.path.join(OUT, kind, sp), exist_ok=True)
        moved = 0
        for i, it in enumerate(items):
            sp = assign[i]
            stem = it['f'].rsplit('.', 1)[0]
            shutil.copy2(it['path'], os.path.join(OUT, 'images', sp, it['f']))
            lab = os.path.join(BASE, 'labels', it['old'], stem + '.txt')
            if os.path.exists(lab):
                shutil.copy2(lab, os.path.join(OUT, 'labels', sp, stem + '.txt'))
            moved += 1
        print(f'已复制 {moved} 张到 {OUT}')
        with open(os.path.join(OUT, 'data.yaml'), 'w', encoding='utf-8') as f:
            f.write(f'path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\n'
                    f'nc: 3\nnames: [\'break\', \'thunderbolt\', \'wear\']\ntask: detect\n')
        print(f'已写 {OUT}/data.yaml')


if __name__ == '__main__':
    main()
