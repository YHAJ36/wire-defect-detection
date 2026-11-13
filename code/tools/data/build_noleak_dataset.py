# -*- coding: utf-8 -*-
"""把段级划分（assign.json）落成完整的 YOLO 数据集：images/labels 的 train/val/test + data.yaml。

来历
    `split_by_scene.py` 只产出了 assign.json（文件名 → split 的映射），物理文件一直没切。
    此前 `dataset_noleak_clean/` 只是为**纯推理评估**临时切的子集，且它的 data.yaml 里
    `train:` 直接指向旧 `dataset_final_bbox/images/train`（1428 张）——做推理无害，
    但**重训就是泄漏入口**（新 train 会混进本该属于 test 的图）。故本脚本另切一份完整的
    `dataset_noleak/`，三段全部按段隔离，data.yaml 不引用任何旧目录。

口径
    源：D:/2/dataset_final_bbox（1785 张：旧 train 1428 + val 178 + test 179）
    划分：D:/2/dataset_noleak/assign.json（dHash≤2 且 64×64 灰度 MAE≤20 双闸 → 并查集
          → 1080 个视觉段 → 段级 8:1:1，seed=42）
    产物：D:/2/dataset_noleak/{images,labels}/{train,val,test}、data.yaml、split_report.json

为什么复制而不是硬链接
    硬链接会让新旧两份"同一个 inode"，日后若对源文件做任何原地处理（清洗、重标注），
    新数据集会被静默改变，而泄漏校验却查不出来。数据总共 135 MB，复制的代价可以忽略。

用法
    python build_noleak_dataset.py            # 切分 + 校验
    python build_noleak_dataset.py --verify   # 只校验已有产物
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

SRC = 'D:/2/dataset_final_bbox'
ASSIGN = 'D:/2/dataset_noleak/assign.json'
DST = 'D:/2/dataset_noleak'
SPLITS = ('train', 'val', 'test')
NAMES = ['break', 'thunderbolt', 'wear']


def build_index():
    """filename -> (源 split, 图片路径, 标签路径)"""
    idx = {}
    for sp in SPLITS:
        d = f'{SRC}/images/{sp}'
        for f in os.listdir(d):
            stem = os.path.splitext(f)[0]
            idx[f] = (sp, f'{d}/{f}', f'{SRC}/labels/{sp}/{stem}.txt')
    return idx


def copy_split(assign, idx):
    n = Counter()
    for sp in SPLITS:
        os.makedirs(f'{DST}/images/{sp}', exist_ok=True)
        os.makedirs(f'{DST}/labels/{sp}', exist_ok=True)
    for f, sp in assign.items():
        _, ip, lp = idx[f]
        shutil.copy2(ip, f'{DST}/images/{sp}/{f}')
        shutil.copy2(lp, f'{DST}/labels/{sp}/{os.path.basename(lp)}')
        n[sp] += 1
    return n


def write_yaml():
    txt = (
        f'# 段级无泄漏数据集（split_by_scene.py: dHash<=2 & MAE<=20 双闸 → 并查集 → 1080 段 → 8:1:1）\n'
        f'# 由 build_noleak_dataset.py 生成；task=detect，nc=3\n'
        f'path: {DST}\n'
        f'train: images/train\n'
        f'val: images/val\n'
        f'test: images/test\n'
        f'nc: 3\n'
        f"names: ['break', 'thunderbolt', 'wear']\n"
    )
    open(f'{DST}/data.yaml', 'w', encoding='utf-8', newline='\n').write(txt)


def verify():
    """四道硬校验：文件齐全 / 无跨 split 重名 / 段不跨 split / 标签合法。"""
    a = json.load(open(ASSIGN, encoding='utf-8'))
    assign, seg_of = a['assign'], a['seg_of']
    ok = True

    # ① 物理文件数
    seen = {}
    for sp in SPLITS:
        imgs = sorted(os.listdir(f'{DST}/images/{sp}'))
        lbls = sorted(os.listdir(f'{DST}/labels/{sp}'))
        imgs = [x for x in imgs if x.lower().endswith(('.jpg', '.png'))]
        lbls = [x for x in lbls if x.endswith('.txt')]
        assert len(imgs) == len(lbls), f'{sp}: 图 {len(imgs)} ≠ 标签 {len(lbls)}'
        for f in imgs:
            if f in seen:
                ok = False
                print(f'  ✗ 文件名跨 split 重复：{f}（{seen[f]} / {sp}）')
            seen[f] = sp
        print(f'  {sp:5s} images={len(imgs):4d} labels={len(lbls):4d}')

    # ② 与 assign.json 逐条一致
    exp = Counter(assign.values())
    got = Counter(seen.values())
    for sp in SPLITS:
        if exp[sp] != got[sp]:
            ok = False
            print(f'  ✗ {sp} 数量不符：assign={exp[sp]} 实际={got[sp]}')
    if set(seen) != set(assign):
        ok = False
        print(f'  ✗ 文件名集合不符：多出 {len(set(seen)-set(assign))}，缺失 {len(set(assign)-set(seen))}')

    # ③ 段不跨 split（泄漏的机器判据）
    segs = defaultdict(set)
    for f, s in seg_of.items():
        segs[s].add(assign[f])
    cross = [s for s, v in segs.items() if len(v) > 1]
    print(f'  视觉段 {len(segs)} 个，跨 split {len(cross)} 个')
    if cross:
        ok = False
        print(f'  ✗ 段跨 split：{cross[:10]}')

    # ④ 标签格式 + 类别分布
    per = defaultdict(Counter)
    bad = 0
    for sp in SPLITS:
        for lbl in os.listdir(f'{DST}/labels/{sp}'):
            if not lbl.endswith('.txt'):
                continue
            for ln, line in enumerate(open(f'{DST}/labels/{sp}/{lbl}', encoding='utf-8')):
                p = line.split()
                if not p:
                    continue
                if len(p) != 5:
                    bad += 1
                    continue
                c = int(p[0])
                if c < 0 or c > 2 or any(not (0 <= float(x) <= 1) for x in p[1:]):
                    bad += 1
                    continue
                per[sp][NAMES[c]] += 1
    if bad:
        ok = False
        print(f'  ✗ 非法标签行 {bad}')
    else:
        print('  标签格式：全部合法（5 列 / 类别 0-2 / 坐标 ∈ [0,1]）')
    for sp in SPLITS:
        tot = sum(per[sp].values())
        print(f'  {sp:5s} boxes={tot:5d}  ' + '  '.join(f'{n}={per[sp][n]}' for n in NAMES))

    # ⑤ 不残留 ultralytics 标签缓存
    caches = [f'{DST}/labels/{sp}.cache' for sp in SPLITS] + \
             [f'{DST}/labels/{sp}.cache' for sp in SPLITS]
    left = [c for c in set(caches) if os.path.exists(c)]
    if left:
        for c in left:
            os.remove(c)
        print(f'  已清理过期缓存：{left}')

    print('\n校验通过 ✓' if ok else '\n校验失败 ✗')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true', help='只校验已有产物')
    args = ap.parse_args()

    if not args.verify:
        a = json.load(open(ASSIGN, encoding='utf-8'))
        assign = a['assign']
        print(f'assign.json: {len(assign)} 张，{a["segments"]} 段，ratio={a["ratio"]} seed={a["seed"]}')
        idx = build_index()
        print(f'源数据索引：{len(idx)} 张（{SRC}）')
        missing = [f for f in assign if f not in idx]
        if missing:
            sys.exit(f'❌ assign 里有 {len(missing)} 张在源数据中找不到：{missing[:5]}')
        n = copy_split(assign, idx)
        print('已复制：' + '  '.join(f'{k}={v}' for k, v in sorted(n.items())))
        write_yaml()
        print(f'已写 {DST}/data.yaml')

    print('\n=== 校验 ===')
    if not verify():
        sys.exit(1)


if __name__ == '__main__':
    main()
