# -*- coding: utf-8 -*-
"""从 dataset_noleak 派生单类数据集（通用版，--cls 指定保留哪一类）。

与 build_break_ds.py 的区别：类别可选，输出目录/类名跟着变。
  class 0 = break（断股）   1 = strike（雷击）   2 = wear（磨损）

做法：
  - 图片用硬链接，不占额外磁盘；跨盘失败则退回复制。
  - 标签只保留目标类的框，并把类别号重映射成 0（单类必须从 0 开始）。
  - 不含该类目标的图也保留，写空标签 —— 当负样本用，压误检。

用法：
  python build_cls_ds.py --cls 1 --name strike
输出：
  D:/2/k210/v5/dataset_<name>/
"""
import argparse
import os
import shutil

SRC = r'D:/2/dataset_noleak'
ROOT = r'D:/2/k210/v5'
SPLITS = ('train', 'val', 'test')
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
NAMES = {0: 'break', 1: 'strike', 2: 'wear'}


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
        return 'link'
    except OSError:
        shutil.copy2(src, dst)
        return 'copy'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cls', type=int, required=True, help='保留的原始类别号')
    ap.add_argument('--name', default=None, help='输出目录后缀，默认用类别英文名')
    a = ap.parse_args()
    name = a.name or NAMES[a.cls]
    dst_root = os.path.join(ROOT, f'dataset_{name}')

    for s in SPLITS:
        si = os.path.join(SRC, 'images', s)
        sl = os.path.join(SRC, 'labels', s)
        di = os.path.join(dst_root, 'images', s)
        dl = os.path.join(dst_root, 'labels', s)
        os.makedirs(di, exist_ok=True)
        os.makedirs(dl, exist_ok=True)

        kept = pos = n_img = 0
        for fn in sorted(os.listdir(si)):
            if not fn.lower().endswith(IMG_EXT):
                continue
            n_img += 1
            stem = os.path.splitext(fn)[0]
            link_or_copy(os.path.join(si, fn), os.path.join(di, fn))

            lab = os.path.join(sl, stem + '.txt')
            rows = []
            if os.path.exists(lab):
                for ln in open(lab, encoding='utf-8'):
                    p = ln.split()
                    if len(p) >= 5 and int(float(p[0])) == a.cls:
                        rows.append('0 ' + ' '.join(p[1:5]))
            if rows:
                pos += 1
                kept += len(rows)
            with open(os.path.join(dl, stem + '.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(rows) + ('\n' if rows else ''))
        print(f'[{s:5s}] 图片 {n_img:4d}  含目标 {pos:4d}  框 {kept:4d}')

    yaml_txt = f"""# 单类（{name}）派生集，由 build_cls_ds.py 生成
# 源 dataset_noleak 的 class {a.cls}，类别重映射为 0
path: {dst_root.replace(chr(92), '/')}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['{name}']
"""
    with open(os.path.join(dst_root, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(yaml_txt)
    print(f'\n[data] {os.path.join(dst_root, "data.yaml")}')


if __name__ == '__main__':
    main()
