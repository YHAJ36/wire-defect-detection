# -*- coding: utf-8 -*-
"""从 dataset_noleak 派生单类（只保留 break / 断股）数据集。

做法：
  - 图片用硬链接（os.link），不占额外磁盘；跨盘失败则退回复制。
  - 标签只保留 class 0 的框，其余类别整行删掉。
  - 不含 break 的图也保留，标签写成空文件 —— 当作纯背景负样本，
    有助于压低误检（YOLO 原生支持空标签）。
输出：D:/2/k210/v5/dataset_break/
"""
import os
import shutil
import sys

SRC = r'D:/2/dataset_noleak'
DST = r'D:/2/k210/v5/dataset_break'
SPLITS = ('train', 'val', 'test')
KEEP_CLASS = 0          # break
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
        return 'link'
    except OSError:
        shutil.copy2(src, dst)
        return 'copy'


def main():
    stat = {}
    for s in SPLITS:
        si = os.path.join(SRC, 'images', s)
        sl = os.path.join(SRC, 'labels', s)
        di = os.path.join(DST, 'images', s)
        dl = os.path.join(DST, 'labels', s)
        os.makedirs(di, exist_ok=True)
        os.makedirs(dl, exist_ok=True)

        kept_box = 0
        pos_img = 0
        n_img = 0
        how = {'link': 0, 'copy': 0}
        for fn in sorted(os.listdir(si)):
            if not fn.lower().endswith(IMG_EXT):
                continue
            n_img += 1
            stem = os.path.splitext(fn)[0]
            how[link_or_copy(os.path.join(si, fn), os.path.join(di, fn))] += 1

            lab = os.path.join(sl, stem + '.txt')
            rows = []
            if os.path.exists(lab):
                for ln in open(lab, encoding='utf-8'):
                    p = ln.split()
                    if len(p) >= 5 and int(float(p[0])) == KEEP_CLASS:
                        rows.append('0 ' + ' '.join(p[1:5]))
            if rows:
                pos_img += 1
                kept_box += len(rows)
            with open(os.path.join(dl, stem + '.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(rows) + ('\n' if rows else ''))

        stat[s] = (n_img, pos_img, kept_box, how)
        print(f'[{s:5s}] 图片 {n_img:4d}  含断股 {pos_img:4d}  断股框 {kept_box:4d}  '
              f'硬链接 {how["link"]} 复制 {how["copy"]}')

    yaml_txt = f"""# 单类（断股 break）派生集，由 build_break_ds.py 从 dataset_noleak 生成
# 只保留 class 0；无 break 的图作为纯背景负样本保留（空标签）
path: {DST.replace(chr(92), '/')}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['break']
"""
    with open(os.path.join(DST, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(yaml_txt)
    print(f'\n[data] {os.path.join(DST, "data.yaml")}')


if __name__ == '__main__':
    main()
