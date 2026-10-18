# -*- coding: utf-8 -*-
"""参考文献结构图索引：把 References/ 下论文逐页渲染成缩略图联络表。

用途：给“YOLOv11 + Triplet 网络结构示意图”找可对照的论文原图。
本脚本只做两件事：
  1) References/_gallery/contact/pN_sheetM.png —— 每 20 页一张联络表；
  2) References/_gallery/candidates.md      —— 每页的图题/关键词索引。

不改动 References/ 原 PDF。生成物已在 .gitignore 中排除，不入库。
"""
import os
import re
import math

import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, 'References')
OUT = os.path.join(REF, '_gallery')
PAGES = os.path.join(OUT, 'pages')
CONTACT = os.path.join(OUT, 'contact')
DPI = 90
COLS, ROWS = 5, 4
THUMB_W = 300

PAPERS = [
    ('p1', '基于YOLO11s绝缘子缺陷检测的轻量化算法_王国国(1).pdf'),
    ('p2', '基于YOLO的输电线路绝缘子缺陷和异物检测_王振岳(1).pdf'),
    ('p3', '基于多模态融合的输电线路鸟巢精细化检测方法_江灏.pdf'),
    ('p4', '基于改进YOLOv11的龙眼簇成熟度分类识别方法_黄培创.pdf'),
    ('p5', '基于改进YOLO的输电线路绝缘子缺陷检测模型及应用研究_齐浩宇(1).pdf'),
]

CAP_RE = re.compile(r'^(图|Fig\.?|Figure)\s*\d', re.I)
KW = ('结构', '框架', '网络模型', '整体模型', '改进的YOLO', '总体', '流程')
FONT = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', 18)


def render_page(page, path):
    pix = page.get_pixmap(dpi=DPI)
    pix.save(path)
    return Image.open(path)


def main():
    os.makedirs(PAGES, exist_ok=True)
    os.makedirs(CONTACT, exist_ok=True)
    index = ['# References 逐页图题索引', '']
    for slug, fn in PAPERS:
        pdf = os.path.join(REF, fn)
        if not os.path.exists(pdf):
            index.append(f'## {slug} {fn}\n\n**缺失**\n')
            continue
        doc = pymupdf.open(pdf)
        pdir = os.path.join(PAGES, slug)
        os.makedirs(pdir, exist_ok=True)
        index.append(f'## {slug} {fn}  ({len(doc)} 页)')
        thumbs = []
        for i, page in enumerate(doc):
            png = os.path.join(pdir, f'p{i + 1:03d}.png')
            if not os.path.exists(png):
                page.get_pixmap(dpi=DPI).save(png)
            im = Image.open(png).convert('RGB')
            im.thumbnail((THUMB_W, 10000))
            thumbs.append((i + 1, im))
            text = page.get_text('text')
            lines = [x.strip() for x in text.splitlines() if x.strip()]
            caps = [x for x in lines if CAP_RE.match(x)]
            kws = [x for x in lines if any(k in x for k in KW)]
            if caps or kws:
                index.append(f'- p{i + 1}: ' + ' | '.join((caps + kws)[:6])[:300])
        index.append('')
        for s in range(math.ceil(len(thumbs) / (COLS * ROWS))):
            batch = thumbs[s * COLS * ROWS:(s + 1) * COLS * ROWS]
            cell_w = max(im.width for _, im in batch) + 16
            cell_h = max(im.height for _, im in batch) + 36
            sheet = Image.new('RGB', (COLS * cell_w, ROWS * cell_h), 'white')
            dr = ImageDraw.Draw(sheet)
            for j, (pno, im) in enumerate(batch):
                r, c = divmod(j, COLS)
                x = c * cell_w + 8
                y = r * cell_h + 28
                sheet.paste(im, (x, y))
                dr.text((x, y - 24), f'p{pno}', fill='black', font=FONT)
            out = os.path.join(CONTACT, f'{slug}_sheet{s + 1}.png')
            sheet.save(out)
            print('contact', out)
        doc.close()
    with open(os.path.join(OUT, 'candidates.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(index))
    print('index', os.path.join(OUT, 'candidates.md'))


if __name__ == '__main__':
    main()
