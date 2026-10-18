# -*- coding: utf-8 -*-
"""生成《参考文献结构图合集_v1.pdf》。

内容：
  1) 封面；
  2) 画图工具与操作建议（2 页）；
  3) 10 张论文结构图（裁剪 + 出处 + 可借鉴点 + 本项目怎么画）；
  4) 本项目已生成的 YOLOv11s + TripletAttention 目标图。

用法：
  D:/yolo-env/Scripts/python.exe report/make_refs_pdf.py

注意：论文图片版权归原作者/期刊所有，本 PDF 仅供个人学习与绘图参考，
不随 GitHub 项目上传。
"""
import os

import pymupdf
from PIL import Image
from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(BASE, 'References')
TMP = os.path.join(BASE, 'report', '_refs_pdf')
OUT = os.path.join(BASE, 'report', '参考文献结构图合集_v1.pdf')
ARCH = os.path.join(BASE, 'report', 'figures', 'yolo11_triplet_arch.png')

INK = HexColor('#1F3864')
BLUE = HexColor('#2E5395')
LIGHT = HexColor('#D9E2F3')
PALE = HexColor('#F3F6FB')
ORANGE = HexColor('#C55A11')
GRAY = HexColor('#595959')
GRID = HexColor('#8EA9DB')

FONT = 'MSYH'
FONT_BD = 'MSYHBD'
pdfmetrics.registerFont(TTFont(FONT, r'C:\Windows\Fonts\msyh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont(FONT_BD, r'C:\Windows\Fonts\msyhbd.ttc', subfontIndex=0))

PW, PH = landscape(A4)  # 841.89 x 595.28

FILES = {
    'p1': '基于YOLO11s绝缘子缺陷检测的轻量化算法_王国国(1).pdf',
    'p2': '基于YOLO的输电线路绝缘子缺陷和异物检测_王振岳(1).pdf',
    'p3': '基于多模态融合的输电线路鸟巢精细化检测方法_江灏.pdf',
    'p4': '基于改进YOLOv11的龙眼簇成熟度分类识别方法_黄培创.pdf',
    'p5': '基于改进YOLO的输电线路绝缘子缺陷检测模型及应用研究_齐浩宇(1).pdf',
}

# (slug, page, 裁剪框 PDF points, 图题, 论文, 原图位置)
REFS = [
    ('p4', 3, (48, 396, 537, 752),
     '图1  改进YOLOv11（CBAM注意力增强）模型总体结构',
     '黄培创 等《基于改进YOLOv11的龙眼簇成熟度分类识别方法》',
     '第 3 页 · 图1（老师给的样例原图）'),
    ('p4', 5, (48, 52, 496, 298),
     '图2  CBAM注意力增强模块',
     '黄培创 等《基于改进YOLOv11的龙眼簇成熟度分类识别方法》',
     '第 5 页 · 图2'),
    ('p1', 3, (305, 406, 527, 686),
     '图1  YOLO11s 结构图',
     '王国国 等《基于YOLO11s绝缘子缺陷检测的轻量化算法》',
     '第 3 页 · 图1'),
    ('p1', 6, (60, 46, 274, 312),
     '图5  改进后的 YOLO11s 模型结构',
     '王国国 等《基于YOLO11s绝缘子缺陷检测的轻量化算法》',
     '第 6 页 · 图5'),
    ('p2', 26, (45, 112, 450, 750),
     '图3.2  YOLOv7 改进模型结构示意图',
     '王振岳《基于YOLO的输电线路绝缘子缺陷和异物检测》',
     '第 26 页 · 图3.2（扫描页）'),
    ('p2', 28, (192, 365, 378, 642),
     '图3.4  CA 模块结构示意图',
     '王振岳《基于YOLO的输电线路绝缘子缺陷和异物检测》',
     '第 28 页 · 图3.4（扫描页）'),
    ('p3', 3, (72, 455, 513, 760),
     '图1  鸟巢精细化检测方法流程图',
     '江灏 等《基于多模态融合的输电线路鸟巢精细化检测方法》',
     '第 3 页 · 图1'),
    ('p5', 28, (163, 374, 445, 762),
     '图2-10  YOLOv8s 结构图',
     '齐浩宇《基于改进YOLO的输电线路绝缘子缺陷检测模型及应用研究》',
     '第 28 页 · 图2-10'),
    ('p5', 36, (134, 456, 465, 768),
     '图3-1  RID-YOLOv7 网络结构',
     '齐浩宇《基于改进YOLO的输电线路绝缘子缺陷检测模型及应用研究》',
     '第 36 页 · 图3-1'),
    ('p5', 37, (160, 252, 440, 512),
     '图3-2  CA 注意力机制模块',
     '齐浩宇《基于改进YOLO的输电线路绝缘子缺陷检测模型及应用研究》',
     '第 37 页 · 图3-2'),
]

NOTES = {
    ('p4', 3): dict(
        borrow=[
            '横向排版：Backbone 从左到右排成一条链，层号 0~12 写在方块上方，'
            '每个方块下方写“通道×分辨率”，一眼能看出下采样过程。',
            'Backbone、Head 各用一个虚线圆角框分区；Neck 不单独框，靠三列布局体现。',
            '最左边放输入图像缩略图，最右边放 Head 输出，形成“端到端”的视觉闭环。',
            '底部一排 5 个模块细节面板（Conv / C3k2 / CBAM / C2PSA / Bottleneck），'
            '每个面板只画 3~5 个方块，不抢主图。',
            '最下方放图例：下采样、C3k2、注意力模块、逐元素相乘、残差相加、s=2。',
        ],
        ours=[
            '保留“主结构 + 底部模块面板 + 图例”的版式；主结构改成 Backbone → Neck（P3/P4/P5 三列）→ Detect。',
            '把 CBAM 面板替换成 TripletAttention 面板（三分支 C×W / H×C / H×W）。',
            'Head 是 Detect 不是 Classify，输出写 3×(4+3)（3 个尺度 × (4 框 + 3 类)）。',
            '在 SPPF 后插入橙色 Triplet 块，旁边标注“仅 1 处，+300 参数”。',
            '输入写 896×896×3，抽头写 P3 112²、P4 56²、P5 28²。',
        ],
        caution='这是老师给的样例，版式优先照它；但它的输入是 640、任务是分类，'
                '我们的输入是 896、任务是 3 类检测，数字不要照抄。',
    ),
    ('p4', 5): dict(
        borrow=[
            '上下两条分支（通道注意力 / 空间注意力），每支内部用“池化 → 共享 MLP 或 7×7 卷积 → Sigmoid → ⊗”表达。',
            '左侧画输入特征图，右侧画输出特征图，中间用箭头串起来，读者不用看文字也能懂。',
            '底部放图例：特征图、池化结果、注意力权重、逐元素相乘、逐元素相加。',
        ],
        ours=[
            'Triplet 是三条平行分支（C×W、H×C、H×W），不是两条；每支都是 ZPool → 7×7 Conv → BN → Sigmoid → ⊗。',
            '一定要把 BN 画出来：这是 Triplet 与 CBAM 的关键差别（BN 去直流、钉尺度）。',
            '三支结果不是 Concat，而是逐元素平均（1/3 Σ）。',
        ],
        caution='CBAM 的通道分支用共享 MLP；Triplet 每支只有一个 7×7 Conv（bias=False）+ BN，参数量只有 300。',
    ),
    ('p1', 3): dict(
        borrow=[
            'YOLO11s 基线分块：左侧 Backbone 竖排（Conv / C3k2 / SPPF / C2PSA），中间 Neck（C3k2 / Concat / Upsample），右侧 Head（Detect×3）。',
            '用颜色区分 Backbone / Neck / Head 三个区域，P3/P4/P5 三个检测尺度分别标注。',
        ],
        ours=[
            '层号按 ultralytics 8.3.180 的 0–23 标；我们的 Triplet 插在 layer 9（SPPF）之后、layer 10（C2PSA）之前。',
            '输入 896 时三个尺度是 P3 112×112×128、P4 56×56×256、P5 28×28×512。',
        ],
        caution='论文里的 YOLO11s 图通常是 640 输入、80 类；通道数一样，但分辨率和输出维度要按我们的 896/3 类改。',
    ),
    ('p1', 6): dict(
        borrow=[
            '改进模块用红色高亮，并在 Backbone/Neck 的对应位置标出来，读者一眼能找到改动点。',
            '左侧 Backbone 竖排、右侧 Neck+Head，适合纵向页面；如果画横向，可参考 p4 第 3 页。',
            '底部用图例说明红色块的含义。',
        ],
        ours=[
            '我们只在 SPPF 后插入 1 处 Triplet，不要画成多个注意力块；用橙色高亮，和“本文改动”图例对应。',
            '插入位置写清楚：SPPF(layer 9) → Triplet → C2PSA(layer 10)。',
        ],
        caution='这张图里改了多个位置（Backbone 和 Neck 都有红色块），我们的实验配置只有 neck_sppf 一处，画多会与代码/实验记录不符。',
    ),
    ('p2', 26): dict(
        borrow=[
            '主图横向排布，改进区域用红色虚线框圈出，并在框上方写改进说明（如“将 SPPCSPC 替换为 RFB”）。',
            '下方用“模块 = 展开结构”的等号形式，把 CBS、MP、ELAN、RepConv、RFB 逐个拆开，信息密度高但分区清楚。',
            '每个模块用固定配色，图例统一。',
        ],
        ours=[
            '主图 + 底部模块面板的版式可以直接借鉴；Triplet 面板也可以用“Triplet = 三分支 → 平均”的等号展开。',
            '改进点用橙色虚线框圈出，框上写“SPPF 后插入 Triplet（+300）”。',
        ],
        caution='这张是扫描页，线条和文字偏模糊；照布局即可，不要照抄它的模块名（RFB/MP/ELAN 是 YOLOv7 的）。',
    ),
    ('p2', 28): dict(
        borrow=[
            '输入 → 残差 → X/Y 方向池化 → Concat+Conv → BN+非线性 → 两个 Conv → 两个 Sigmoid → Re-weight，张量流向非常清楚。',
            '每个方块只写一个操作，箭头不交叉；这是“模块细节图”的好范本。',
        ],
        ours=[
            'Triplet 的 C×W 分支可以类比 CA 的 X/Y 池化：沿宽度池化得到 C×W，再 7×7 Conv + BN + Sigmoid。',
            '但 Triplet 三支独立、不 Concat，最后三支平均；不要把 CA 的“Concat+Conv”照搬。',
        ],
        caution='CA 有 BN + 非线性；Triplet 每支是 Conv → BN → Sigmoid，顺序不同，别画反。',
    ),
    ('p3', 3): dict(
        borrow=[
            '多分支总框架：输入 → 三个模块（图像特征、语义信息、多模态融合）→ 输出，模块之间用虚线框和箭头连接。',
            '每个大模块内部再画小子块，形成“总—分”两级结构。',
        ],
        ours=[
            '如果导师要“方法总框架”，可以照这个画一张：输入 → Backbone → Neck → Triplet → Detect → 输出；再把 Triplet 单独放大。',
            '我们的主图已经是“总—分”结构（主结构 + 底部模块面板），不需要再画流程图。',
        ],
        caution='这张是多模态方法，不是检测网络结构；只借布局，不借模块内容。',
    ),
    ('p5', 28): dict(
        borrow=[
            '每个 stage 标出模块名、通道数、分辨率；Neck 的 Upsample / Concat 用箭头明确标出。',
            'Head 三个尺度并排，输出维度写在旁边。',
            '整体是“左 Backbone、中 Neck、右 Head”的经典三段式。',
        ],
        ours=[
            '我们的 Neck 三列（P3/P4/P5）和 layer 12/13/15/16/18/19/21/22 标注可以照这个风格。',
            'Head 写 Detect P3/P4/P5，输出 3×(4+3)，并注明 anchor-free。',
        ],
        caution='YOLOv8s 是 C2f，YOLO11s 是 C3k2 + C2PSA；模块名不要混。',
    ),
    ('p5', 36): dict(
        borrow=[
            'Backbone / Neck / Head 三个虚线大框，改进点用红色块标出，底部放图例。',
            '主图只画数据流，模块细节放到下方或另一张图，避免主图过密。',
        ],
        ours=[
            '我们的主图就是三个虚线框：Backbone（含 Triplet）、Neck（FPN+PAN）、Head（Detect）。',
            'Triplet 用橙色块 + 星号 ★，图例写“本文改动（Triplet）”。',
        ],
        caution='这张图的改进点是 WIoU + CA + GSConv 等多个；我们只画 Triplet 一个结构改动，WIoU 属于损失函数，不要画进网络结构图。',
    ),
    ('p5', 37): dict(
        borrow=[
            'H/W 两个方向的池化 + 共享卷积 + Sigmoid，张量维度（C×H×1、C×1×W）标得很清楚。',
            '分支合并用 ⊗ 或 ⊕ 表达，图例统一。',
        ],
        ours=[
            'Triplet 三支分别沿 C×W、H×C、H×W 做池化；每支 7×7 Conv + BN + Sigmoid。',
            '最后三支逐元素平均（1/3 Σ），不是 Concat，也不是简单相加。',
        ],
        caution='CA 的池化是 H 和 W 两个方向；Triplet 是三支，其中一支是 H×W 平面。维度标注不要写错。',
    ),
}


def wrap(text, font, size, width):
    lines = []
    for para in text.split('\n'):
        cur = ''
        for ch in para:
            if pdfmetrics.stringWidth(cur + ch, font, size) <= width:
                cur += ch
            else:
                if cur:
                    lines.append(cur)
                cur = ch
        lines.append(cur)
    return lines


def para(c, x, y, text, font=FONT, size=9.5, leading=14, width=280,
         color=GRAY):
    c.setFont(font, size)
    c.setFillColor(color)
    for line in wrap(text, font, size, width):
        c.drawString(x, y, line)
        y -= leading
    return y


def bullets(c, x, y, items, font=FONT, size=9.3, leading=13.5, width=280,
            color=GRAY, bullet_color=ORANGE):
    for item in items:
        c.setFont(font, size)
        c.setFillColor(bullet_color)
        c.drawString(x, y, '•')
        c.setFillColor(color)
        lines = wrap(item, font, size, width - 11)
        for line in lines:
            c.drawString(x + 11, y, line)
            y -= leading
        y -= 3
    return y


def header(c, idx, total, title, paper, loc):
    c.setFillColor(white)
    c.rect(0, 0, PW, PH, fill=1, stroke=0)
    c.setFillColor(ORANGE)
    c.rect(0, PH - 6, PW, 6, fill=1, stroke=0)
    c.setFont(FONT, 8.5)
    c.setFillColor(GRAY)
    c.drawString(36, PH - 26, f'参考图 {idx:02d} / {total:02d}')
    c.setFont(FONT_BD, 14.5)
    c.setFillColor(INK)
    c.drawString(36, PH - 52, title)
    c.setFont(FONT, 9)
    c.setFillColor(GRAY)
    c.drawString(36, PH - 69, paper)
    c.drawString(36, PH - 83, loc)


def crop_page(slug, pno, rect, dpi=250):
    os.makedirs(TMP, exist_ok=True)
    src = pymupdf.open(os.path.join(REF, FILES[slug]))
    page = src[pno - 1]
    pix = page.get_pixmap(dpi=dpi, clip=pymupdf.Rect(*rect))
    png = os.path.join(TMP, f'{slug}_p{pno:03d}.png')
    pix.save(png)
    src.close()
    # 转 JPEG 控制 PDF 体积
    jpg = os.path.join(TMP, f'{slug}_p{pno:03d}.jpg')
    im = Image.open(png).convert('RGB')
    im.save(jpg, quality=88, optimize=True)
    return jpg, im.width, im.height


def ref_page(c, idx, total, item):
    slug, pno, rect, title, paper, loc = item
    header(c, idx, total, title, paper, loc)
    jpg, iw, ih = crop_page(slug, pno, rect)
    # 左侧图片区
    area_x, area_y, area_w, area_h = 36, 62, 470, 440
    scale = min(area_w / iw, area_h / ih)
    w, h = iw * scale, ih * scale
    x = area_x + (area_w - w) / 2
    y = area_y + (area_h - h) / 2
    c.setStrokeColor(GRID)
    c.setLineWidth(0.6)
    c.rect(x - 3, y - 3, w + 6, h + 6, fill=0, stroke=1)
    c.drawImage(jpg, x, y, w, h)
    c.setFont(FONT, 7.8)
    c.setFillColor(GRAY)
    c.drawCentredString(x + w / 2, y - 14, '原论文图（已裁剪，仅供对照）')

    # 右侧文字区
    tx, tw = 528, 280
    yy = PH - 110
    c.setFont(FONT_BD, 11.5)
    c.setFillColor(BLUE)
    c.drawString(tx, yy, '可借鉴点')
    yy -= 18
    notes = NOTES[(slug, pno)]
    yy = bullets(c, tx, yy, notes['borrow'], width=tw)
    yy -= 6
    c.setFont(FONT_BD, 11.5)
    c.setFillColor(ORANGE)
    c.drawString(tx, yy, '本项目怎么画')
    yy -= 18
    yy = bullets(c, tx, yy, notes['ours'], width=tw, bullet_color=BLUE)
    yy -= 6
    c.setFont(FONT_BD, 9.5)
    c.setFillColor(INK)
    c.drawString(tx, yy, '注意')
    yy -= 15
    para(c, tx, yy, notes['caution'], size=8.8, leading=12.8, width=tw,
         color=GRAY)

    # 页脚
    c.setFont(FONT, 7.5)
    c.setFillColor(GRAY)
    c.drawString(36, 20, '仅供个人学习与绘图参考；论文版权归原作者/期刊所有。')
    c.drawRightString(PW - 36, 20, f'{idx} / {total}')
    c.showPage()


def cover(c):
    c.setFillColor(white)
    c.rect(0, 0, PW, PH, fill=1, stroke=0)
    c.setFillColor(INK)
    c.rect(0, PH - 132, PW, 132, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont(FONT_BD, 27)
    c.drawString(40, PH - 55, '参考文献结构图合集')
    c.setFont(FONT, 13.5)
    c.drawString(40, PH - 84, 'YOLOv11s + TripletAttention 网络结构示意图 · 对照素材')
    c.setFont(FONT, 9.5)
    c.drawString(40, PH - 106, '2026-10-19 · 仅供个人学习与绘图参考')

    c.setFillColor(INK)
    c.setFont(FONT_BD, 13)
    c.drawString(40, PH - 175, '收录论文')
    papers = [
        ('p4', '黄培创 等《基于改进YOLOv11的龙眼簇成熟度分类识别方法》', '第3、5页'),
        ('p1', '王国国 等《基于YOLO11s绝缘子缺陷检测的轻量化算法》', '第3、6页'),
        ('p2', '王振岳《基于YOLO的输电线路绝缘子缺陷和异物检测》', '第26、28页'),
        ('p3', '江灏 等《基于多模态融合的输电线路鸟巢精细化检测方法》', '第3页'),
        ('p5', '齐浩宇《基于改进YOLO的输电线路绝缘子缺陷检测模型及应用研究》', '第28、36、37页'),
    ]
    y = PH - 202
    for slug, text, pages in papers:
        c.setFillColor(ORANGE)
        c.setFont(FONT_BD, 10)
        c.drawString(44, y, slug)
        c.setFillColor(INK)
        c.setFont(FONT, 9.6)
        c.drawString(72, y, text)
        c.setFillColor(GRAY)
        c.setFont(FONT, 8.8)
        c.drawString(72, y - 13, pages)
        y -= 38

    # 右侧目标图缩略
    if os.path.exists(ARCH):
        im = Image.open(ARCH)
        iw, ih = im.size
        w = 330
        h = w * ih / iw
        x, yy = PW - w - 40, PH - 205 - h
        c.setStrokeColor(GRID)
        c.setLineWidth(0.8)
        c.rect(x - 4, yy - 4, w + 8, h + 8, fill=0, stroke=1)
        c.drawImage(ARCH, x, yy, w, h)
        c.setFont(FONT, 8.5)
        c.setFillColor(GRAY)
        c.drawCentredString(x + w / 2, yy - 18, '本项目目标图（已生成，可打开 .drawio 修改）')

    c.setFillColor(PALE)
    c.rect(40, 55, PW - 80, 58, fill=1, stroke=0)
    c.setFillColor(ORANGE)
    c.setFont(FONT_BD, 9.5)
    c.drawString(54, 92, '版权说明')
    para(c, 54, 77, '论文图片版权归原作者及期刊所有；本合集仅用于个人学习、绘图对照，'
                    '不随 GitHub 项目上传，不用于任何商业用途。', size=8.8, leading=12,
         width=PW - 110, color=GRAY)
    c.showPage()


def guide_tools(c):
    c.setFillColor(white)
    c.rect(0, 0, PW, PH, fill=1, stroke=0)
    c.setFillColor(INK)
    c.rect(0, PH - 8, PW, 8, fill=1, stroke=0)
    c.setFont(FONT_BD, 20)
    c.setFillColor(INK)
    c.drawString(40, PH - 48, '这个图在哪里画？用什么软件 / App？')
    c.setFont(FONT, 9.5)
    c.setFillColor(GRAY)
    c.drawString(40, PH - 67, '结论先说：电脑上首选 draw.io（diagrams.net），其次 PowerPoint；'
                              'iPad 用 draw.io 网页版或 Concepts；手机不建议。')

    col_w = 372
    x1, x2 = 40, 440
    y1 = PH - 100

    c.setFont(FONT_BD, 12)
    c.setFillColor(BLUE)
    c.drawString(x1, y1, '1. draw.io / diagrams.net（首选）')
    y = y1 - 18
    y = bullets(c, x1, y, [
        '免费，网页版打开即用：app.diagrams.net；也有 Windows/macOS/Linux 客户端。',
        '不用注册，首次打开选 “Device” 即可把文件存在本地。',
        '导出 PNG（300 dpi）、SVG、PDF 都支持；矢量 PDF 放大不糊。',
        '我们已生成可编辑源文件：report/figures/yolo11_triplet_arch.drawio。',
    ], width=col_w)
    y -= 4
    c.setFont(FONT_BD, 10)
    c.setFillColor(ORANGE)
    c.drawString(x1, y, '打开方式')
    y -= 15
    y = para(c, x1, y, '打开 app.diagrams.net → File → Open From → Device → '
                       '选择 yolo11_triplet_arch.drawio → 直接改文字/挪方块。',
             size=9, leading=13, width=col_w, color=GRAY)
    y -= 8
    c.setFont(FONT_BD, 12)
    c.setFillColor(BLUE)
    c.drawString(x1, y, '2. PowerPoint（次选，交作业方便）')
    y -= 18
    y = bullets(c, x1, y, [
        '插入 → 形状：圆角矩形、箭头、虚线框；用“连接符”自动吸附。',
        '开始 → 排列 → 对齐/分布，先把主干方块排齐，再画箭头。',
        '导出：文件 → 导出 → 更改文件类型 → PDF；或另存为 PNG。',
        '缺点：箭头自动绕行能力弱，模块多时容易乱。',
    ], width=col_w)

    y2 = PH - 100
    c.setFont(FONT_BD, 12)
    c.setFillColor(BLUE)
    c.drawString(x2, y2, '3. Visio / Figma / Illustrator（进阶）')
    y = y2 - 18
    y = bullets(c, x2, y, [
        'Visio：Windows 付费，流程图/工程图最专业，但学校场景没必要。',
        'Figma：免费版够用，矢量、组件、对齐强；适合追求排版质量。',
        'Illustrator：出版级矢量图，学习成本最高；老师若要求期刊风格可用。',
        'Python + SVG（我们当前做法）：可复现、可批量改，但改版不如拖拽直观。',
    ], width=col_w)
    y -= 4
    c.setFont(FONT_BD, 12)
    c.setFillColor(BLUE)
    c.drawString(x2, y, '4. iPad / 手机')
    y -= 18
    y = bullets(c, x2, y, [
        'iPad：draw.io 网页版可直接用；Concepts 适合无限画布草图。',
        'Procreate 是位图，文字和箭头后期不好改，不推荐画结构图。',
        '手机屏幕太小，只适合看，不适合画。',
    ], width=col_w)
    y -= 4
    c.setFont(FONT_BD, 10)
    c.setFillColor(ORANGE)
    c.drawString(x2, y, '推荐组合')
    y -= 15
    para(c, x2, y, '先用我们的 SVG/PNG/PDF 当底稿 → draw.io 里照着摆 → '
                    '导出 300 dpi PNG 插进 Word/PPT，同时留一份 PDF 矢量版。',
         size=9, leading=13, width=col_w, color=GRAY)

    # 工具速查表
    ty = 262
    c.setFont(FONT_BD, 11)
    c.setFillColor(INK)
    c.drawString(40, ty, '工具选择速查')
    rows = [
        ('工具', '平台', '费用', '导出格式', '推荐度'),
        ('draw.io', '网页 / 桌面', '免费', 'PNG / SVG / PDF', '★★★★★'),
        ('PowerPoint', 'Windows / macOS', '已有 Office', 'PNG / PDF', '★★★★'),
        ('Visio', 'Windows', '付费', 'PNG / SVG / PDF', '★★★'),
        ('Figma', '网页 / 桌面', '免费版够用', 'PNG / SVG / PDF', '★★★★'),
        ('Illustrator', 'Windows / macOS', '付费', 'AI / SVG / PDF', '★★★'),
        ('Procreate', 'iPad', '付费', 'PNG / PSD（位图）', '★★'),
    ]
    colx = [44, 150, 300, 420, 620]
    rh = 21
    c.setFillColor(LIGHT)
    c.rect(40, ty - 22, 760, rh, fill=1, stroke=0)
    for j, row in enumerate(rows):
        yy = ty - 36 - j * rh
        font = FONT_BD if j == 0 else FONT
        c.setFont(font, 8.8)
        c.setFillColor(INK if j == 0 else GRAY)
        for x, text in zip(colx, row):
            c.drawString(x, yy, text)
        if j:
            c.setStrokeColor(HexColor('#E3E8F2'))
            c.setLineWidth(0.4)
            c.line(40, yy - 6, 800, yy - 6)
    c.showPage()


def guide_steps(c):
    c.setFillColor(white)
    c.rect(0, 0, PW, PH, fill=1, stroke=0)
    c.setFillColor(INK)
    c.rect(0, PH - 8, PW, 8, fill=1, stroke=0)
    c.setFont(FONT_BD, 20)
    c.setFillColor(INK)
    c.drawString(40, PH - 48, '画图顺序与排版规范')
    c.setFont(FONT, 9.5)
    c.setFillColor(GRAY)
    c.drawString(40, PH - 67, '照着这个顺序画，通常 1~2 小时能完成一版；先保证结构对，再抠细节。')

    x1, x2 = 40, 440
    col_w = 372
    y = PH - 100
    c.setFont(FONT_BD, 12)
    c.setFillColor(BLUE)
    c.drawString(x1, y, '推荐顺序（从主到次）')
    y -= 20
    steps = [
        ('① 主干', '先画 Backbone 的 0–10 共 11 个方块 + Triplet，定好层号、通道数、分辨率。'),
        ('② Neck', '画 P3/P4/P5 三列：先画上采样路径（P5→P4→P3），再画下采样路径（P3→P4→P5）。'),
        ('③ Head', '画 Detect P3/P4/P5 三行，写输出 3×(4+3)，注明 anchor-free。'),
        ('④ 模块面板', '底部依次画 Conv / C3k2 / SPPF / C2PSA / Triplet / Detect，每个 3~5 个方块。'),
        ('⑤ 图例与注释', '加图例、标题、输入/输出说明；把“仅 1 处、+300 参数”写在 Triplet 旁。'),
        ('⑥ 导出', '先导出 PNG 检查，再导出 PDF 矢量版；PNG 用 300 dpi。'),
    ]
    for name, text in steps:
        c.setFont(FONT_BD, 10)
        c.setFillColor(ORANGE)
        c.drawString(x1, y, name)
        c.setFont(FONT, 9.2)
        c.setFillColor(GRAY)
        for line in wrap(text, FONT, 9.2, col_w - 78):
            c.drawString(x1 + 78, y, line)
            y -= 13
        y -= 5

    y2 = PH - 100
    c.setFont(FONT_BD, 12)
    c.setFillColor(BLUE)
    c.drawString(x2, y2, '排版规范（避免返工）')
    y = y2 - 20
    y = bullets(c, x2, y, [
        '字体：中文微软雅黑 / 黑体，英文 Arial；主图 9~12 pt，注释 8~9 pt。',
        '颜色：只用 3 种主色（深蓝=主干、绿=Neck、橙=本文改动），其余用灰。',
        '箭头：数据流用实线，模块边界/抽头用虚线；同一张图不要混用多种箭头样式。',
        '对齐：先用网格/对齐工具排方块，再画连线；连线尽量走横平竖直。',
        '标注：每个方块至少写“模块名 + 通道数/分辨率”；层号写在方块上方。',
        '留白：主图与模块面板之间留 30~40 px，别让箭头穿字。',
        '导出：PNG 300 dpi（插 Word/PPT），PDF 矢量（打印/提交），SVG（后续再改）。',
    ], width=col_w)
    y -= 6
    c.setFont(FONT_BD, 10)
    c.setFillColor(ORANGE)
    c.drawString(x2, y, '本项目的配色')
    y -= 16
    c.setFillColor(HexColor('#1F3864'))
    c.rect(x2, y - 2, 16, 10, fill=1, stroke=0)
    c.setFillColor(GRAY)
    c.setFont(FONT, 8.8)
    c.drawString(x2 + 22, y, '主干 / 文字')
    c.setFillColor(HexColor('#548235'))
    c.rect(x2 + 120, y - 2, 16, 10, fill=1, stroke=0)
    c.setFillColor(GRAY)
    c.drawString(x2 + 142, y, 'Neck / 特征融合')
    c.setFillColor(HexColor('#C55A11'))
    c.rect(x2 + 260, y - 2, 16, 10, fill=1, stroke=0)
    c.setFillColor(GRAY)
    c.drawString(x2 + 282, y, '本文改动（Triplet）')

    # 常见错误
    cy = 286
    c.setFont(FONT_BD, 12)
    c.setFillColor(ORANGE)
    c.drawString(40, cy, '常见错误（画完对照检查）')
    cy -= 20
    mistakes = [
        '输入尺寸写成 640；本项目是 896，P3/P4/P5 应为 112/56/28，不是 80/40/20。',
        'Detect 输出写成 3×(4+1+3)：YOLO11 是 anchor-free，没有 objectness，应为 3×(4+3)。',
        'Triplet 面板漏画 BN，或画成 CBAM 的“通道+空间”两分支；Triplet 是 C×W / H×C / H×W 三支。',
        '把 WIoU 画进网络结构图：WIoU 是损失函数，不是网络层，只在实验章节说明。',
        '改进块画了多个位置：我们的实验只有 neck_sppf 一处（SPPF 后），画多会与记录不符。',
        '箭头穿过文字或方块、连线不对齐：先排方块再连箭头，连线走横平竖直。',
        '导出 PNG 只有 72 dpi，放大模糊：至少 300 dpi；提交打印再附一份矢量 PDF。',
    ]
    cy = bullets(c, 40, cy, mistakes, size=9.2, leading=14, width=760,
                 bullet_color=ORANGE)
    c.showPage()


def final_page(c):
    c.setFillColor(white)
    c.rect(0, 0, PW, PH, fill=1, stroke=0)
    c.setFillColor(ORANGE)
    c.rect(0, PH - 6, PW, 6, fill=1, stroke=0)
    c.setFont(FONT_BD, 16)
    c.setFillColor(INK)
    c.drawString(36, PH - 40, '本项目目标图：YOLOv11s + TripletAttention 网络结构示意图')
    c.setFont(FONT, 9)
    c.setFillColor(GRAY)
    c.drawString(36, PH - 57, '已生成 SVG / PNG / PDF / draw.io 四种格式；'
                              '源文件在 report/figures/，可用 draw.io 直接改。')
    im = Image.open(ARCH)
    iw, ih = im.size
    w = 690
    h = w * ih / iw
    x = (PW - w) / 2
    y = PH - 78 - h
    c.setStrokeColor(GRID)
    c.setLineWidth(0.6)
    c.rect(x - 3, y - 3, w + 6, h + 6, fill=0, stroke=1)
    c.drawImage(ARCH, x, y, w, h)
    c.setFont(FONT, 8.5)
    c.setFillColor(GRAY)
    c.drawString(36, 26, '对照 p4 第3页的版式：主结构 + 底部模块面板 + 图例；'
                          'Triplet 为三分支 ZPool→7×7 Conv→BN→Sigmoid→⊗，最后 1/3 平均。')
    c.showPage()


def main():
    c = canvas.Canvas(OUT, pagesize=landscape(A4))
    c.setTitle('参考文献结构图合集 v1')
    c.setAuthor('D:/2 输电线缺陷检测项目')
    cover(c)
    guide_tools(c)
    guide_steps(c)
    total = len(REFS)
    for i, item in enumerate(REFS, 1):
        ref_page(c, i, total, item)
    final_page(c)
    c.save()
    print('wrote', OUT)


if __name__ == '__main__':
    main()
