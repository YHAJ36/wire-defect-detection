# -*- coding: utf-8 -*-
"""YOLOv11s + TripletAttention 网络结构示意图 v2。

v2 的设计目标：保留 v1 的平面配色与信息量，借鉴参考论文结构图的组织方式：
  · Backbone 按 P1/2 … P5/32 阶段分组，层号在上、分辨率在下；
  · P3/P4/P5 抽头用彩色虚线明确接入 Neck 三列；
  · Neck 保留完整层号 12/13/15/16/18/19/21/22；
  · 底部模块细节按 Conv / C3k2 / SPPF / C2PSA / Triplet / Detect 排列；
  · Head 每尺度输出修正为 4 box + 3 cls = 7 通道（不是 3×(4+1+3)）。

产物：
  report/figures/yolo11_triplet_arch_v2.svg / .png / .pdf / .drawio
"""
import os
import subprocess
import sys
import xml.sax.saxutils as sx

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'report', 'figures')
W, H = 1800, 1240

INK = '#1F3864'
BLUE = '#2E5395'
LIGHT = '#D9E2F3'
PALE = '#F3F6FB'
ORANGE = '#C55A11'
ORANGE_BG = '#FBE5D6'
GREEN = '#548235'
GREEN_BG = '#E2EFDA'
GRAY = '#595959'
GRID = '#8EA9DB'
WHITE = '#FFFFFF'

FONT = 'Microsoft YaHei, SimHei, sans-serif'


class SVG:
    def __init__(self):
        self.parts = []
        self.esc = sx.escape

    def raw(self, s):
        self.parts.append(s)

    def text(self, x, y, lines, size=12, color=INK, weight='normal',
             anchor='middle', opacity=1.0):
        if isinstance(lines, str):
            lines = [lines]
        style = (f'font-family:{FONT};font-size:{size}px;fill:{color};'
                 f'font-weight:{weight};text-anchor:{anchor};'
                 f'dominant-baseline:middle;opacity:{opacity}')
        self.raw(f'<text x="{x}" y="{y}" style="{style}">')
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else size * 1.22
            self.raw(f'<tspan x="{x}" dy="{dy}">{self.esc(line)}</tspan>')
        self.raw('</text>')

    def rect(self, x, y, w, h, fill=WHITE, stroke=INK, sw=1.4, rx=6,
             dash=None, opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} '
                 f'opacity="{opacity}"/>')

    def box(self, x, y, w, h, lines, fill=WHITE, stroke=INK, sw=1.4,
            size=12, color=INK, weight='normal', dash=None):
        self.rect(x, y, w, h, fill, stroke, sw, dash=dash)
        self.text(x + w / 2, y + h / 2, lines, size=size, color=color,
                  weight=weight)

    def line(self, x1, y1, x2, y2, color=INK, sw=1.6, dash=None, arrow=True):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        m = ' marker-end="url(#arrow)"' if arrow else ''
        self.raw(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                 f'stroke="{color}" stroke-width="{sw}"{d}{m}/>')

    def path(self, points, color=INK, sw=1.6, dash=None, arrow=True):
        d = 'M ' + ' L '.join(f'{x} {y}' for x, y in points)
        dd = f' stroke-dasharray="{dash}"' if dash else ''
        m = ' marker-end="url(#arrow)"' if arrow else ''
        self.raw(f'<path d="{d}" fill="none" stroke="{color}" '
                 f'stroke-width="{sw}"{dd}{m}/>')

    def render(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                f'viewBox="0 0 {W} {H}">'
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
                'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>'
                '</defs>')
        return head + ''.join(self.parts) + '</svg>'


def build():
    s = SVG()
    s.raw(f'<rect width="{W}" height="{H}" fill="white"/>')

    # ================= 标题 =================
    s.text(40, 34, 'YOLOv11s + TripletAttention 网络结构示意图', size=27,
           color=INK, weight='bold', anchor='start')
    s.text(40, 70, '896×896，nc=3；TripletAttention 仅在 SPPF 后注入 1 处（+300 参数）；'
                   '层号与 ultralytics 8.3.180 一致',
           size=14.5, color=GRAY, anchor='start')

    # ================= Backbone =================
    s.rect(175, 100, 1190, 300, fill='#FBFCFE', stroke=BLUE, sw=1.6,
           rx=12, dash='7 5')
    s.text(770, 120, 'Backbone（YOLOv11s）', size=15, color=BLUE, weight='bold')

    s.box(40, 182, 110, 120, ['输入图像', '896×896×3'], fill=PALE, size=13)
    s.line(150, 242, 188, 242)

    blocks = [
        ('0', 'Conv', ['32, 3×3, s2'], '448²'),
        ('1', 'Conv', ['64, 3×3, s2'], '224²'),
        ('2', 'C3k2 ×2', ['128'], '224²'),
        ('3', 'Conv', ['128, 3×3, s2'], '112²'),
        ('4', 'C3k2 ×2', ['256'], '112²'),
        ('5', 'Conv', ['256, 3×3, s2'], '56²'),
        ('6', 'C3k2 ×2', ['256'], '56²'),
        ('7', 'Conv', ['512, 3×3, s2'], '28²'),
        ('8', 'C3k2 ×2', ['512'], '28²'),
        ('9', 'SPPF', ['512'], '28²'),
        ('插入', 'Triplet', ['Attention', '+300'], '28²'),
        ('10', 'C2PSA', ['512'], '28²'),
    ]
    x0, yb, bw, bh, gap = 190, 195, 86, 94, 12
    step = bw + gap
    centers = []
    for i, (idx, name, detail, size) in enumerate(blocks):
        x = x0 + i * step
        centers.append(x + bw / 2)
        is_triplet = 'Triplet' in name
        s.box(x, yb, bw, bh, [name] + detail,
              fill=ORANGE_BG if is_triplet else WHITE,
              stroke=ORANGE if is_triplet else INK,
              sw=2.4 if is_triplet else 1.4,
              size=10.5,
              color=ORANGE if is_triplet else INK,
              weight='bold' if is_triplet else 'normal')
        s.text(x + bw / 2, yb - 10, idx, size=10, color=GRAY)
        if i in (4, 6, 11):
            # 抽头块的尺寸标注右移，避免竖线穿过文字
            s.text(x + bw / 2 + 7, yb + bh + 17, size, size=10.5,
                   color=GRAY, anchor='start')
        else:
            s.text(x + bw / 2, yb + bh + 17, size, size=10.5, color=GRAY)
        if i:
            s.line(centers[i - 1] + 9, yb + bh / 2, x - 4, yb + bh / 2)

    # Backbone 阶段括号（P1/2 … P5/32）
    stages = [
        (0, 0, 'P1/2  448²', GRAY),
        (1, 2, 'P2/4  224²', GRAY),
        (3, 4, 'P3/8  112²', GREEN),
        (5, 6, 'P4/16  56²', BLUE),
        (7, 11, 'P5/32  28²', ORANGE),
    ]
    for i, j, label, col in stages:
        x1 = x0 + i * step
        x2 = x0 + j * step + bw
        s.path([(x1, 158), (x1, 152), (x2, 152), (x2, 158)],
               color=col, sw=1.2, arrow=False)
        s.text((x1 + x2) / 2, 138, label, size=9.5, color=col, weight='bold')

    # Triplet 强调标注
    tx = centers[10]
    s.path([(tx, yb + bh), (tx, 315), (1100, 315)],
           color=ORANGE, sw=1.6, arrow=False)
    s.text(1092, 315, ['本文改动：SPPF 后插入', 'TripletAttention（+300 参数）'],
           size=10.5, color=ORANGE, anchor='end')

    # P3/P4/P5 抽头：先下到 y=508，再横向接到 Neck 三列顶部
    taps = [
        (4, 625, 400, 'P3  112×112×256', GREEN, 'start', 633),
        (6, 821, 790, 'P4  56×56×256', BLUE, 'start', 829),
        (11, 1311, 1180, 'P5  28×28×512', ORANGE, 'end', 1303),
    ]
    for bi, tx, colx, label, col, anchor, lx in taps:
        s.path([(tx, yb + bh), (tx, 508), (colx, 508), (colx, 516)],
               color=col, sw=1.5, dash='5 4')
        s.text(lx, 340, label, size=10, color=col, anchor=anchor)

    # ================= Neck =================
    s.rect(175, 455, 1190, 310, fill='#FBFCFE', stroke=GREEN, sw=1.6,
           rx=12, dash='7 5')
    s.text(195, 478, 'Neck（FPN + PAN）', size=15, color=GREEN,
           weight='bold', anchor='start')

    col = {'P3': 275, 'P4': 665, 'P5': 1055}
    cw = 250
    for name, x in col.items():
        s.text(x + cw / 2, 500, f'{name} 分支', size=13, color=GREEN, weight='bold')

    # P5 列
    x = col['P5']
    s.box(x, 520, cw, 44, ['P5 特征  28×28×512', '（C2PSA / layer 10 输出）'],
          fill=ORANGE_BG, stroke=ORANGE, size=10.5)
    s.box(x, 573, cw, 48, ['Concat  [P4↓, P5]', 'layer 21'], fill=WHITE, size=10.5)
    s.box(x, 630, cw, 52, ['C3k2 ×2  → P5 out', 'layer 22  512×28²'],
          fill=GREEN_BG, stroke=GREEN, size=10.5)
    s.line(x + cw / 2, 564, x + cw / 2, 571)
    s.line(x + cw / 2, 621, x + cw / 2, 628)

    # P4 列
    x = col['P4']
    s.box(x, 520, cw, 44, ['P4 特征  56×56×256', '（layer 6）'], fill=PALE, size=10.5)
    s.box(x, 573, cw, 48, ['Concat  [P5↑, P4]', 'layer 12'], fill=WHITE, size=10.5)
    s.box(x, 630, cw, 28, ['C3k2 ×2  layer 13'], fill=GREEN_BG, stroke=GREEN, size=10)
    s.box(x, 666, cw, 48, ['Concat  [P3↓, P4]', 'layer 18'], fill=WHITE, size=10.5)
    s.box(x, 720, cw, 34, ['C3k2 ×2 → P4 out', 'layer 19  256×56²'],
          fill=GREEN_BG, stroke=GREEN, size=9.5)
    s.line(x + cw / 2, 564, x + cw / 2, 571)
    s.line(x + cw / 2, 621, x + cw / 2, 628)
    s.line(x + cw / 2, 658, x + cw / 2, 664)
    s.line(x + cw / 2, 714, x + cw / 2, 718)

    # P3 列
    x = col['P3']
    s.box(x, 520, cw, 44, ['P3 特征  112×112×256', '（layer 4）'], fill=PALE, size=10.5)
    s.box(x, 573, cw, 48, ['Concat  [P4↑, P3]', 'layer 15'], fill=WHITE, size=10.5)
    s.box(x, 630, cw, 52, ['C3k2 ×2  → P3 out', 'layer 16  128×112²'],
          fill=GREEN_BG, stroke=GREEN, size=10.5)
    s.line(x + cw / 2, 564, x + cw / 2, 571)
    s.line(x + cw / 2, 621, x + cw / 2, 628)

    # 列间：上采样（P5→P4→P3）与下采样（P3→P4→P5）
    s.line(col['P5'], 542, col['P4'] + cw + 4, 542, color=BLUE, sw=2)
    s.text((col['P5'] + col['P4'] + cw) / 2, 530, 'Upsample ×2', size=9.5, color=BLUE)

    s.path([(col['P4'], 644), (600, 644), (600, 597), (col['P3'] + cw + 4, 597)],
           color=BLUE, sw=2)
    s.text(606, 584, 'Upsample ×2', size=9.5, color=BLUE, anchor='start')

    s.path([(col['P3'] + cw, 656), (590, 656), (590, 690), (col['P4'] - 4, 690)],
           color=GREEN, sw=2)
    s.text(598, 676, 'Conv s2', size=9.5, color=GREEN, anchor='start')

    s.path([(col['P4'] + cw, 737), (985, 737), (985, 597), (col['P5'] - 4, 597)],
           color=GREEN, sw=2)
    s.text(989, 660, 'Conv s2', size=9.5, color=GREEN, anchor='start')

    # ================= Head =================
    s.rect(1395, 455, 365, 310, fill='#FBFCFE', stroke=ORANGE, sw=1.6,
           rx=12, dash='7 5')
    s.text(1577, 478, 'Head（Detect）', size=15, color=ORANGE, weight='bold')

    head_rows = [
        (['P3 112×112×128', 'layer 16'], 'Detect P3', 500),
        (['P4 56×56×256', 'layer 19'], 'Detect P4', 575),
        (['P5 28×28×512', 'layer 22'], 'Detect P5', 650),
    ]
    for src, det, y in head_rows:
        s.box(1408, y, 130, 46, src, fill=PALE, size=9.5)
        s.box(1550, y, 95, 46, [det], fill=ORANGE_BG, stroke=ORANGE, size=10.5)
        s.text(1656, y + 17, '4 box', size=9.0, color=GRAY, anchor='start')
        s.text(1656, y + 31, '3 cls', size=9.0, color=GRAY, anchor='start')
        s.line(1538, y + 23, 1548, y + 23, color=ORANGE)
        s.line(1645, y + 23, 1652, y + 23, color=ORANGE)

    s.text(1577, 710, '每个 Detect 头输出 3 类分数（三个尺度共享）：',
           size=9.5, color=GRAY)
    cls_chips = [
        (1408, 108, '断股 break'),
        (1522, 108, '雷击 thunderbolt'),
        (1636, 114, '磨损 wear'),
    ]
    for cx, cw, label in cls_chips:
        s.box(cx, 720, cw, 28, [label], fill=ORANGE_BG, stroke=ORANGE, size=9.5)
    s.text(1577, 760, '每尺度 4 + 3 = 7 通道；三尺度合计 3 × 7 = 21',
           size=9.0, color=GRAY)

    # Neck → Head：三条输出经下方总线进入 Head 左 rail
    s.path([(527, 656), (527, 775), (1384, 775), (1384, 500), (1398, 500)],
           color=ORANGE, sw=1.8)
    s.path([(917, 737), (917, 783), (1370, 783), (1370, 575), (1390, 575)],
           color=ORANGE, sw=1.8)
    s.path([(1307, 656), (1307, 791), (1356, 791), (1356, 650), (1390, 650)],
           color=ORANGE, sw=1.8)

    # ================= 底部模块细节 =================
    py, ph = 810, 350
    panels = [
        (40, 210, 'Conv'),
        (268, 310, 'C3k2（False / True）'),
        (596, 230, 'SPPF'),
        (844, 270, 'C2PSA'),
        (1132, 420, 'TripletAttention（本文）'),
        (1570, 190, 'Detect'),
    ]
    for x, w, title in panels:
        is_triplet = 'Triplet' in title
        s.rect(x, py, w, ph, fill=ORANGE_BG if is_triplet else '#FAFBFD',
               stroke=ORANGE if is_triplet else GRID, sw=1.6, rx=10, dash='5 4')
        s.text(x + w / 2, py + 14, title, size=12.5,
               color=ORANGE if is_triplet else BLUE, weight='bold')

    # --- Conv ---
    x, w = 40, 210
    for j, (label, yy) in enumerate([('Conv2d', 860), ('BatchNorm2d', 925),
                                     ('SiLU', 990)]):
        s.box(x + 40, yy, 130, 42, [label], fill=WHITE, size=10.5)
        if j:
            s.line(x + 105, yy - 23, x + 105, yy - 4)
    s.text(x + w / 2, 1075, ['Conv = Conv2d + BN + SiLU', '（k=3×3，s=2 时下采样）'],
           size=9.2, color=GRAY)

    # --- C3k2 ---
    x, w = 268, 310
    s.box(x + 12, 865, 58, 38, ['输入'], fill=PALE, size=9.5)
    s.box(x + 88, 865, 66, 38, ['Conv 1×1'], fill=WHITE, size=9)
    s.box(x + 170, 836, 125, 40, ['Bottleneck ×n', 'k=2（True/False）'],
          fill=WHITE, size=8.8)
    s.box(x + 170, 884, 125, 40, ['Bottleneck ×n', 'shortcut'], fill=WHITE, size=8.8)
    s.box(x + 88, 942, 66, 38, ['Concat'], fill=PALE, size=9.5)
    s.box(x + 170, 942, 66, 38, ['Conv 1×1'], fill=WHITE, size=9)
    s.box(x + 158, 1002, 90, 36, ['输出'], fill=PALE, size=9.5)
    s.line(x + 70, 884, x + 86, 884)
    s.line(x + 154, 884, x + 168, 856)
    s.line(x + 232, 876, x + 232, 882)
    s.path([(x + 232, 924), (x + 232, 932), (x + 154, 932), (x + 154, 952)],
           color=INK, sw=1.6)
    s.line(x + 154, 961, x + 168, 961)
    s.line(x + 154, 884, x + 121, 940)
    s.line(x + 203, 980, x + 203, 1000)
    s.text(x + w / 2, 1075, ['C3k2 = 1×1 Conv 分支 + k2 Bottleneck 分支',
                             'Concat 后 1×1 Conv；True 分支带 shortcut'],
           size=8.8, color=GRAY)

    # --- SPPF ---
    x, w = 596, 230
    s.box(x + 30, 865, 70, 40, ['Conv 1×1'], fill=WHITE, size=9)
    s.box(x + 120, 845, 70, 34, ['MaxPool', '5×5'], fill=WHITE, size=8.8)
    s.box(x + 120, 890, 70, 34, ['MaxPool', '5×5'], fill=WHITE, size=8.8)
    s.box(x + 120, 935, 70, 34, ['MaxPool', '5×5'], fill=WHITE, size=8.8)
    s.box(x + 30, 950, 70, 40, ['Concat'], fill=PALE, size=9.5)
    s.box(x + 30, 1005, 70, 40, ['Conv 1×1'], fill=WHITE, size=9)
    s.line(x + 100, 885, x + 118, 862)
    s.line(x + 155, 879, x + 155, 888)
    s.line(x + 155, 924, x + 155, 933)
    s.line(x + 118, 952, x + 102, 952)
    s.line(x + 65, 905, x + 65, 948)
    s.line(x + 65, 990, x + 65, 1003)
    s.text(x + w / 2, 1080, ['串联 5×5 MaxPool 扩大感受野', 'Concat 三个尺度后 1×1 Conv 融合'],
           size=8.8, color=GRAY)

    # --- C2PSA ---
    x, w = 844, 270
    s.box(x + 15, 885, 58, 38, ['输入'], fill=PALE, size=9.5)
    s.box(x + 90, 885, 62, 38, ['Conv 1×1'], fill=WHITE, size=9)
    s.box(x + 168, 855, 78, 34, ['PSA', '多头自注意力'], fill=WHITE, size=8.0)
    s.box(x + 168, 900, 78, 34, ['PSA', '多头自注意力'], fill=WHITE, size=8.0)
    s.box(x + 90, 955, 62, 38, ['Concat'], fill=PALE, size=9.5)
    s.box(x + 170, 955, 62, 38, ['Conv 1×1'], fill=WHITE, size=9)
    s.line(x + 73, 904, x + 88, 904)
    s.line(x + 152, 904, x + 168, 872)
    s.line(x + 152, 904, x + 168, 917)
    s.line(x + 206, 889, x + 206, 898)
    s.line(x + 206, 934, x + 206, 953)
    s.line(x + 152, 974, x + 168, 974)
    s.line(x + 121, 923, x + 121, 953)
    s.text(x + w / 2, 1075, ['C2PSA = 1×1 Conv + PSA + Concat + 1×1 Conv',
                             'PSA：多头自注意力，强化全局上下文'],
           size=8.8, color=GRAY)

    # --- TripletAttention ---
    x, w = 1132, 420
    s.text(x + 18, 844, '输入 x（B,C,H,W）', size=9.5, color=INK, anchor='start')
    branches = [('C×H', 875), ('C×W', 930), ('H×W', 985)]
    for label, yy in branches:
        s.text(x + 18, yy, label, size=10, color=ORANGE, anchor='start',
               weight='bold')
        bx = x + 60
        for j, t in enumerate(['ZPool', '7×7 Conv', 'BN', 'Sigmoid', '⊗']):
            s.box(bx + j * 58, yy - 15, 52, 30, [t], fill=WHITE,
                  stroke=ORANGE if t in ('BN', 'Sigmoid') else GRID,
                  size=8.5, color=ORANGE if t == 'BN' else INK)
        for j in range(4):
            s.line(bx + j * 58 + 52, yy, bx + (j + 1) * 58, yy)
        s.line(bx + 4 * 58 + 52, yy, bx + 4 * 58 + 58, yy, color=ORANGE)
    s.box(x + 350, 860, 58, 140, ['平均', '1/3', 'Σ'],
          fill=ORANGE_BG, stroke=ORANGE, size=9)
    s.text(x + 210, 1040,
           '每支路：sigmoid(BN(Conv7×7(ZPool(x_rot)))) × x_rot',
           size=8.6, color=GRAY)
    s.text(x + 210, 1062,
           '输出 = 1/3 Σ 三分支；BN 消 DC、钉尺度（CBAM 无此层）',
           size=8.6, color=GRAY)
    s.text(x + 210, 1090, '插入位置：SPPF → Triplet → C2PSA（P5/32）',
           size=9.2, color=ORANGE, weight='bold')

    # --- Detect ---
    x, w = 1570, 190
    s.box(x + 15, 875, 70, 42, ['P3/P4/P5', '特征'], fill=PALE, size=8.8)
    s.box(x + 105, 875, 70, 42, ['3×3 Conv', '×2'], fill=WHITE, size=8.8)
    s.box(x + 25, 945, 140, 42, ['输出', 'box(4) + cls(3)'],
          fill=ORANGE_BG, stroke=ORANGE, size=8.8)
    s.line(x + 85, 896, x + 103, 896)
    s.line(x + 140, 917, x + 140, 943)
    s.text(x + w / 2, 1015, ['anchor-free', '每尺度 4 + 3 = 7 通道'],
           size=8.8, color=GRAY)
    s.text(x + w / 2, 1060, ['P3 112²', 'P4 56²', 'P5 28²'],
           size=8.8, color=GRAY)

    # ================= 图例 =================
    s.raw(f'<rect x="40" y="1175" width="{W - 80}" height="34" '
          f'fill="#FAFBFD" stroke="{GRID}" stroke-width="0.8" rx="5"/>')
    items = [
        (ORANGE_BG, ORANGE, None, '本文改动（Triplet）'),
        (GREEN_BG, GREEN, None, 'Neck 融合 / P3 抽头'),
        (PALE, GRID, None, '特征图 / Concat'),
        (WHITE, INK, None, '卷积 / 基础模块'),
    ]
    lx = 60
    for fill, stroke, dash, label in items:
        s.rect(lx, 1183, 18, 14, fill=fill, stroke=stroke, sw=1.2, rx=3,
               dash=dash)
        s.text(lx + 26, 1190, label, size=10, color=INK, anchor='start')
        lx += 235
    s.rect(lx, 1183, 18, 14, fill='none', stroke=GRAY, sw=1.2, rx=3,
           dash='4 3')
    s.text(lx + 26, 1190, '虚线 = 分组 / 抽头', size=10, color=INK, anchor='start')
    s.text(W - 60, 1190,
           '输入 896×896；P3/8=112²，P4/16=56²，P5/32=28²；anchor-free，nc=3',
           size=10, color=GRAY, anchor='end')

    return s.render()


def edge_candidates():
    return [
        r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
        r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    ]


def render_external(svg_path):
    exe = os.environ.get('EDGE')
    if not exe:
        for c in edge_candidates():
            if os.path.exists(c):
                exe = c
                break
    if not exe:
        print('未找到 Edge/Chrome，跳过 PNG/PDF 渲染；SVG 已生成。')
        return
    html = os.path.join(OUT, 'yolo11_triplet_arch_v2.html')
    with open(html, 'w', encoding='utf-8') as f:
        f.write('<html><head><meta charset="utf-8">'
                '<style>@page{size:%dpx %dpx;margin:0}'
                'html,body{margin:0;padding:0;width:%dpx;height:%dpx;overflow:hidden}'
                'svg{display:block}</style></head><body>' % (W, H, W, H))
        f.write(open(svg_path, encoding='utf-8').read())
        f.write('</body></html>')
    png = os.path.join(OUT, 'yolo11_triplet_arch_v2.png')
    pdf = os.path.join(OUT, 'yolo11_triplet_arch_v2.pdf')
    url = 'file:///' + html.replace('\\', '/')
    for cmd in (
        [exe, '--headless=new', '--disable-gpu', '--hide-scrollbars',
         f'--window-size={W},{H}', f'--screenshot={png}', url],
        [exe, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
         f'--print-to-pdf={pdf}', '--no-margins', url],
    ):
        try:
            subprocess.run(cmd, check=True, timeout=120,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print('rendered', cmd[4])
        except Exception as e:
            print('render failed', e)


def write_drawio():
    """生成 v2 的可编辑 draw.io 初始文件（结构近似，便于继续手调）。"""
    cells = []
    cid = [1]

    def nid():
        cid[0] += 1
        return f'c{cid[0]}'

    def box(x, y, w, h, label, fill=WHITE, stroke=INK, size=12, bold=False):
        i = nid()
        style = (f'rounded=1;whiteSpace=wrap;html=1;fillColor={fill};'
                 f'strokeColor={stroke};fontColor={INK};fontSize={size};'
                 f'fontFamily=Microsoft YaHei;'
                 + ('fontStyle=1;' if bold else ''))
        cells.append(
            f'<mxCell id="{i}" value="{sx.escape(label)}" style="{style}" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def edge(a, b, label='', dash=False, color=INK):
        i = nid()
        style = (f'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;'
                 f'strokeColor={color};strokeWidth=1.5;'
                 + ('dashed=1;' if dash else ''))
        cells.append(
            f'<mxCell id="{i}" value="{sx.escape(label)}" style="{style}" '
            f'edge="1" parent="1" source="{a}" target="{b}">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>')

    box(30, 10, 1740, 60,
        'YOLOv11s + TripletAttention 网络结构示意图（896×896，nc=3）',
        fill='none', stroke='none', size=22, bold=True)
    inp = box(30, 170, 110, 100, '输入图像\n896×896×3', fill=PALE, size=13)
    box(170, 100, 1200, 340, '', fill='#FBFCFE', stroke=BLUE)
    box(180, 105, 220, 30, 'Backbone（YOLOv11s）', fill='none', stroke='none',
        size=14, bold=True)

    blocks = [
        ('0  Conv\n32, 3×3, s2\n448²', WHITE, ''),
        ('1  Conv\n64, 3×3, s2\n224²', WHITE, ''),
        ('2  C3k2×2\n128\n224²', WHITE, ''),
        ('3  Conv\n128, 3×3, s2\n112²', WHITE, ''),
        ('4  C3k2×2\n256\n112²', WHITE, ''),
        ('5  Conv\n256, 3×3, s2\n56²', WHITE, ''),
        ('6  C3k2×2\n256\n56²', WHITE, ''),
        ('7  Conv\n512, 3×3, s2\n28²', WHITE, ''),
        ('8  C3k2×2\n512\n28²', WHITE, ''),
        ('9  SPPF\n512\n28²', WHITE, ''),
        ('插入  Triplet\nAttention\n+300 参数', ORANGE_BG, ORANGE),
        ('10  C2PSA\n512\n28²', WHITE, ''),
    ]
    prev = inp
    ids = []
    for i, (label, fill, stroke) in enumerate(blocks):
        b = box(190 + i * 98, 195, 86, 100, label, fill=fill,
                stroke=stroke or INK, size=10, bold=bool(stroke))
        edge(prev, b)
        ids.append(b)
        prev = b

    nb = box(170, 455, 1200, 310, 'Neck（FPN + PAN）', fill='#FBFCFE',
             stroke=GREEN, size=14, bold=True)
    neck = {}
    specs = [
        ('P3', 275, 500, 'P3 特征\n112×112×256\nConcat[P4↑,P3] #15 → C3k2 #16'),
        ('P4', 665, 500, 'P4 特征\n56×56×256\nConcat[P5↑,P4] #12 → C3k2 #13\nConcat[P3↓,P4] #18 → C3k2 #19'),
        ('P5', 1055, 500, 'P5 特征\n28×28×512\nConcat[P4↓,P5] #21 → C3k2 #22'),
    ]
    for name, x, y, label in specs:
        neck[name] = box(x, y, 240, 150, label, fill=GREEN_BG, stroke=GREEN,
                         size=10)
    edge(neck['P5'], neck['P4'], 'Upsample ×2', color=BLUE)
    edge(neck['P4'], neck['P3'], 'Upsample ×2', color=BLUE)
    edge(neck['P3'], neck['P4'], 'Conv s2', color=GREEN)
    edge(neck['P4'], neck['P5'], 'Conv s2', color=GREEN)
    edge(ids[11], neck['P5'], 'P5')
    edge(ids[6], neck['P4'], 'P4', dash=True)
    edge(ids[4], neck['P3'], 'P3', dash=True)

    box(1400, 455, 360, 310, 'Head（Detect）', fill='#FBFCFE', stroke=ORANGE,
        size=14, bold=True)
    h1 = box(1430, 520, 300, 50, 'Detect P3\n112×112×128 → 4 + 3 = 7',
             fill=ORANGE_BG, stroke=ORANGE, size=10)
    h2 = box(1430, 600, 300, 50, 'Detect P4\n56×56×256 → 4 + 3 = 7',
             fill=ORANGE_BG, stroke=ORANGE, size=10)
    h3 = box(1430, 680, 300, 50, 'Detect P5\n28×28×512 → 4 + 3 = 7',
             fill=ORANGE_BG, stroke=ORANGE, size=10)
    edge(neck['P3'], h1, color=ORANGE)
    edge(neck['P4'], h2, color=ORANGE)
    edge(neck['P5'], h3, color=ORANGE)

    detail = [
        (30, 810, 210, 350, 'Conv = Conv2d + BN + SiLU'),
        (258, 810, 310, 350,
         'C3k2 = 1×1 Conv + k2 Bottleneck ×n + Concat + 1×1 Conv'),
        (586, 810, 230, 350,
         'SPPF = 1×1 Conv + 串行 5×5 MaxPool ×3 + Concat + 1×1 Conv'),
        (834, 810, 270, 350,
         'C2PSA = 1×1 Conv + PSA 多头自注意力 + Concat + 1×1 Conv'),
        (1122, 810, 420, 350,
         'TripletAttention（本文）\n三分支 C×H / C×W / H×W\n'
         '每支：ZPool → 7×7 Conv → BN → Sigmoid → ⊗\n'
         '输出 = 1/3 Σ 三分支\n插入：SPPF → Triplet → C2PSA'),
        (1560, 810, 200, 350,
         'Detect\nanchor-free\nbox(4) + cls(3)\n每尺度 7 通道\nP3/P4/P5'),
    ]
    for x, y, w, h, label in detail:
        fill = ORANGE_BG if 'Triplet' in label else '#FAFBFD'
        stroke = ORANGE if 'Triplet' in label else GRID
        box(x, y, w, h, label, fill=fill, stroke=stroke, size=11,
            bold='Triplet' in label)

    xml = ('<mxfile host="app.diagrams.net" modified="2026-10-20T00:00:00.000Z" '
           'agent="Claude" version="24.0.0">'
           '<diagram id="yolo11-triplet-v2" name="YOLOv11s+Triplet-v2">'
           '<mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" '
           'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
           'pageWidth="1800" pageHeight="1240" math="0" shadow="0"><root>'
           '<mxCell id="0"/><mxCell id="1" parent="0"/>'
           + ''.join(cells) +
           '</root></mxGraphModel></diagram></mxfile>')
    path = os.path.join(OUT, 'yolo11_triplet_arch_v2.drawio')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(xml)
    print('wrote', path)


def main():
    os.makedirs(OUT, exist_ok=True)
    svg = build()
    svg_path = os.path.join(OUT, 'yolo11_triplet_arch_v2.svg')
    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(svg)
    print('wrote', svg_path)
    write_drawio()
    if '--svg-only' not in sys.argv:
        render_external(svg_path)


if __name__ == '__main__':
    main()
