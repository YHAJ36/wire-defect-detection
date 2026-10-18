# -*- coding: utf-8 -*-
"""生成 YOLOv11s + TripletAttention 网络结构示意图（SVG / PNG / PDF / draw.io）。

依据本项目真实结构（非论文示意图）：
  · YOLO11s，输入 896，nc=3；layer 0-23 与 ultralytics 8.3.180 一致；
  · TripletAttention 插在 SPPF(layer 9) 之后、C2PSA(layer 10) 之前，
    只注入这一次（--mode neck_sppf），+300 参数；
  · 输出尺度 P3/8=112²、P4/16=56²、P5/32=28²（896 输入）。

产物：
  report/figures/yolo11_triplet_arch.svg      矢量源文件（可用浏览器/Illustrator 打开）
  report/figures/yolo11_triplet_arch.png      预览图（Edge headless 渲染）
  report/figures/yolo11_triplet_arch.pdf      打印版
  report/figures/yolo11_triplet_arch.drawio   可编辑 draw.io 源文件（图形块与连线）

用法：
  D:/yolo-env/Scripts/python.exe report/make_arch_diagram.py
  # 若 Edge 不在默认路径，设 EDGE=... 或只生成 SVG
"""
import os
import subprocess
import sys
import xml.sax.saxutils as sx

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'report', 'figures')
W, H = 1800, 1180

# 配色：与项目报告保持一致（深蓝 + 浅蓝 + 橙色强调）
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
             anchor='middle', italic=False, opacity=1.0):
        if isinstance(lines, str):
            lines = [lines]
        style = (f'font-family:{FONT};font-size:{size}px;fill:{color};'
                 f'font-weight:{weight};text-anchor:{anchor};'
                 f'dominant-baseline:middle;opacity:{opacity}')
        if italic:
            style += ';font-style:italic'
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
        self.text(x + w / 2, y + h / 2, lines, size=size, color=color, weight=weight)

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

    def circle(self, x, y, r, fill=WHITE, stroke=INK, sw=1.4):
        self.raw(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"/>')

    def render(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                f'viewBox="0 0 {W} {H}">'
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
                'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>'
                '<marker id="arrowOrange" viewBox="0 0 10 10" refX="9" refY="5" '
                'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{ORANGE}"/></marker>'
                '</defs>')
        return head + ''.join(self.parts) + '</svg>'


def build():
    s = SVG()
    s.raw(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.text(40, 34, 'YOLOv11s + TripletAttention 网络结构示意图', size=27,
           color=INK, weight='bold', anchor='start')
    s.text(40, 70, '基线 YOLOv11s（896×896，nc=3）；在 SPPF 之后插入 TripletAttention（仅 1 处，+300 参数）',
           size=15, color=GRAY, anchor='start')

    # ================= 主结构：Input + Backbone =================
    s.box(40, 165, 110, 120, ['输入图像', '896×896×3'], fill=PALE, size=13)
    s.line(150, 225, 188, 225)

    s.rect(175, 105, 1190, 290, fill='#FBFCFE', stroke=BLUE, sw=1.6, rx=12, dash='7 5')
    s.text(770, 124, 'Backbone（YOLOv11s）', size=15, color=BLUE, weight='bold')

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
        ('插入', 'Triplet ★', ['Attention'], '28²'),
        ('10', 'C2PSA', ['512'], '28²'),
    ]
    x0, yb, bw, bh, gap = 195, 175, 82, 92, 10
    centers = []
    for i, (idx, name, detail, size) in enumerate(blocks):
        x = x0 + i * (bw + gap)
        centers.append(x + bw / 2)
        fill = ORANGE_BG if 'Triplet' in name else WHITE
        stroke = ORANGE if 'Triplet' in name else INK
        sw = 2.4 if 'Triplet' in name else 1.4
        s.box(x, yb, bw, bh, [name] + detail, fill=fill, stroke=stroke, sw=sw,
              size=11.5, color=ORANGE if 'Triplet' in name else INK,
              weight='bold' if 'Triplet' in name else 'normal')
        s.text(x + bw / 2, yb - 14, idx, size=10, color=GRAY)
        s.text(x + bw / 2, yb + bh + 17, size, size=10.5, color=GRAY)
        if i:
            s.line(centers[i - 1] + 9, yb + bh / 2, x - 4, yb + bh / 2)

    # Triplet 强调标注
    tx = centers[10]
    s.path([(tx, yb + bh + 30), (tx, yb + bh + 52), (tx + 60, yb + bh + 52)],
           color=ORANGE, sw=1.6, arrow=False)
    s.text(tx + 68, yb + bh + 52, ['本文改动：SPPF 后插入', 'TripletAttention（+300）'],
           size=10.5, color=ORANGE, anchor='start')

    # P3/P4/P5 抽头（虚线，向下进入 Neck）
    tap_y = yb + bh + 60
    for bi, label, col in [(4, 'P3  112×112×256', GREEN), (6, 'P4  56×56×256', BLUE),
                           (11, 'P5  28×28×512', ORANGE)]:
        cx = centers[bi]
        s.line(cx, yb + bh + 8, cx, 425, color=col, sw=1.5, dash='5 4')
        s.text(cx + 6, 410, label, size=10, color=col, anchor='start')

    # ================= Neck =================
    s.rect(175, 430, 1190, 300, fill='#FBFCFE', stroke=GREEN, sw=1.6, rx=12, dash='7 5')
    s.text(770, 450, 'Neck（FPN + PAN）', size=15, color=GREEN, weight='bold')

    # 三列
    col = {'P3': 275, 'P4': 665, 'P5': 1055}
    cw = 250
    # 列标题
    for name, x in col.items():
        s.text(x + cw / 2, 477, f'{name} 分支', size=13, color=GREEN, weight='bold')

    # P5 列
    x = col['P5']
    s.box(x, 495, cw, 44, ['P5 特征  28×28×512', '（C2PSA 输出）'], fill=ORANGE_BG, stroke=ORANGE, size=10.5)
    s.box(x, 548, cw, 48, ['Concat  [P4↓, P5]', 'layer 21'], fill=WHITE, size=10.5)
    s.box(x, 605, cw, 52, ['C3k2 ×2  → P5 out', 'layer 22  512×28²'], fill=GREEN_BG, stroke=GREEN, size=10.5)
    s.line(x + cw / 2, 539, x + cw / 2, 546)
    s.line(x + cw / 2, 596, x + cw / 2, 603)

    # P4 列
    x = col['P4']
    s.box(x, 495, cw, 44, ['P4 特征  56×56×256', '（layer 6）'], fill=PALE, size=10.5)
    s.box(x, 548, cw, 48, ['Concat  [P5↑, P4]', 'layer 12'], fill=WHITE, size=10.5)
    s.box(x, 605, cw, 28, ['C3k2 ×2  layer 13'], fill=GREEN_BG, stroke=GREEN, size=10)
    s.box(x, 640, cw, 48, ['Concat  [P3↓, P4]', 'layer 18'], fill=WHITE, size=10.5)
    s.box(x, 694, cw, 34, ['C3k2 ×2 → P4 out', 'layer 19  256×56²'], fill=GREEN_BG, stroke=GREEN, size=9)
    s.line(x + cw / 2, 539, x + cw / 2, 546)
    s.line(x + cw / 2, 596, x + cw / 2, 603)
    s.line(x + cw / 2, 633, x + cw / 2, 638)
    s.line(x + cw / 2, 688, x + cw / 2, 692)

    # P3 列
    x = col['P3']
    s.box(x, 495, cw, 44, ['P3 特征  112×112×256', '（layer 4）'], fill=PALE, size=10.5)
    s.box(x, 548, cw, 48, ['Concat  [P4↑, P3]', 'layer 15'], fill=WHITE, size=10.5)
    s.box(x, 605, cw, 52, ['C3k2 ×2  → P3 out', 'layer 16  128×112²'], fill=GREEN_BG, stroke=GREEN, size=10.5)
    s.line(x + cw / 2, 539, x + cw / 2, 546)
    s.line(x + cw / 2, 596, x + cw / 2, 603)

    # 列间箭头：上采样（P5→P4→P3）与下采样（P3→P4→P5）
    y_up = 523
    s.line(col['P5'], y_up, col['P4'] + cw + 4, y_up, color=BLUE, sw=2)
    s.text((col['P5'] + col['P4'] + cw) / 2, y_up - 12, 'Upsample ×2', size=9.5, color=BLUE)
    # P4→P3 的上采样取 layer 13（C3k2 输出），不是 layer 6
    s.path([(col['P4'], 619), (600, 619), (600, 572), (col['P3'] + cw + 4, 572)],
           color=BLUE, sw=2)
    s.text(628, 599, 'Upsample ×2', size=9.5, color=BLUE)

    # P3 out → P4 的 Concat[P3↓,P4]（layer 18）
    s.path([(col['P3'] + cw, 631), (590, 631), (590, 664), (col['P4'] - 4, 664)],
           color=GREEN, sw=2)
    s.text(622, 652, 'Conv s2', size=9.5, color=GREEN)
    # P4 out → P5 的 Concat[P4↓,P5]（layer 21）
    s.path([(col['P4'] + cw, 711), (985, 711), (985, 572), (col['P5'] - 4, 572)],
           color=GREEN, sw=2)
    s.text(1012, 645, 'Conv s2', size=9.5, color=GREEN)

    # ================= Head =================
    s.rect(1395, 430, 365, 300, fill='#FBFCFE', stroke=ORANGE, sw=1.6, rx=12, dash='7 5')
    s.text(1577, 450, 'Head（Detect）', size=15, color=ORANGE, weight='bold')
    head_rows = [
        (['P3 112×112×128', 'layer 16'], 'Detect P3', '3×(4+3)', 495),
        (['P4 56×56×256', 'layer 19'], 'Detect P4', '3×(4+3)', 575),
        (['P5 28×28×512', 'layer 22'], 'Detect P5', '3×(4+3)', 655),
    ]
    for src, det, out, y in head_rows:
        s.box(1408, y, 130, 46, src, fill=PALE, size=9.5)
        s.box(1550, y, 95, 46, [det], fill=ORANGE_BG, stroke=ORANGE, size=10.5)
        s.text(1705, y + 23, out, size=9.5, color=GRAY)
        s.line(1538, y + 23, 1548, y + 23, color=ORANGE)
        s.line(1645, y + 23, 1662, y + 23, color=ORANGE)
    s.text(1577, 718, 'anchor-free；类别：break / thunderbolt / wear', size=9.5, color=GRAY)

    # Neck → Head：三条输出汇入 Head 左侧总线，再分三行送入 Detect
    s.path([(col['P3'] + cw, 631), (545, 631), (545, 740), (1388, 740)],
           color=ORANGE, sw=2, arrow=False)
    s.path([(col['P4'] + cw, 711), (935, 711), (935, 725), (1388, 725)],
           color=ORANGE, sw=2, arrow=False)
    s.path([(col['P5'] + cw, 631), (1388, 631)], color=ORANGE, sw=2, arrow=False)
    s.line(1388, 740, 1388, 518, color=ORANGE, sw=2, arrow=False)
    for y in (518, 598, 678):
        s.line(1388, y, 1398, y, color=ORANGE, sw=2)

    # ================= 模块细节 =================
    py, ph = 790, 330
    panels = [
        (40, 220, 'Conv'),
        (280, 300, 'C3k2（False / True）'),
        (600, 230, 'SPPF'),
        (850, 270, 'C2PSA'),
        (1140, 420, 'TripletAttention（本文）'),
        (1580, 200, 'Detect'),
    ]
    for x, w, title in panels:
        stroke = ORANGE if 'Triplet' in title else GRID
        fill = ORANGE_BG if 'Triplet' in title else '#FAFBFD'
        s.rect(x, py, w, ph, fill=fill, stroke=stroke, sw=1.6, rx=10, dash='5 4')
        s.text(x + w / 2, py + 20, title, size=13,
               color=ORANGE if 'Triplet' in title else BLUE, weight='bold')

    # --- Conv 面板 ---
    x, w = 40, 220
    for j, (label, yy) in enumerate([('Conv2d', 845), ('BatchNorm2d', 915), ('SiLU', 985)]):
        s.box(x + 45, yy, 130, 42, [label], fill=WHITE, size=10.5)
        if j:
            s.line(x + 110, yy - 28, x + 110, yy - 4)
    s.text(x + w / 2, 1060, ['Conv = Conv2d + BN + SiLU', '（k=3×3，s=2 时下采样）'], size=9.5, color=GRAY)

    # --- C3k2 面板 ---
    x, w = 280, 300
    s.box(x + 15, 860, 62, 40, ['输入'], fill=PALE, size=10)
    s.box(x + 95, 860, 62, 40, ['Conv 1×1'], fill=WHITE, size=9.5)
    s.box(x + 175, 825, 105, 34, ['Bottleneck ×n', 'k=2（True/False）'], fill=WHITE, size=9)
    s.box(x + 175, 868, 105, 34, ['Bottleneck ×n', 'shortcut'], fill=WHITE, size=9)
    s.box(x + 95, 930, 62, 40, ['Concat'], fill=PALE, size=10)
    s.box(x + 175, 930, 62, 40, ['Conv 1×1'], fill=WHITE, size=9.5)
    s.box(x + 120, 990, 80, 36, ['输出'], fill=PALE, size=10)
    s.line(x + 77, 880, x + 93, 880)
    s.line(x + 157, 880, x + 173, 850)
    s.line(x + 157, 880, x + 173, 885)
    s.line(x + 227, 902, x + 227, 928)
    s.line(x + 157, 950, x + 173, 950)
    s.line(x + 157, 880, x + 126, 928)
    s.line(x + 160, 968, x + 160, 988)
    s.text(x + w / 2, 1060, ['C3k2 = 1×1 Conv 分支 + k2 Bottleneck 分支', 'Concat 后 1×1 Conv；True 分支带 shortcut'], size=9, color=GRAY)

    # --- SPPF 面板 ---
    x, w = 600, 230
    s.box(x + 35, 865, 70, 40, ['Conv 1×1'], fill=WHITE, size=9.5)
    s.box(x + 125, 845, 70, 34, ['MaxPool', '5×5'], fill=WHITE, size=9)
    s.box(x + 125, 885, 70, 34, ['MaxPool', '5×5'], fill=WHITE, size=9)
    s.box(x + 125, 925, 70, 34, ['MaxPool', '5×5'], fill=WHITE, size=9)
    s.box(x + 35, 940, 70, 40, ['Concat'], fill=PALE, size=10)
    s.box(x + 35, 995, 70, 36, ['Conv 1×1'], fill=WHITE, size=9.5)
    s.line(x + 105, 885, x + 123, 862)
    s.line(x + 105, 885, x + 123, 902)
    s.line(x + 105, 885, x + 123, 942)
    s.line(x + 160, 879, x + 160, 883)
    s.line(x + 160, 919, x + 160, 923)
    s.line(x + 70, 958, x + 120, 958)
    s.line(x + 70, 980, x + 70, 993)
    s.text(x + w / 2, 1070, ['串联 5×5 MaxPool 扩大感受野，', 'Concat 三个尺度后 1×1 Conv 融合'], size=9, color=GRAY)

    # --- C2PSA 面板 ---
    x, w = 850, 270
    s.box(x + 20, 880, 62, 40, ['输入'], fill=PALE, size=10)
    s.box(x + 100, 880, 62, 40, ['Conv 1×1'], fill=WHITE, size=9.5)
    s.box(x + 180, 850, 72, 34, ['PSA', '多头自注意力'], fill=WHITE, size=9)
    s.box(x + 180, 895, 72, 34, ['PSA', '多头自注意力'], fill=WHITE, size=9)
    s.box(x + 100, 950, 62, 40, ['Concat'], fill=PALE, size=10)
    s.box(x + 185, 950, 62, 40, ['Conv 1×1'], fill=WHITE, size=9.5)
    s.line(x + 82, 900, x + 98, 900)
    s.line(x + 162, 900, x + 178, 867)
    s.line(x + 162, 900, x + 178, 912)
    s.line(x + 216, 884, x + 216, 893)
    s.line(x + 216, 929, x + 216, 948)
    s.line(x + 162, 970, x + 183, 970)
    s.line(x + 131, 920, x + 131, 948)
    s.text(x + w / 2, 1070, ['C2PSA = 1×1 Conv + PSA 自注意力 + Concat + 1×1 Conv', 'PSA：多头自注意力，强化全局上下文'], size=9, color=GRAY)

    # --- TripletAttention 面板 ---
    x, w = 1140, 420
    s.text(x + 18, 824, '输入 x（B,C,H,W）', size=9.5, color=INK, anchor='start')
    branches = [('C×W', 855), ('H×C', 910), ('H×W', 965)]
    for label, yy in branches:
        s.text(x + 18, yy, label, size=10, color=ORANGE, anchor='start', weight='bold')
        bx = x + 60
        for j, t in enumerate(['ZPool', '7×7 Conv', 'BN', 'Sigmoid', '⊗']):
            s.box(bx + j * 58, yy - 15, 52, 30, [t], fill=WHITE,
                  stroke=ORANGE if t in ('BN', 'Sigmoid') else GRID,
                  size=8.5, color=ORANGE if t == 'BN' else INK)
        s.line(bx + 52, yy, bx + 58, yy)
        s.line(bx + 110, yy, bx + 116, yy)
        s.line(bx + 168, yy, bx + 174, yy)
        s.line(bx + 226, yy, bx + 232, yy)
        s.line(bx + 344, yy, bx + 350, yy, color=ORANGE)
    s.box(x + 350, 865, 58, 90, ['平均', '1/3', 'Σ'], fill=ORANGE_BG, stroke=ORANGE, size=9)
    s.line(x + 350, 855, x + 350, 965, color=ORANGE, sw=1.6, arrow=False)
    s.text(x + 210, 1010, '每支路：sigmoid(BN(Conv7×7(ZPool(x_rot)))) × x_rot', size=8.8, color=GRAY)
    s.text(x + 210, 1032, '输出 = 1/3 Σ 三分支；BN 消 DC、钉尺度（CBAM 无此层）', size=8.8, color=GRAY)
    s.text(x + 210, 1058, '插入位置：SPPF → Triplet → C2PSA（P5/32）', size=9.2, color=ORANGE, weight='bold')

    # --- Detect 面板 ---
    x, w = 1580, 200
    s.box(x + 20, 855, 70, 42, ['P3/P4/P5', '特征'], fill=PALE, size=9)
    s.box(x + 110, 855, 70, 42, ['3×3 Conv', '×2'], fill=WHITE, size=9)
    s.box(x + 35, 925, 130, 42, ['输出', 'box(4) + cls(3)'], fill=ORANGE_BG, stroke=ORANGE, size=9)
    s.line(x + 90, 876, x + 108, 876)
    s.line(x + 145, 897, x + 145, 923)
    s.text(x + w / 2, 1000, ['anchor-free', '每尺度 3×(4+3)'], size=9, color=GRAY)
    s.text(x + w / 2, 1035, ['P3 112²', 'P4 56²', 'P5 28²'], size=9, color=GRAY)

    # 图例
    s.raw(f'<rect x="40" y="1138" width="{W - 80}" height="30" fill="#FAFBFD" stroke="{GRID}" stroke-width="0.8" rx="5"/>')
    items = [
        (ORANGE_BG, ORANGE, '本文改动（Triplet）'),
        (GREEN_BG, GREEN, 'Neck 特征融合'),
        (PALE, GRID, '特征图 / Concat'),
        (WHITE, INK, '卷积 / 基础模块'),
    ]
    lx = 60
    for fill, stroke, label in items:
        s.rect(lx, 1146, 18, 14, fill=fill, stroke=stroke, sw=1.2, rx=3)
        s.text(lx + 26, 1153, label, size=10, color=INK, anchor='start')
        lx += 230
    s.text(W - 60, 1153, '输入 896×896；P3/8=112²，P4/16=56²，P5/32=28²', size=10,
           color=GRAY, anchor='end')

    return s.render()


def edge_candidates():
    """Edge/Chrome 可执行文件候选。"""
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
    html = os.path.join(OUT, 'yolo11_triplet_arch.html')
    with open(html, 'w', encoding='utf-8') as f:
        f.write('<html><head><meta charset="utf-8">'
                '<style>@page{size:%dpx %dpx;margin:0}'
                'html,body{margin:0;padding:0;width:%dpx;height:%dpx;overflow:hidden}'
                'svg{display:block}</style></head><body>' % (W, H, W, H))
        f.write(open(svg_path, encoding='utf-8').read())
        f.write('</body></html>')
    png = os.path.join(OUT, 'yolo11_triplet_arch.png')
    pdf = os.path.join(OUT, 'yolo11_triplet_arch.pdf')
    url = 'file:///' + html.replace('\\', '/')
    cmds = [
        [exe, '--headless=new', '--disable-gpu', '--hide-scrollbars',
         f'--window-size={W},{H}', f'--screenshot={png}', url],
        [exe, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
         f'--print-to-pdf={pdf}', '--no-margins', url],
    ]
    for cmd in cmds:
        try:
            subprocess.run(cmd, check=True, timeout=120,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print('rendered', cmd[4])
        except Exception as e:
            print('render failed', e)


def write_drawio():
    """生成一个可编辑的 draw.io 初始文件：主流程 + 3 个模块细节。

    不追求与 SVG 逐像素一致，只保证打开后每个模块都是可拖动、可改字的图形。
    """
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
            f'<mxCell id="{i}" value="{sx.escape(label)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def edge(a, b, label='', dash=False, color=INK):
        i = nid()
        style = (f'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;'
                 f'strokeColor={color};strokeWidth=1.5;'
                 + ('dashed=1;' if dash else ''))
        cells.append(
            f'<mxCell id="{i}" value="{sx.escape(label)}" style="{style}" edge="1" parent="1" '
            f'source="{a}" target="{b}"><mxGeometry relative="1" as="geometry"/></mxCell>')

    # 主标题
    box(30, 10, 1740, 60, 'YOLOv11s + TripletAttention 网络结构示意图（896×896，nc=3）',
        fill='none', stroke='none', size=22, bold=True)
    # Input / Backbone
    inp = box(30, 150, 110, 100, '输入图像\n896×896×3', fill=PALE, size=13)
    bb = box(170, 100, 1200, 280, '', fill='#FBFCFE', stroke=BLUE)
    bb_label = box(180, 105, 220, 30, 'Backbone（YOLOv11s）', fill='none', stroke='none',
                   size=14, bold=True)

    blocks = [
        ('0  Conv\n32, 3×3, s2\n448²', WHITE),
        ('1  Conv\n64, 3×3, s2\n224²', WHITE),
        ('2  C3k2×2\n128\n224²', WHITE),
        ('3  Conv\n128, 3×3, s2\n112²', WHITE),
        ('4  C3k2×2\n256\n112²', WHITE),
        ('5  Conv\n256, 3×3, s2\n56²', WHITE),
        ('6  C3k2×2\n256\n56²', WHITE),
        ('7  Conv\n512, 3×3, s2\n28²', WHITE),
        ('8  C3k2×2\n512\n28²', WHITE),
        ('9  SPPF\n512\n28²', WHITE),
        ('插入  Triplet\nAttention\n28²', ORANGE_BG),
        ('10  C2PSA\n512\n28²', WHITE),
    ]
    xs = 190
    prev = inp
    ids = []
    for i, (label, fill) in enumerate(blocks):
        b = box(xs + i * 96, 175, 84, 100, label, fill=fill,
                stroke=ORANGE if fill == ORANGE_BG else INK,
                size=10.5, bold=fill == ORANGE_BG)
        edge(prev, b)
        ids.append(b)
        prev = b

    # Neck
    nb = box(170, 430, 1200, 300, 'Neck（FPN + PAN）', fill='#FBFCFE', stroke=GREEN, size=14, bold=True)
    neck = {}
    specs = [
        ('P3', 250, 500, 'P3 特征\n112×112×128\nConcat[P4↑,P3] → C3k2 #16'),
        ('P4', 650, 500, 'P4 特征\n56×56×256\nConcat[P5↑,P4] → C3k2 #13\nConcat[P3↓,P4] → C3k2 #19'),
        ('P5', 1050, 500, 'P5 特征\n28×28×512\nConcat[P4↓,P5] → C3k2 #22'),
    ]
    for name, x, y, label in specs:
        neck[name] = box(x, y, 240, 120, label, fill=GREEN_BG, stroke=GREEN, size=10.5)
    edge(neck['P5'], neck['P4'], 'Upsample ×2', color=BLUE)
    edge(neck['P4'], neck['P3'], 'Upsample ×2', color=BLUE)
    edge(neck['P3'], neck['P4'], 'Conv s2', color=GREEN)
    edge(neck['P4'], neck['P5'], 'Conv s2', color=GREEN)
    edge(ids[11], neck['P5'], 'P5')
    edge(ids[6], neck['P4'], 'P4', dash=True)
    edge(ids[4], neck['P3'], 'P3', dash=True)

    # Head
    hb = box(1400, 430, 360, 300, 'Head（Detect）', fill='#FBFCFE', stroke=ORANGE, size=14, bold=True)
    h1 = box(1430, 500, 300, 50, 'Detect P3\n112×112×3×(4+1+3)', fill=ORANGE_BG, stroke=ORANGE, size=10.5)
    h2 = box(1430, 570, 300, 50, 'Detect P4\n56×56×3×(4+1+3)', fill=ORANGE_BG, stroke=ORANGE, size=10.5)
    h3 = box(1430, 640, 300, 50, 'Detect P5\n28×28×3×(4+1+3)', fill=ORANGE_BG, stroke=ORANGE, size=10.5)
    edge(neck['P3'], h1, color=ORANGE)
    edge(neck['P4'], h2, color=ORANGE)
    edge(neck['P5'], h3, color=ORANGE)

    # 模块细节
    detail = [
        (30, 790, 240, 330, 'Conv = Conv2d + BN + SiLU'),
        (290, 790, 320, 330, 'C3k2 = 1×1 Conv + k2 Bottleneck ×n + Concat + 1×1 Conv'),
        (630, 790, 240, 330, 'SPPF = 1×1 Conv + 串行 5×5 MaxPool ×3 + Concat + 1×1 Conv'),
        (890, 790, 280, 330, 'C2PSA = 1×1 Conv + PSA 多头自注意力 + Concat + 1×1 Conv'),
        (1190, 790, 400, 330, 'TripletAttention（本文）\n三分支 C×W / H×C / H×W\n每支：ZPool → 7×7 Conv → BN → Sigmoid → ⊗\n输出 = 1/3 Σ 三分支\n插入：SPPF → Triplet → C2PSA'),
        (1610, 790, 170, 330, 'Detect\nanchor-free\nbox(4) + cls(3)\nP3/P4/P5'),
    ]
    for x, y, w, h, label in detail:
        fill = ORANGE_BG if 'Triplet' in label else '#FAFBFD'
        stroke = ORANGE if 'Triplet' in label else GRID
        box(x, y, w, h, label, fill=fill, stroke=stroke, size=11,
            bold='Triplet' in label)

    xml = ('<mxfile host="app.diagrams.net" modified="2026-10-19T00:00:00.000Z" agent="Claude" version="24.0.0">'
           '<diagram id="yolo11-triplet" name="YOLOv11s+Triplet">'
           '<mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" '
           'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1800" pageHeight="1180" '
           'math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
           + ''.join(cells) +
           '</root></mxGraphModel></diagram></mxfile>')
    path = os.path.join(OUT, 'yolo11_triplet_arch.drawio')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(xml)
    print('wrote', path)


def main():
    os.makedirs(OUT, exist_ok=True)
    svg = build()
    svg_path = os.path.join(OUT, 'yolo11_triplet_arch.svg')
    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(svg)
    print('wrote', svg_path)
    write_drawio()
    if '--svg-only' not in sys.argv:
        render_external(svg_path)


if __name__ == '__main__':
    main()
