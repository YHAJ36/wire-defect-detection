# -*- coding: utf-8 -*-
"""
Render report/model_data_draft_v0.md to a landscape A4 PDF.

Usage:
    python report/make_model_pdf.py
    REPORT_PNG=<dir> python report/make_model_pdf.py   # also dump page PNGs

Output:
    report/模型改进与数据对照_v1.pdf
"""
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, LongTable, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(BASE, 'report')
MD = os.path.join(REP, 'model_data_draft_v0.md')
OUT = os.path.join(REP, '模型改进与数据对照_v1.pdf')

pdfmetrics.registerFont(TTFont('SimHei', r'C:\Windows\Fonts\simhei.ttf'))
pdfmetrics.registerFont(TTFont('SimKai', r'C:\Windows\Fonts\simkai.ttf'))

PAGE = landscape(A4)
PW, PH = PAGE
LM = RM = 12 * mm
TM, BM = 11 * mm, 10 * mm
FRAME_W = PW - LM - RM

BLUE = colors.HexColor('#1F3864')
LIGHT = colors.HexColor('#D9E2F3')
GRID = colors.HexColor('#A6A6A6')
GRAY = colors.HexColor('#555555')

S_TITLE = ParagraphStyle('title', fontName='SimHei', fontSize=26, leading=34,
                         alignment=1, textColor=BLUE, spaceAfter=10)
S_SUB = ParagraphStyle('sub', fontName='SimHei', fontSize=11, leading=17,
                       alignment=1, textColor=GRAY)
S_H1 = ParagraphStyle('h1', fontName='SimHei', fontSize=16, leading=22,
                      textColor=BLUE, spaceBefore=2, spaceAfter=8)
S_H2 = ParagraphStyle('h2', fontName='SimHei', fontSize=12.5, leading=17,
                      textColor=colors.HexColor('#2E5395'), spaceBefore=9, spaceAfter=5)
S_H3 = ParagraphStyle('h3', fontName='SimHei', fontSize=10.5, leading=14,
                      textColor=colors.HexColor('#333333'), spaceBefore=7, spaceAfter=4)
S_H4 = ParagraphStyle('h4', fontName='SimHei', fontSize=9.5, leading=13,
                      textColor=colors.HexColor('#444444'), spaceBefore=6, spaceAfter=3)
S_BODY = ParagraphStyle('body', fontName='SimHei', fontSize=9, leading=13,
                        wordWrap='CJK', spaceAfter=3)
S_NOTE = ParagraphStyle('note', fontName='SimKai', fontSize=8.5, leading=11.5,
                        wordWrap='CJK', textColor=colors.HexColor('#444444'),
                        backColor=colors.HexColor('#FFF9E6'), borderPadding=4,
                        leftIndent=6, rightIndent=6, spaceBefore=2, spaceAfter=5)
S_CAP = ParagraphStyle('cap', fontName='SimHei', fontSize=9, leading=12,
                       textColor=colors.HexColor('#2E5395'), spaceBefore=7, spaceAfter=3,
                       keepWithNext=1)
S_CELL = ParagraphStyle('cell', fontName='SimHei', fontSize=6.6, leading=8.2,
                        wordWrap='CJK', splitLongWords=1)
S_CELL_S = ParagraphStyle('cells', fontName='SimHei', fontSize=6.4, leading=8.0,
                          wordWrap='CJK', splitLongWords=1)
S_HEAD = ParagraphStyle('head', fontName='SimHei', fontSize=6.8, leading=8.4,
                        wordWrap='CJK', textColor=BLUE)
S_HEAD_S = ParagraphStyle('heads', fontName='SimHei', fontSize=6.6, leading=8.2,
                          wordWrap='CJK', textColor=BLUE)
S_SRC = ParagraphStyle('src', parent=S_NOTE, keepWithNext=1)


def _split_long_url(u):
    """Break a long URL at a slash so it does not end with a 1-char orphan line."""
    if len(u) <= 52:
        return u
    pos = u.rfind('/', 0, 52)
    if pos < 10:
        pos = 52
    return u[:pos + 1] + '\x00BR\x00' + u[pos + 1:]


def inline(t):
    """Convert a small Markdown subset into ReportLab markup."""
    # SimHei / SimKai lack these two glyphs; map to visual equivalents.
    t = t.replace('−', '-').replace('⇒', '→')
    t = re.sub(r'https?://[A-Za-z0-9\-._~/]+', lambda m: _split_long_url(m.group(0)), t)
    t = t.replace('<br>', '\x00BR\x00')
    t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    t = t.replace('\x00BR\x00', '<br/>')
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<link href="\2" color="#1F4E79">\1</link>', t)
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'`([^`]+)`', r'<font name="SimKai">\1</font>', t)
    return t


def strip_md(t):
    return re.sub(r'[*`]', '', t).replace('<br>', ' ')


def text_len(t):
    t = re.sub(r'<[^>]+>', '', t)
    n = 0
    for ch in t:
        n += 2 if ord(ch) > 127 else 1
    return n


def col_widths(header, rows, avail):
    n = len(header)
    weights = []
    for i in range(n):
        vals = [header[i]] + [r[i] if i < len(r) else '' for r in rows[:80]]
        ml = max(text_len(strip_md(v)) for v in vals)
        weights.append(min(max(ml, 4), 26))
    total = sum(weights) or 1
    widths = [max(24.0, avail * w / total) for w in weights]
    scale = avail / sum(widths)
    return [w * scale for w in widths]


def make_table(header, rows):
    ncol = len(header)
    small = ncol >= 9
    cs = S_CELL_S if small else S_CELL
    hs = S_HEAD_S if small else S_HEAD
    data = [[Paragraph(inline(str(c)), hs) for c in header]]
    for r in rows:
        r = list(r) + [''] * (ncol - len(r))
        data.append([Paragraph(inline(str(c)), cs) for c in r[:ncol]])
    widths = col_widths(header, rows, FRAME_W)
    t = LongTable(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), LIGHT),
        ('GRID', (0, 0), (-1, -1), 0.3, GRID),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 2.2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2.2),
        ('TOPPADDING', (0, 0), (-1, -1), 1.6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7F9FC')]),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('SimHei', 7)
    canvas.setFillColor(GRAY)
    canvas.drawString(LM, 5.5 * mm, '模型改进与数据对照 v1 · 2026-10-17')
    canvas.drawRightString(PW - RM, 5.5 * mm, '第 %d 页' % canvas.getPageNumber())
    canvas.restoreState()


def cover(story):
    story.append(Spacer(1, 62))
    story.append(Paragraph('模型改进与数据对照', S_TITLE))
    story.append(Paragraph('输电线缺陷检测项目 · 本项目实验 + P1–P6 文献 + 网络来源', S_SUB))
    story.append(Spacer(1, 18))
    story.append(Paragraph('版本 v1　|　2026-10-17　|　数据截至 2026-10-17', S_SUB))
    story.append(Spacer(1, 40))
    box = Table([[Paragraph(
        '本册用于与导师讨论和后续实验选型。所有数字均标注来源；'
        '跨论文的数据集、类别数和指标定义不同，P/R/mAP 只能做方向性对照，不能直接排名。'
        '本项目 896 基线统一引用 n=3 均值 0.9079 [0.9044, 0.9122]；0.9148 仅作封版归档值。',
        S_BODY)]], colWidths=[FRAME_W * 0.78])
    box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EEF3FA')),
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#B4C6E7')),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(box)
    story.append(PageBreak())


def summary(story):
    story.append(Paragraph('先看这一页：数字口径', S_H1))
    rows = [
        ['本项目 v_final 320 / 640', '单种子、best.pt、test；只看方向，不做显著性结论'],
        ['本项目 896 n=3', '可用于结论；基线 0.9079 [0.9044, 0.9122]；0.9148 仅归档'],
        ['文献 P1–P6', '数据集 / 类别 / 指标不同，不可跨论文排名'],
        ['网络来源', '未核实，只能当候选；PDF 中已核验链接并标注名实不符项'],
    ]
    t = Table([[Paragraph(a, S_BODY), Paragraph(b, S_BODY)] for a, b in rows],
              colWidths=[150, FRAME_W - 150], hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#EEF3FA')),
        ('GRID', (0, 0), (-1, -1), 0.3, GRID),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))
    story.append(Paragraph('为什么目标是 95、难点在哪里', S_H2))
    story.append(Paragraph(
        '本项目数据集为 3 类（break / thunderbolt / wear），train / val / test = 1428 / 178 / 179 张，'
        'test 共 283 个实例。v11 在 896 下的逐类 mAP50：break 0.870、thunderbolt 0.962、wear 0.912，'
        '总体 0.915（n=3 均值 0.9079）。mAP50 是三类平均，所以要把总体从 0.908 提到 0.950，'
        '在 thunderbolt 保持 0.962、wear 提到 0.94 的前提下，break 必须到 0.948（+7.8pt）；'
        '若 wear 不动，break 必须到 0.976。结论：后续实验的主战场是 break（小目标、框最小），'
        '不是再叠加一个通用注意力模块。', S_BODY))
    story.append(Spacer(1, 6))
    kt = Table([
        [Paragraph('<b>类别</b>', S_BODY), Paragraph('<b>896 test mAP50</b>', S_BODY),
         Paragraph('<b>到总体 95 的要求</b>', S_BODY)],
        [Paragraph('break', S_BODY), Paragraph('0.870', S_BODY), Paragraph('约 0.94–0.97（+7～10pt）', S_BODY)],
        [Paragraph('thunderbolt', S_BODY), Paragraph('0.962', S_BODY), Paragraph('保持或略升', S_BODY)],
        [Paragraph('wear', S_BODY), Paragraph('0.912', S_BODY), Paragraph('约 0.93–0.94', S_BODY)],
    ], colWidths=[110, 130, FRAME_W - 240], hAlign='LEFT')
    kt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), LIGHT),
        ('GRID', (0, 0), (-1, -1), 0.3, GRID),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(kt)
    story.append(Spacer(1, 6))
    story.append(Paragraph(inline(
        '来源：`logs/eval_896.log`（v11 896 test，179 张 / 283 实例）；`logs/eval_test_640.log`；'
        '`report/_v5_constants.json`；`exp_upscale/logs/batch2_eval.tsv`、`batch3_eval.tsv`。'), S_NOTE))
    story.append(Spacer(1, 8))
    story.append(Paragraph('本册怎么读', S_H2))
    for b in ['§0 阅读约定：数字口径与引用规则。',
              '§1 按模型族 → 改进类别：1.1 YOLOv5 … 1.8 非 YOLO；每族先列本项目实测，再按类别列文献方案。',
              '§2 来源索引：文献优先；网络来源同列，按“有论文题名 → 仅开源 → 仅检索名称”排序。',
              '§3 可比性与引用警告；§4 附录：per-class、batch1、逐种子明细。']:
        story.append(Paragraph(inline(b), S_BODY, bulletText='•'))
    story.append(PageBreak())


def reverse_index(story):
    story.append(Paragraph('反向索引：改进类别 → 谁做过', S_H1))
    story.append(Paragraph(
        '第 1 章按“模型族 → 改进类别”编排；本页反过来，便于查“某个改进谁做过、做到什么程度”。'
        '“本项目”只列已完成的实验；“文献”指 References/ 下 P1–P6；“网络”指未核实的候选方向。', S_BODY))
    rows = [
        ['基线 / 输入分辨率', '本项目 YOLOv5s / v11s / v12s；P1–P6 各自基线', '本项目 320/640/896；论文多为 640'],
        ['注意力', '本项目 CBAM(backbone/neck/both)、CBAM+BN、Triplet；P1 CA；P2 CA/GAM；P4 CBAM；P5 CA；P6 SimAM', '本项目 n=3 无可检测增益；文献均有名义提升'],
        ['特征融合 / 颈部', 'P2 RFB/SPPCSPC；P5 Slim-neck、TFE+Dy-SSFF；P6 CGAFusion、CSFCN、AFPN', '网络：RepGFPN、LMAFPN、PFPN、Dy-CCFM'],
        ['损失函数', 'P1 SIOU；P2 WIoU v3、Focal-EIoU；P5 WIoU', '网络：Focaler-IoU、MPDIoU'],
        ['轻量化卷积 / 剪枝', 'P1 Ghost；P5 Slim-neck、通道剪枝；P6 ADown、DySnakeConv、EMSDC、C3k2_LFE', '网络：AKConv/LDConv、iRMB、Mamba'],
        ['上采样 / 检测头', 'P5 DySample、DyHead；P6 Detect_AFPN_P345', '本项目尚未做 P2 小目标头'],
        ['数据增强 / 训练策略', 'P4 AdamW/cos_lr/warmup/erasing；P5 气象增强', '网络：GAN、Mask-Guided Copy-Paste'],
        ['多模态 / 分割', 'P3 K-Net + 形状/几何 + BERT + LMF', '与本项目检测任务不同'],
        ['知识蒸馏 / NAS', 'References 中无', '网络：FIAEPI-KD、OFN+蒸馏、EHANN-NAS、AutoNAD'],
    ]
    t = Table([[Paragraph(h, S_HEAD) for h in ['改进类别', '本项目 / 文献', '网络来源（候选）']]] +
              [[Paragraph(a, S_CELL), Paragraph(b, S_CELL), Paragraph(c, S_CELL)] for a, b, c in rows],
              colWidths=[105, FRAME_W * 0.52, FRAME_W - 105 - FRAME_W * 0.52], hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), LIGHT),
        ('GRID', (0, 0), (-1, -1), 0.3, GRID),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7F9FC')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))


def parse_markdown(story):
    lines = open(MD, encoding='utf-8').read().splitlines()
    i = 0
    # Drop everything before the first section heading (old title + stale note).
    while i < len(lines) and not lines[i].strip().startswith('## '):
        i += 1
    first_h1 = True
    started = False
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s.startswith('# '):
            i += 1
            continue
        if s.startswith('## 5.'):
            break
        if s.startswith('## '):
            if started:
                story.append(PageBreak())
            started = True
            story.append(Paragraph(inline(s[3:]), S_H1))
            i += 1
            continue
        if s.startswith('### '):
            story.append(Paragraph(inline(s[4:]), S_H2))
            i += 1
            continue
        if s.startswith('#### '):
            story.append(Paragraph(inline(s[5:]), S_H3))
            i += 1
            continue
        if s.startswith('##### '):
            story.append(Paragraph(inline(s[6:]), S_H4))
            i += 1
            continue
        if s.startswith('|'):
            block = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                block.append(lines[i].strip())
                i += 1
            if len(block) >= 2:
                def cells(row):
                    return [c.strip() for c in row.strip().strip('|').split('|')]
                header = cells(block[0])
                body = [cells(r) for r in block[2:]]
                story.append(make_table(header, body))
                story.append(Spacer(1, 5))
            continue
        if s.startswith('> '):
            buf = []
            while i < len(lines) and lines[i].strip().startswith('> '):
                buf.append(lines[i].strip()[2:])
                i += 1
            story.append(Paragraph(inline(' '.join(buf)), S_NOTE))
            continue
        if s.startswith('- '):
            story.append(Paragraph(inline(s[2:]), S_BODY, bulletText='•'))
            i += 1
            continue
        if re.match(r'^\d+\.\s', s):
            story.append(Paragraph(inline(s), S_BODY, bulletText=''))
            i += 1
            continue
        # Table caption: absorb a following 出处 line into the same flowable so
        # the caption cannot be orphaned at the bottom of a page.
        if s.startswith('**') and s.endswith('**') and s.count('**') == 2:
            cap = s
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            src = None
            if j < len(lines):
                cand = lines[j].strip()
                if cand.startswith('*出处') and cand.endswith('*'):
                    src = cand.strip('*')
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1
            html = inline(cap)
            if src:
                html += ('<br/><font name="SimKai" size="7.6" color="#555555">'
                         + inline(src) + '</font>')
            story.append(Paragraph(html, S_CAP))
            continue
        # paragraph: join until blank / heading / table
        buf = [s]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith('#') or nxt.startswith('|')
                    or nxt.startswith('> ') or nxt.startswith('- ')
                    or re.match(r'^\d+\.\s', nxt)):
                break
            buf.append(nxt)
            i += 1
        text = ' '.join(buf)
        if text.startswith('*出处') or (text.startswith('*') and text.endswith('*') and len(text) > 2):
            is_src = text.startswith('*出处')
            text = text.strip('*')
            story.append(Paragraph(inline(text), S_SRC if is_src else S_NOTE))
        elif text.startswith('**') and text.endswith('**') and text.count('**') == 2:
            story.append(Paragraph(inline(text), S_CAP))
        else:
            story.append(Paragraph(inline(text), S_BODY))


def build():
    doc = BaseDocTemplate(OUT, pagesize=PAGE,
                          leftMargin=LM, rightMargin=RM, topMargin=TM, bottomMargin=BM,
                          title='模型改进与数据对照 v1', author='D:/2 输电线缺陷检测项目')
    frame = Frame(LM, BM, FRAME_W, PH - TM - BM, id='normal',
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id='main', frames=[frame], onPage=footer)])

    story = []
    cover(story)
    summary(story)
    reverse_index(story)
    parse_markdown(story)
    doc.build(story)

    print('OK', OUT)
    png_dir = os.environ.get('REPORT_PNG')
    if png_dir:
        import pymupdf
        os.makedirs(png_dir, exist_ok=True)
        d = pymupdf.open(OUT)
        for k in range(d.page_count):
            pix = d[k].get_pixmap(dpi=110)
            pix.save(os.path.join(png_dir, 'page_%02d.png' % (k + 1)))
        print('PNG', png_dir, d.page_count)


if __name__ == '__main__':
    build()
