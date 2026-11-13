# -*- coding: utf-8 -*-
"""生成 10-15 页汇报版 docx，随后由 Word 导出 PDF"""
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

FIG = 'D:/2/report/_summary_figs'
OUT = 'D:/2/report/汇报版_输电线缺陷检测_总结.docx'
DARK = RGBColor(0x1a, 0x1a, 0x1a)
GRAY = RGBColor(0x55, 0x55, 0x55)
ACC = RGBColor(0x1f, 0x3c, 0x73)

doc = Document()

# ---------- 页面 ----------
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.2)
sec.bottom_margin = Cm(2.0)
sec.left_margin = Cm(2.1)
sec.right_margin = Cm(2.1)

# 正文默认
st = doc.styles['Normal']
st.font.name = 'Times New Roman'
st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
st.paragraph_format.line_spacing = 1.45
st.paragraph_format.space_after = Pt(4)


def cn(run, name='宋体', size=10.5, bold=False, color=DARK):
    run.font.name = 'Times New Roman'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return run


def para(text='', size=10.5, bold=False, name='宋体', color=DARK,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True, before=0, after=4,
         spacing=1.45, newpage=False):
    p = doc.add_paragraph()
    if newpage:
        p.paragraph_format.page_break_before = True
    p.paragraph_format.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = spacing
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    if text:
        cn(p.add_run(text), name, size, bold, color)
    return p


def rich(parts, size=10.5, indent=True, before=0, after=4, spacing=1.45,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    """parts = [(text, bold, color), ...]"""
    p = doc.add_paragraph()
    p.paragraph_format.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = spacing
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    for (t, b, c) in parts:
        cn(p.add_run(t), '宋体', size, b, c if c else DARK)
    return p


def h1(text, before=0, after=10, newpage=False):
    p = doc.add_paragraph()
    if newpage:
        p.paragraph_format.page_break_before = True
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.2
    cn(p.add_run(text), '黑体', 15.5, True, ACC)
    # 下方细线
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '4')
    bottom.set(qn('w:color'), '1F3C73')
    pbdr.append(bottom)
    pPr.append(pbdr)
    return p


def h2(text, before=10, after=5, newpage=False):
    p = doc.add_paragraph()
    if newpage:
        p.paragraph_format.page_break_before = True
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.2
    cn(p.add_run(text), '黑体', 12, True, DARK)
    return p


def bullet(head, body, size=10.5):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Pt(14)
    p.paragraph_format.first_line_indent = Pt(-14)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.45
    cn(p.add_run('▪ '), '宋体', size, False, ACC)
    if head:
        cn(p.add_run(head), '黑体', size, True, DARK)
    if body:
        cn(p.add_run(body), '宋体', size, False, DARK)
    return p


def fig(path, width_cm, caption, note=None):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(path, width=Cm(width_cm))
    c = doc.add_paragraph()
    c.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(3 if note else 8)
    c.paragraph_format.line_spacing = 1.15
    cn(c.add_run(caption), '宋体', 9, False, GRAY)
    if note:
        n = doc.add_paragraph()
        n.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        n.paragraph_format.space_after = Pt(8)
        n.paragraph_format.line_spacing = 1.3
        cn(n.add_run(note), '宋体', 9, False, GRAY)


def table(rows, widths, header=True, size=9, first_col_center=True):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.width = Cm(widths[j])
            pp = cell.paragraphs[0]
            pp.paragraph_format.space_before = Pt(2)
            pp.paragraph_format.space_after = Pt(2)
            pp.paragraph_format.line_spacing = 1.1
            if j > 0 or not first_col_center:
                pp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                pp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cn(pp.add_run(str(val)), '宋体', size,
               bold=(i == 0 and header), color=DARK)
            if i == 0 and header:
                tcPr = cell._tc.get_or_add_tcPr()
                shd = OxmlElement('w:shd')
                shd.set(qn('w:val'), 'clear')
                shd.set(qn('w:fill'), 'E8EDF5')
                tcPr.append(shd)
    # 列宽写死
    for j, w in enumerate(widths):
        for cell in t.columns[j].cells:
            cell.width = Cm(w)
    return t


# ============================================================
# 封面
# ============================================================
for _ in range(5):
    doc.add_paragraph()

p = doc.add_paragraph(); p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
cn(p.add_run('基于 YOLOv11 的输电线路缺陷检测'), '黑体', 24, True, ACC)

p = doc.add_paragraph(); p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(10)
cn(p.add_run('——13 个改进方向的实验评估与瓶颈分析'), '黑体', 15, False, GRAY)

p = doc.add_paragraph(); p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(28)
cn(p.add_run('汇报版'), '宋体', 11, False, GRAY)

for _ in range(6):
    doc.add_paragraph()

for line in ['汇报人：项目组', '汇报时间：2026 年 10 月']:
    p = doc.add_paragraph(); p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    cn(p.add_run(line), '宋体', 12, False, DARK)

# ============================================================
# 第一部分 核心结论
# ============================================================
h1('一、核心结论', newpage=True)

para('本项目的目标不是"造一个更强的检测器"，而是在给定数据上做一次可信的评估：'
     '把当前主流的小目标改进方向逐一实测一遍，看哪些真正有效、哪些只是噪声。'
     '结论可以用四句话概括。', indent=True)

h2('1. 13 个改进方向，无一能显著超越基线', before=8)
para('覆盖结构（CBAM、TripletAttention、P2 检测头、YOLO26 代际）、损失（WIoU v3）、'
     '数据与训练（mosaic 削弱、close_mosaic、磨损过采样、1024 分辨率、150 轮预算）、'
     '集成（三种子 WBF 融合、类级模型选择）四类。13 个方向全部跑完，'
     '在三种子口径下排名第一的配置仍然是"什么都不改"的基线。')

h2('2. 测量精度不足以分辨这么小的效应')
rich([('三种子噪声地板为 ', False, None), ('1.60 pt', True, ACC),
      ('（357 图 val+test 合并集，同配置换随机种子的跨度），而绝大多数改进效应落在 ', False, None),
      ('0.2~0.9 pt', True, ACC),
      ('。也就是说，尺子的最小刻度比被测物大 3 倍以上——在这个量级上，'
       '任何"提升 0.3 分"的说法都无法与随机波动区分。', False, None)])

h2('3. 真正的瓶颈在数据侧，不在模型')
rich([('标注自身的框尺寸抖动达 ', False, None), ('13.8%', True, ACC),
      ('（相邻帧同一目标的相对宽差标准差），模型预测误差为 ', False, None),
      ('14.65%', True, ACC),
      ('。模型只比人工标注的噪声高 6%——它已经贴着标注上限，'
       '继续加模块只是在这个噪声里捞信号。', False, None)])

h2('4. 最终基线指标（v3 口径，2026-10-27 重定）')
rich([('按视觉段整体重划分并从头重训后，跨段泛化不是一个点，而是 ', False, None),
      ('74.7% ~ 89.2%', True, ACC),
      (' 的区间：三种子在 203 图 test 上 mAP@0.5 = 74.67%（mAP@0.5:0.95 = 37.71%，'
       'P = 79.65%、R = 68.29%），而同一批训练产出的同一颗 s42 权重在 val 181 图上为 89.15% ——'
       '只换"留出哪批视觉段"就差 13.2 pt，而三种子之间的跨度只有 2.74 pt。', False, None)])
rich([('旧口径 92.12%（179 图，含 44.1% 同段泄漏）与 89.98%（帧级隔离 100 图）均已作废，仅作对照。'
       '推理端三种子 WBF 在同一张卷内把 74.67% 提到 77.76%（+3.09 pt），'
       '是唯一越过 1.60 pt 噪声地板的增益；TTA 三颗种子全负，弃用。', False, None)])

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(14)
p.paragraph_format.space_after = Pt(4)
p.paragraph_format.left_indent = Pt(6)
p.paragraph_format.line_spacing = 1.4
cn(p.add_run('一句话：'), '黑体', 10.5, True, ACC)
cn(p.add_run('这不是"试了很多方法都没用"，而是"证明了这条路的上限在哪里，以及上限由什么决定"。'
             '前者的结论是"再试试别的"，后者的结论是"该换赛道了"。'), '宋体', 10.5, False, DARK)

h2('评估口径（全文统一，不可混用）', before=18)
para('本汇报所有数字均按以下口径计算，跨表比较前请先核对口径：', indent=True, after=5)
bullet('v3 主口径（第 4 点、第五部分）：', '按 1080 个视觉段整体重划分（跨段 0）后从头重训的 6 个 run，'
                              'test 203 图 / val 181 图；896 输入、NMS IoU 0.5、conf 0.001；三种子均值。'
                              '同一颗 s42 权重在两张段级卷上分别 75.91% / 89.15%，这是区间口径的来源。')
bullet('历史主表口径（13 个方向的 Δ）：', '179 图 test 集，NMS IoU 0.5，conf 0.001，取 last.pt（终态权重）。'
                                '该 test 含 44.1% 与训练集同段的图，绝对值偏乐观；'
                                '但 13 个方向是在同一张卷上互比，方向性结论不受影响。')
bullet('合并集口径：', '357 图（val + test）仅在终评阶段使用，且只用于 close_mosaic 与 mosaic 两臂的复核，'
                    '不出现在主表。')
bullet('召回率：', '主口径为 85.17%（三种子均值，conf 0.001）。另有两个不同口径不得与主口径混写：'
                '单颗 s42 在 conf 0.001 下为 83.63%，同一权重在 conf 0.25 的部署阈值下为 82.78%。')
bullet('权重选择：', '统一用 last.pt。best.pt 口径的数值只用于说明"多轮次训练的红利来自挑检查点"，'
                  '不作为结论依据。')

# ============================================================
# 第二部分
# ============================================================
h1('二、13 个方向的探索与失败')   # 评估口径溢出到本页，不再强制分页以免留白

para('先明确任务本身：三类缺陷——断股（钢绞线散开）、雷击损伤（表面灼伤）、磨损。'
     '目标普遍很小，断股框的中位边长只有约 40 px，这是所有小目标改进方法被引入的动机。',
     indent=True)

fig(f'{FIG}/fig1_samples_gt.jpg', 16.2,
    '图 1  三类缺陷样例与人工标注框（左：原图与标框；右：框内区域放大）',
    '断股是散开的钢丝，标注边界没有客观答案；雷击是表面灼伤，边界相对明确；'
    '磨损是成片区域，一张图常含多个框。')

para('数据规模：1785 张图像、3010 个标注框（断股 1295、雷击 799、磨损 916），'
     '按 8:1:1 随机划分为 1428 / 178 / 179 张（v1 口径）。'
     'v3 口径下改按 1080 个视觉段整体重划为 1401 / 181 / 203 张，跨段泄漏为 0。')
para('三类缺陷的检测难度差异很大，而且原因不同：雷击损伤的 AP50 达到 0.980，'
     '因为灼伤区域边界明确、外观一致；磨损是 0.932，它常成片出现、一张图里有多个框，'
     '小框的定位误差会被平均掉；断股最低，只有 0.847。'
     '值得注意的是，断股并不是"看不清"——第三部分会证明，它是"边界本身没有共识"。')


para('13 个方向的结果汇总如下。Δ 值统一为相对基线在 179 图 test 上的 mAP@0.5 变化（单位：百分点）；'
     '三种子的方向取三颗均值与基线的差，单种子的方向取该种子与同种子基线的差。',
     indent=True, after=6, newpage=True)

rows = [['#', '改进方向', 'ΔmAP50', 'ΔmAP50-95', '判定'],
        ['1', 'CBAM + BN 注意力', '−0.68', '−1.44', '无效'],
        ['2', 'TripletAttention', '−0.29', '−1.00', '无效'],
        ['3', 'P2 小目标头 v1', '−0.54', '−3.35', '无效（初始化缺陷）'],
        ['4', 'P2 小目标头 v2', '−0.69', '−1.04', '无效'],
        ['5', 'WIoU v3 损失', '+0.34', '−0.89', '不可判'],
        ['6', '磨损 2× 过采样', '+0.90', '—', '不可判'],
        ['7', 'YOLO26s 换代', '−0.19', '−1.75', '打平'],
        ['8', '1024 分辨率', '+0.16', '—', '不可判'],
        ['9', '训练预算 60→150 轮', '−0.09', '+0.54', '无收益'],
        ['10', '三种子 WBF 融合', '+0.11', '+2.13', '不可判'],
        ['11', '类级模型选择', '+0.24', '—', '不可判'],
        ['12', 'mosaic 0.8→0.4', '−2.92', '−2.60', '负效应'],
        ['13', 'close_mosaic 5→20', '−1.10', '+0.42', '不可判']]
table(rows, [1.1, 5.4, 2.2, 2.6, 4.4], size=8.8)

p = doc.add_paragraph()
p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(3)
p.paragraph_format.space_after = Pt(8)
cn(p.add_run('表 1  13 个改进方向的实测效应量（179 图 test，NMS 0.5，conf 0.001）'), '宋体', 9, False, GRAY)

fig(f'{FIG}/fig5_effect_sizes.png', 15.2,
    '图 2  13 个方向的效应量（横轴：相对基线的 mAP@0.5 变化，单位百分点）')

h2('三个典型的负结果', before=0, newpage=True)

para('把 13 个方向按"为什么失败"归并，可以归成三类。每一类都改变了我们对模型的理解。',
     indent=True, after=8)

h2('① 加注意力：真的插进去了，但等于没加', before=6)
para('CBAM 注入 neck 层后，参数量增加 84,458，与理论值逐位吻合——这证明模块确实进了网络'
     '（第一次尝试时注入被框架内部重建静默丢弃，参数量与基线完全一致，这个坑在第四节细说）。')
para('但效果为零，甚至是负的（−0.68 pt）。机理是：空间注意力的卷积输出存在一个较大的正向 DC 偏置，'
     '经 Sigmoid 后饱和，输出几乎恒为常数，再被紧随其后的 BatchNorm 当作常数偏置吸收。'
     '实测特征能量比：输入时框内 / 框外为 1.19~1.94，经过 CBAM 后变成 0.98~1.09——'
     '注意力把差异抹平了。')

h2('② 加 P2 检测头：有尺度，没样本', before=8)
para('动机很直接：断股中位 40 px，加一层 1/4 分辨率的检测分支应该有帮助。'
     '第一次实验发现新增的 161 个张量全部零迁移（初始化缺陷），修好初始化后重跑，仍然 −0.69 pt。')
para('归因：mosaic 0.8 意味着 80% 的样本由 4 图拼接，每 batch 只有约 46 个正样本；'
     '分摊到 4 个检测尺度后，P2 分支拿到的正样本密度更低。'
     '"增加了尺度，但单位面积的正样本被稀释"——这与后来 mosaic 削弱失败是同一个机理。')

h2('③ 削弱 mosaic：掉了 2.92 分，反向证明它是必需的', before=8)
para('原本的假设是：断股在 92% 的训练轮里被 4 图拼接、缩到一半，细节被系统性稀释，'
     '所以把 mosaic 从 0.8 降到 0.4 应该改善断股。')
para('实测方向完全相反：断股 −3.59 pt，三类全跌，总体 −2.92 pt（179 图）/ −2.69 pt（357 图合并集）。'
     '这是 13 个方向里唯一越过噪声地板的，而且方向是负的。'
     '结论不是"mosaic 没用"，而是"mosaic 在这份小数据上是必需的正则，拿掉它模型立刻过拟合"。')

h2('三类失败原因归纳', before=14)
rows = [['失败类型', '代表方向', '机理'],
        ['加了等于没加', 'CBAM、TripletAttention', '注意力被下一层 BN 吸收；或参数变化不触及瓶颈'],
        ['加了反而更差', 'P2 小目标头 v1 / v2', '增加检测尺度稀释了单位面积的正样本密度'],
        ['破坏必需正则', 'mosaic 0.8→0.4', '小数据上 mosaic 是主要正则，削弱后立刻过拟合'],
        ['已到平台期', '150 轮、1024 分辨率', '训练早于 84% 处过峰；896 后分辨率已饱和'],
        ['单种子不可判', 'WIoU、过采样、WBF、类级选择', '效应量落在 1.6 pt 噪声地板内，无法与随机波动区分']]
table(rows, [3.2, 5.4, 7.4], size=8.8)
p = doc.add_paragraph()
p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(3)
p.paragraph_format.space_after = Pt(6)
cn(p.add_run('表 2  13 个方向按失败原因归并'), '宋体', 9, False, GRAY)

# ============================================================
# 第三部分
# ============================================================
h1('三、为什么模型无法提升：核心证据', newpage=True)

para('如果 13 个方向都无效，最自然的解释是"方法没选对"。但我们有更硬的解释：'
     '模型已经贴到了标注精度的天花板。', indent=True)

h2('3.1 两个数字的比较', before=10)

rows = [['指标', '数值', '含义'],
        ['标注自身的框尺寸抖动', '13.8%', '相邻帧同一目标，相对宽差的标准差'],
        ['模型预测的框尺寸抖动', '14.65%', '同一模型在相邻帧上的预测，同样口径'],
        ['两者比值', '1.06', '模型只比人工标注的噪声高 6%']]
table(rows, [5.6, 2.8, 7.3], size=9)
p = doc.add_paragraph()
p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(3)
p.paragraph_format.space_after = Pt(8)
cn(p.add_run('表 3  标注噪声与模型误差的对比'), '宋体', 9, False, GRAY)

para('这个对比的构造方式很关键：数据集是按帧抽取的，编号相邻的两张图往往来自同一段视频。'
     '找出两张相邻帧里 IoU > 0.3 的同类框配对（认为是同一个物理目标），比较它们的宽高——'
     '如果标注完全一致，相对差应该接近 0。'
     '检验前提也做了：帧间隔从 1 增加到 20，抖动只从 0.138 涨到 0.188，'
     '那 0.05 的增量才是真实形变，其余 0.138 就是噪声地板。')

fig(f'{FIG}/fig7_adjacent_label.png', 16.4,
    '图 3  相邻两帧上同一个断股目标的标注框：IoU 仅 0.478（训练集 1633 / 1634，1920×1080）',
    '两帧隔不到 0.1 秒，物理形变可忽略；框宽 47 px、中心却水平偏了 15.5 px（相当于框宽的 33%），'
    '重叠面积只剩 48%。这不是个别现象，全库 1131 对相邻帧里，同一目标跨帧 IoU 的中位数是 0.477。')

h2('3.2 一个反直觉的旁证：大框反而最差', before=0, newpage=True)
para('按目标边长分桶统计，如果误差来自"看不清"，应该是小目标差、大目标好。'
     '实测恰恰相反：断股在 ≥96 px 这一档 AP50 只有 0.354，而同一档的雷击损伤是 0.966。')
para('几何上大框的容错更宽松，检测难度更低，实际却掉得最狠——'
     '这只能来自标注边界的不一致：一根散开的断股，"框整个散开区域"还是"只框断点"，'
     '不同标注者、不同帧能差两三倍。目标越大，这种分歧的绝对幅度越大。')

h2('3.3 逐框追踪：不是看不见，是不敢确认', before=10)
para('对断股表现最差的 1920×1080 桶（71 张图、153 个框）做逐框追踪，conf 0.001 全量输出：')

rows = [['类别', '标注框', '正确检出', '检出但 conf<0.25', '框歪', '认错类', '真漏检'],
        ['断股', '77', '58', '13（其中 9 个 <0.1）', '6', '0', '0'],
        ['雷击', '31', '31', '0', '0', '0', '0'],
        ['磨损', '45', '35', '9', '1', '0', '0']]
table(rows, [2.2, 2.4, 2.4, 4.4, 1.8, 1.6, 1.6], size=9)
p = doc.add_paragraph()
p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(3)
p.paragraph_format.space_after = Pt(8)
cn(p.add_run('表 4  1920×1080 桶内逐标注框去向（IoU 0.5 匹配）'), '宋体', 9, False, GRAY)

para('断股是 0 个真漏检、0 个认错类，92.2% 的框在 IoU 0.5 内框到了。'
     '失败全部集中在两类：13 个检出但置信度低于 0.25（其中 9 个低于 0.1），6 个框歪——'
     '而 6 个框歪的全部落在目标边长小于 80 px 的位置。'
     '也就是说，模型从来没有"看不见断股"，它是"看见了但不敢确认"。')

fig(f'{FIG}/fig6_break_failure.png', 15.8,
    '图 4  (a) 三类缺陷的逐框去向；(b) 预测框冗余倍数与 AP 的对应关系',
    '断股每张图报 6.2 个框，真实密度只有 1.08 个，冗余 5.7 倍；'
    '但 368 个误报里 95% 的置信度低于 0.25（中位 0.003），部署时会被阈值滤掉——'
    '它们不是背景误检，是同一目标的多重响应。')

h2('3.4 为什么再加模块也不会有效', before=0, newpage=True)
para('把上面三组证据合起来，可以得到一个三层结构，越往下越硬：')

h2('第一层（模型侧）：效应量天花板', before=8)
para('所有模块类改动的真实效应在 0.2~0.5 pt，而三种子噪声地板是 1.60 pt。'
     '这不是"这次没调好"，而是在这个数据规模与这个判据下，模块类改动的信号天然淹没在噪声里。'
     '13 个方向里唯一的越线者（mosaic 削弱，−2.92）是"拿掉一个必需正则"，属于破坏性操作，'
     '恰好说明了同一件事：只有伤筋动骨才会越出噪声。')

h2('第二层（判据侧）：统计设计的数学极限', before=8)
para('三种子的符号检验，p 值最小只能到 2/2³ = 0.25，无论结果多一致都达不到 0.05 显著。'
     '要达到 p < 0.05 需要 n ≥ 6 且全部同号。这是统计设计问题，不是实验质量问题——'
     '换句话说，即使某个改进真的有效，用 3 个种子也无法"证明"它有效，只能"看不出它无效"。')

h2('第三层（数据侧）：真瓶颈', before=8)
para('标注噪声 13.8%、模型抖动 14.65%、相邻帧框 IoU 中位 0.477、大框 AP 反而最低——'
     '四条独立证据指向同一个结论：模型不是在拟合真实世界，而是在拟合一份边界不一致的标注。'
     '它学到的是"平均位置"，而这个平均位置与任何单个标注框的 IoU 都不高。')
para('在这个前提下，继续调模型等于在噪声里捞信号。这解释了为什么 13 个方向全部失败，'
     '也说明了为什么"再试一个模块"不是合理的下一步。')

rows = [['层次', '瓶颈', '量化证据', '能否靠调模型解决'],
        ['模型侧', '效应量天花板', '改动效应 0.2~0.5 pt，噪声地板 1.60 pt', '不能'],
        ['判据侧', '统计功效不足', 'n = 3 符号检验 p 最小值 0.25', '不能（需 n ≥ 6）'],
        ['数据侧', '标注噪声', '抖动 13.8%，模型 14.65%，比值 1.06', '不能（需重标注）']]
table(rows, [2.4, 4.2, 6.6, 3.4], size=8.8)
p = doc.add_paragraph()
p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(3)
p.paragraph_format.space_after = Pt(8)
cn(p.add_run('表 5  三层瓶颈与对应证据'), '宋体', 9, False, GRAY)

para('三层的关系是递进的：模型侧的效应量之所以测不出来，一半原因在判据侧（尺子太粗），'
     '另一半原因在数据侧（被测对象本身在晃）。而判据侧可以通过加种子数解决，数据侧只能通过重标注解决——'
     '两者都不在网络结构里。')

# ============================================================
# 第四部分
# ============================================================
h1('四、工程排雷与自我修正', newpage=True)

para('这一节记录实验过程中差点导致错误结论的坑。它们的共同点是：'
     '如果没被发现，结论会错，而且错得没有任何报错提示。', indent=True)

h2('① YOLO26 代际对比：环境版本导致静默降级', before=10)
para('第一次评估 YOLO26s 时，用的是 YOLOv11 那套环境（ultralytics 8.3.180）。'
     '该版本没有 YOLO26 的 end2end 推理路径与 reg_max=1 配置，加载权重后静默降级成常规检测头，'
     '不报任何错。结果是 mAP50 只有 0.8145，比 V11 低 5.94 分——差点写成"新一代反而更差"。')
rich([('换成对应版本（8.4.178）重评后，三颗种子为 0.9130 / 0.9296 / 0.9152，均值 0.9193，'
       '与基线 0.9212 只差 ', False, None), ('−0.19 pt', True, ACC),
      ('，判定从"显著更差"改为"打平"。两套环境的差距是 ', False, None),
      ('9.85 pt', True, ACC), ('。', False, None)])

h2('② best.pt 是移动靶：训练中途和终态不是一回事', before=10)
para('训练到第 N 轮时的 best.pt，与训练结束时的 best.pt 是两个不同的文件。'
     '150 轮实验里，best.pt 在 ep126 就已经出现，后 24 轮的验证指标的标准差是 0.42 且趋势为负——'
     '模型其实早已过峰。')
para('这直接影响结论：用 best.pt 口径看，150 轮比 60 轮"提升" +0.84 / +2.13 pt；'
     '用 last.pt 口径看是 −0.09 / +0.54。前者那 0.84 分来自"多了 5 个检查点可以挑"，'
     '不是模型变强。所以本项目所有主表统一用 last.pt 口径。')

h2('③ Windows 多进程：workers > 0 会卡死', before=10)
para('DataLoader 的 worker 进程在 Windows 上通过 spawn 启动，与训练主进程争抢显存与句柄，'
     '表现为训练在第一个 batch 前无声挂起，日志停在"Scanning"且不报错。'
     '改成 workers=0（主进程加载）后稳定。代价是数据加载慢，对本项目的数据规模可接受。')

h2('④ fitness 公式的版本差异', before=10)
para('模型选择用的适应度函数在 ultralytics 不同版本里权重不同（本项目口径为 '
     'fitness = 0.1 × mAP50 + 0.9 × mAP50-95）。若用另一版本的公式复算历史 run 的 best.pt，'
     '会选出不同的检查点。所有跨版本比较必须先统一这个口径。')

h2('⑤ rect 矩形输入：框架静默清零四个增强项', before=10)
para('为消除 letterbox 灰边，尝试了 rect=True 的矩形输入。第 8 轮训练发散（'
     'train/box 从 2.482 单调升到 3.021，mAP50 从 0.0198 降到 0.0123，'
     '而同期基线在第 7 轮已经到 0.4379）。')
para('查框架源码发现：data/dataset.py 在 rect=True 时无条件把 mosaic / mixup / cutmix 置零，'
     '并强制 shuffle=False，同时改动了四个变量且不提示。'
     '因此这次实验无法构成单变量对照，只能归为"未能形成对照"，不计入 13 个方向。'
     '这也是本研究的一条方法论结论：框架的"便利开关"可能偷偷改变实验设计。')

# ============================================================
# 第五部分
# ============================================================
h1('五、后续规划与展望', newpage=True)

para('既然瓶颈在数据，下一步有价值的投入就只剩数据侧。模型侧的低成本选项已经穷尽——'
     '13 个方向 + 6 条诊断路径全部试完，0 个越过噪声地板。', indent=True)

h2('① 重新标注：统一边界定义', before=10)
para('当前 1785 张图、3010 个框的标注边界不一致，这是四条独立证据共同指向的问题。'
     '重标注的核心不是"画得更准"，而是先定义清楚规则：')
bullet('断股——', '是否包含整段散开区域？散开的钢丝末端是否计入？一束中多根断裂算一个框还是多个？')
bullet('雷击——', '灼伤边缘的渐变色带算不算？相邻两个灼伤点是否合并？')
bullet('磨损——', '成片磨损的边界如何界定？小块磨损是否单独成框？')
para('规则确定后需做一致性检验：同一批图由两人独立标注，计算框 IoU 分布，'
     '目标是把跨帧 / 跨人的 IoU 中位从当前的 0.477 提到 0.75 以上。'
     '这是唯一可能把 AP 真正推上去的路径。')

h2('② 按视频段重新划分数据集（已完成，三轮）', before=10)
para('原始划分按 8:1:1 随机抽样，而数据是帧抽取的，同一段视频的相邻帧被分进了不同集合——'
     '179 图 test 里 44.1%（79 张）与训练图同处一个视觉段（dHash 汉明 ≤2 且 64×64 灰度 MAE ≤20 '
     '双闸聚类判定的 1080 个视觉段）。模型在 test 上看到的几乎是它背过的帧。')
para('第一轮：帧级隔离，剔除同段图后剩 100 张，基线由 92.12% 降到 89.98%；'
     '但相邻段仍在 train，且磨损类只剩 8 图 31 框，统计功效不足。')
para('第二轮：把视觉段当原子整体重划为 1401 / 181 / 203（跨段 0）并从头重训 6 个 run，'
     '三种子 test 74.67%，同一颗权重在 val 上 89.15%。下降幅度不是 2 pt 而是 15 pt 量级——'
     '而且其中相当一部分来自"抽到的段更难"（val 与 test 的图来源构成几乎相同，'
     'break 单类却差 21.6 pt），并不全是模型变差。')
para('第三轮：既然"抽到哪批段"能造成 13 pt 波动，就用六目标局部搜索把段划分本身配平'
     '（图数、三类框数、break 面积和、全体面积和；400k 次交换退火）。结果三类框在三个 split 的占比'
     '全部落在 9.7~11.4%，越界项 0，跨段泄漏仍为 0。该划分已落地，待重训（尚未跑）。')

h2('③ 如果要继续做模型侧：先把判据升级', before=10)
para('若还要评估小效应，必须先把种子数从 3 提到 6 以上（单配置约 10 小时），'
     '否则任何 ±0.5 pt 的结论都不可判。这一步不产生新模型，但决定了后续所有实验是否"可判"。')

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(16)
p.paragraph_format.line_spacing = 1.5
cn(p.add_run('结语：'), '黑体', 10.5, True, ACC)
cn(p.add_run('本项目的主要产出不是"一个更好的检测器"，而是一次完整的证伪：'
             '用 13 个方向、6 条诊断路径和四条独立的标注侧证据，'
             '把"模型侧还有多少空间"这个问题量化到了 1.6 pt 的噪声地板以下。'
             '剩下的空间在数据里，不在网络结构里。'), '宋体', 10.5, False, DARK)

doc.save(OUT)
print('saved', OUT, os.path.getsize(OUT))
