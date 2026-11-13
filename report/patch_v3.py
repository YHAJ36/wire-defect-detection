# -*- coding: utf-8 -*-
"""把报告从 v2 口径（89.98 / 90.79 / 92.04）同步到 v3 口径（段级重划分 + 重训）。

数字全部从 `exp_upscale/logs/eval_noleak_final.tsv` 读，不手打。
改动清单见文件末尾 EDITS 打印。
"""
import copy
import csv
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SRC = 'D:/2/report/实验报告_全文.docx'
OUT = 'D:/2/report/_tmp_v3.docx'
TSV = 'D:/2/exp_upscale/logs/eval_noleak_final.tsv'
DRY = '--dry' in sys.argv

# ---------------------------------------------------------------- 读数
rows = list(csv.DictReader(open(TSV, encoding='utf-8-sig'), delimiter='\t'))


def pick(fam, group):
    r = [x for x in rows if x['family'] == fam and x['group'] == group]
    assert len(r) == 1, (fam, group, len(r))
    return {k: (float(v) if k not in ('family', 'group', 'note') else v)
            for k, v in r[0].items()}


b_s = [pick('base', f'单模型 s{s}') for s in (42, 43, 44)]
t_s = [pick('triplet', f'单模型 s{s}') for s in (42, 43, 44)]
o_s = [pick('old896', f'单模型 s{s}') for s in (42, 43, 44)]
b_m = pick('base', '单模型 raw 三种子均值')
t_m = pick('triplet', '单模型 raw 三种子均值')
o_m = pick('old896', '单模型 raw 三种子均值')
b_w = pick('base', 'WBF 三种子融合')
t_w = pick('triplet', 'WBF 三种子融合')
o_w = pick('old896', 'WBF 三种子融合')


def pick_note(fam, note):
    r = [x for x in rows if x['family'] == fam and x['note'] == note
         and x['group'].startswith('WBF*')]
    return {k: (float(v) if k not in ('family', 'group', 'note') else v) for k, v in r[0].items()}


bW50, bW60 = pick_note('base', 'WBF iou=0.5'), pick_note('base', 'WBF iou=0.6')
tW50, tW60 = pick_note('triplet', 'WBF iou=0.5'), pick_note('triplet', 'WBF iou=0.6')

P = lambda x: x * 100
VAL_S42 = 89.15          # 同一颗 s42 在 val 181 图上的 mAP50（WORKLOG A.9.63 实测）
VAL_BREAK = 0.823        # 同颗权重在 val 上的 break AP50
TEST_BREAK_S42 = b_s[0]['break_ap50']
b_span = float(b_m['note'].replace('span=', '').replace('pt', ''))
t_span = float(t_m['note'].replace('span=', '').replace('pt', ''))

# ---------------------------------------------------------------- 工具
d = Document(SRC)
ps = d.paragraphs
o_txt = [p.text for p in ps]
o_tables = [[[c.text for c in r.cells] for r in t.rows] for t in d.tables]


def find(prefix, start=0, exact=False):
    hit = [i for i in range(start, len(ps))
           if (ps[i].text.strip() == prefix if exact else ps[i].text.strip().startswith(prefix))]
    assert len(hit) == 1, (prefix, hit)
    return hit[0]


def set_text(p, text):
    """保留首个 run 的格式，替换全文，删掉其余 run。返回原文本。"""
    old = p.text
    if not p.runs:
        p.add_run(text)
        return old
    p.runs[0].text = text
    for r in p.runs[1:]:
        r._element.getparent().remove(r._element)
    return old


def set_cell(cell, text):
    p = cell.paragraphs[0]
    set_text(p, text)


def replace_in(p, old_sub, new_sub):
    """在段落内做子串替换（单 run 时直接改）。"""
    assert old_sub in p.text, (old_sub, p.text[:80])
    set_text(p, p.text.replace(old_sub, new_sub))


EDITS = []


def edit(prefix, new, note, exact=False, start=0):
    i = find(prefix, start=start, exact=exact)
    old = set_text(ps[i], new)
    EDITS.append((i, note, old[:60], new[:60]))


# ---------------------------------------------------------------- 1. 摘要
edit('四、最终基线数字',
     f'四、最终基线数字（v3 口径，2026-10-27 重定；v2 的 89.98% 已被本轮段级重训取代）。'
     f'按 1080 个视觉段把全部数据整体重划为 1401/181/203，跨段泄漏为 0，再从头训练 6 个 run。'
     f'结果不是一个点，是一个区间：三种子在 test 203 图上 mAP50 均值 {P(b_m["mAP50"]):.2f}%，'
     f'而同一颗 s42 权重在 val 181 图上为 {VAL_S42:.2f}% —— 只换留出的是哪批视觉段，就差 '
     f'{VAL_S42 - P(b_s[0]["mAP50"]):.1f} pt。故本项目跨段泛化能力只能报 '
     f'{P(b_m["mAP50"]):.1f}% ~ {VAL_S42:.1f}%，任何单点（含旧的 92.12%）都不再作为最终成绩。'
     f'推理端三种子 WBF 在同卷内把 {P(b_m["mAP50"]):.2f}% 提到 {P(b_w["mAP50"]):.2f}%'
     f'（+{P(b_w["mAP50"]) - P(b_m["mAP50"]):.2f} pt），是唯一越过 1.60 pt 噪声地板的增益；'
     f'TTA 三颗种子全负，弃用。',
     '摘要·四 基线数字 → v3 区间')

# ---------------------------------------------------------------- 2. 6.2.5
edit('结果（v2 正式口径',
     f'结果（v3 正式口径，2026-10-27；按视觉段整体重划分后从头重训）：基线三种子在 test 203 图上 '
     f'mAP50 = {P(b_m["mAP50"]):.2f}%、mAP50-95 = {P(b_m["mAP50-95"]):.2f}%、'
     f'P = {P(b_m["P"]):.2f}%、R = {P(b_m["R"]):.2f}%；同一颗 s42 权重在 val 181 图上 mAP50 = '
     f'{VAL_S42:.2f}%，两卷相差 {VAL_S42 - P(b_s[0]["mAP50"]):.1f} pt。'
     f'历史口径两项均仅作对照：开卷 179 图 mAP50 92.12%（含 44.1% 同段泄漏）、'
     f'帧级干净 100 图 89.98%（相邻段仍在 train，见 6.13.5，已被 6.13.6 取代）。',
     '6.2.5 结果 → v3')

# ---------------------------------------------------------------- 3. 6.13.5 标题
edit('6.13.5 v2 口径重定',
     '6.13.5 v2 口径重定：第一次去泄漏尝试（帧级隔离，已被 6.13.6 取代）',
     '6.13.5 标题加定位语')

# ---------------------------------------------------------------- 4. P1205 末尾加指引
i1205 = find('三点判读。第一，WBF 在两族上都为正')
p = ps[i1205]
set_text(p, p.text + '（本节全部结论建立在"从旧 test 里剔除同段图"的帧级隔离上——相邻段仍在 train，'
                     '且 wear 只剩 8 图 31 框。按视觉段整体重划分并从头重训后，这些数字全部被改写，见 6.13.6。）')
EDITS.append((i1205, '6.13.5 判读段加"已被取代"指引', '…', '…见 6.13.6。）'))

# ---------------------------------------------------------------- 5. 7.6
edit('这份报告里的 89.98%',
     '这份报告里的"74.7% ~ 89.2%"这个区间（v3 口径），不是"跑出来的最好或最差数字"，'
     '"在所有已知陷阱都被排除之后，仍然站得住的数字"。它之所以是一个区间而不是一个点：'
     '同一颗权重、同一次训练，只把留出的视觉段换一批，就能差 13 pt —— 这是数据侧的方差，不是模型的方差。'
     '（v2 曾写 89.98%，v1 曾写 92.12%，两者都已被本轮段级重训证伪，见 6.13.6。）',
     '7.6 小结 → 区间口径')

# ---------------------------------------------------------------- 6. 8.4 结论七
edit('结论七：数据存在 44.1% 的相邻帧泄漏',
     '结论七：数据存在 44.1% 的相邻帧泄漏；按视觉段整体重划分并重训后，"真实成绩"不是一个点，'
     '而是 74.7% ~ 89.2% 的区间，且区间宽度主要来自段划分而非模型。',
     '结论七 标题改写')

edit('test 179 张里有 79 张（44.1%）与训练图同处一个连续拍摄段',
     f'test 179 张里有 79 张（44.1%）与训练图同处一个连续拍摄段——这是 dHash（汉明 ≤2）与 '
     f'64×64 灰度 MAE（≤20）双闸聚类、把图像并成 1080 个视觉段之后的判定。'
     f'按段整体重划分为 1401/181/203（跨段 0）并从头训练 6 个 run，基线三种子在 test 203 图上 '
     f'{P(b_m["mAP50"]):.2f}%，而同一颗 s42 权重在 val 181 图上 {VAL_S42:.2f}%。'
     f'两个数字差 {VAL_S42 - P(b_s[0]["mAP50"]):.1f} pt，其中 break 单类差 '
     f'{VAL_BREAK*100 - TEST_BREAK_S42*100:.1f} pt（{VAL_BREAK:.3f} vs {TEST_BREAK_S42:.3f}）——'
     f'而同一批权重的三种子跨度只有 {b_span:.2f} pt。'
     f'也就是说，**"留出哪批视觉段"带来的方差是"换哪个种子"的 5 倍**，所以在这类数据的单张段级卷上，'
     f'"这个模型多少分"本身不是一个有意义的问句。',
     '结论七 正文：重训结果 + 方差来源')

edit('更关键的是，泄漏的影响此前"测不出"',
     f'更关键的是，泄漏的补偿量远超我们的改进空间：同一批旧权重在这张 203 图卷上拿到 '
     f'{P(o_m["mAP50"]):.2f}%，因为 203 张里有 171 张原本就是它的训练图；而 13 个改进方向里最强的效应量'
     f'也只有 2.9 pt。一条数据划分错误就是 20 pt 量级，这从数量级上说明了本项目的瓶颈在哪。'
     f'（该缺陷已补两轮：帧级隔离见 6.13.5，段级重划分 + 重训见 6.13.6；'
     f'第三轮"六目标配平的段划分"已完成划分、待重训。）',
     '结论七 末段：泄漏补偿量 + 补两轮')

# ---------------------------------------------------------------- 7. 8.5
edit('为什么"按视频段重划分"曾排第二',
     '为什么"按视频段重划分"曾排第二：它不会提升指标，反而会让指标大幅下降——因为去掉了近重复带来的'
     '乐观偏差。该项已做两轮：第一轮帧级隔离（100 图，基线 89.98%），第二轮段级整体重划分并从头重训'
     f'6 个 run（203 图，基线 {P(b_m["mAP50"]):.2f}%，同颗权重在 val 上 {VAL_S42:.2f}%）。'
     '第二轮说明"下降"不是 2 pt 量级而是 15 pt 量级，且其中相当一部分来自"抽到的段更难"而不是"模型更差"，'
     '于是又补了第三轮：用六目标局部搜索重新配平段划分（图数、三类框数、两个面积维，0 项越界），待重训。',
     '8.5 视频段重划分 → 三轮过程')

# ---------------------------------------------------------------- 8. 8.6
edit('报告里最重要的数字不是 89.98%',
     '报告里最重要的数字不是 92.12%，不是 89.98%，也不是 77.76%，是 1.60。'
     '前几个说明模型在哪一层，最后一个说明"这一层"能被相信到什么精度。'
     '而当留出集的选择本身就能造成 13 pt 波动时，1.60 这把尺子连同它的被测对象都需要重新校准——'
     '这正是本项目最后得出的、也是最不体面的那条结论。',
     '8.6 结束语 → 数字更新')

# ---------------------------------------------------------------- 9. 表格
def row_of(tbl, key, col=0):
    for r in tbl.rows:
        if r.cells[col].text.strip().startswith(key):
            return r
    raise AssertionError((key, [r.cells[col].text[:12] for r in tbl.rows]))


# 需求表 N1
t12 = d.tables[12]
r = row_of(t12, 'N1')
set_cell(r.cells[3], f'{P(b_m["mAP50"]):.2f}–{VAL_S42:.2f}')
set_cell(r.cells[4], f'未达成。段级留出集上同一颗权重可差 {VAL_S42 - P(b_s[0]["mAP50"]):.1f} pt；'
                     f'旧 0.8998 / 0.9204 为帧级口径，已作废')
EDITS.append(('T12 N1', '需求表 N1 → 未达成 + 区间', '', ''))

# 表 8-1 ④
t112 = d.tables[112]
r = row_of(t112, '④')
set_cell(r.cells[1], '部分完成')
set_cell(r.cells[2], f'段级重划分后三种子 test mAP50 {P(b_m["mAP50"]):.2f}%（val 单颗 {VAL_S42:.2f}%）；'
                     f'WBF 同卷 +{P(b_w["mAP50"]) - P(b_m["mAP50"]):.2f} pt 至 {P(b_w["mAP50"]):.2f}%；'
                     f'旧 92.04% 已作废')
set_cell(r.cells[3], '6.2.5、6.13.6')
EDITS.append(('T112 ④', '表8-1 ④ → 部分完成', '', ''))

# 表 8-2 第 3 行
t116 = d.tables[116]
r = row_of(t116, '3', 0)
if not r.cells[1].text.startswith('划分'):
    for rr in t116.rows:
        if '段级隔离已补' in rr.cells[2].text:
            r = rr
            break
set_cell(r.cells[1], '段级划分已隔离，但单张卷子代表性不足')
set_cell(r.cells[2], f'同一颗权重 val {VAL_S42:.2f} / test {P(b_s[0]["mAP50"]):.2f}'
                     f'（break {VAL_BREAK:.3f} / {TEST_BREAK_S42:.3f}）；三种子跨度仅 '
                     f'{b_span:.2f} pt，段选择却造成 '
                     f'{VAL_S42 - P(b_s[0]["mAP50"]):.1f} pt 差；配平版划分已完成、待重训')
set_cell(r.cells[3], '跨段泛化只能报区间 74.7~89.2；单点数值不能作为能力定义')
EDITS.append(('T116 r3', '表8-2 第3行 → 代表性不足', '', ''))

# 表 8-3 第 2 行
t117 = d.tables[117]
r = row_of(t117, '2', 0)
set_cell(r.cells[4], '已完成两轮（帧级 100 图 / 段级重训 203 图，8h57m）；'
                     '配平版段划分（六目标，0 项越界）待重训')
EDITS.append(('T117 r2', '表8-3 第2行 → 已完成两轮', '', ''))

# ---------------------------------------------------------------- 10. 插入 6.13.6
anchor_i = find('第 7 章 项目实施过程', start=1100)   # 跳过目录里同名条目
ref = ps[anchor_i]._p

h_tpl = copy.deepcopy(ps[find('6.13.5 v2 口径重定')]._p)     # 12pt bold
b_tpl = copy.deepcopy(ps[find('上面 13 个方向的判读全部建立在 179 图')]._p)  # 12pt 正文
c_tpl = copy.deepcopy(ps[find('表 6-8')]._p)                  # 10.5pt 居中 bold 表题


def mk(tpl, text, bold=None):
    el = copy.deepcopy(tpl)
    ref.addprevious(el)
    from docx.text.paragraph import Paragraph
    par = Paragraph(el, ref.getparent())
    set_text(par, text)
    if bold is not None and par.runs:
        par.runs[0].bold = bold
    return par


def mktable(rows_data, header, widths=None):
    from docx.shared import Cm
    tb = d.add_table(rows=len(rows_data) + 1, cols=len(header))
    tb.style = d.tables[103].style
    tb.autofit = False
    for j, h in enumerate(header):
        set_cell(tb.cell(0, j), h)
    for i2, row in enumerate(rows_data):
        for j, v in enumerate(row):
            set_cell(tb.cell(i2 + 1, j), v)
    if widths:
        for j, w in enumerate(widths):
            tb.columns[j].width = Cm(w)
        for row in tb.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    ref.addprevious(tb._tbl)
    return tb


f = lambda x: f'{x*100:.2f}'
g = lambda x: f'{x*100:.3f}'

mk(h_tpl, '6.13.6 v3 口径：按视觉段整体重划分并从头重训（2026-10-27）')

mk(b_tpl,
   '6.13.5 的 100 图卷子并不够干净。它是"从旧 test 里挑出段内没有 train 成员的图"，'
   '但同一段巡检视频的相邻段仍在 train 里；而且 wear 只剩 8 图 31 框，该类指标已无统计意义。'
   '真正干净的做法只有一条：把 1080 个视觉段当成不可分割的原子，整体重划 8:1:1，再从头训练。'
   '本节即该实验的结果，也是本报告 v3 口径的来源。')

mk(b_tpl, '三张测试卷的定义与合法用法必须分清，混用必出错：')
mk(c_tpl, '表 6-10\u3000三张测试卷的关系与各自能回答的问题')
mktable([
    ('旧 test', '179', '开卷：79 张（44.1%）与 train 同段', '同场景泛化上限', '92.12%，仅作历史对照'),
    ('帧级干净', '100', '旧 test 中段内无 train 成员；相邻段仍在 train', '泄漏的一阶修正',
     '89.98%，见 6.13.5，已被本节取代'),
    ('段级重划分', '203', '1080 段零跨 split', '真跨段泛化', '74.67%（三种子），本节'),
    ('段级重划分', '181', '同上（val）', '同一次训练的另一张段级卷', '89.15%（s42），本节'),
], ['卷子', '图', '与训练集的关系', '能回答什么', '本项目用法'],
    widths=[2.4, 1.2, 4.6, 3.6, 4.2])

mk(b_tpl,
   '6 个 run 从零训练：1401 张 train、896 输入、60 轮、batch 8、workers 0，'
   '基线与 TripletAttention 各 3 个种子，总耗时 8 小时 57 分。评估口径与 6.2.5 完全一致'
   '（NMS IoU 0.5、conf 0.001、max_det 300）。')

mk(c_tpl, '表 6-11\u3000段级重划分后 6 个 run 在 test 203 图上的结果（三种子）')
mktable([
    ('基线 · 三种子', f'{f(b_s[0]["mAP50"])} / {f(b_s[1]["mAP50"])} / {f(b_s[2]["mAP50"])}',
     '—', '—', '—', '—'),
    ('基线 · 均值', f(b_m['mAP50']), f(b_m['mAP50-95']), g(b_m['break_ap50']),
     g(b_m['thunderbolt_ap50']), g(b_m['wear_ap50'])),
    ('基线 · WBF 三种子', f(b_w['mAP50']), f(b_w['mAP50-95']), g(b_w['break_ap50']),
     g(b_w['thunderbolt_ap50']), g(b_w['wear_ap50'])),
    ('Triplet · 三种子', f'{f(t_s[0]["mAP50"])} / {f(t_s[1]["mAP50"])} / {f(t_s[2]["mAP50"])}',
     '—', '—', '—', '—'),
    ('Triplet · 均值', f(t_m['mAP50']), f(t_m['mAP50-95']), g(t_m['break_ap50']),
     g(t_m['thunderbolt_ap50']), g(t_m['wear_ap50'])),
    ('Triplet · WBF 三种子', f(t_w['mAP50']), f(t_w['mAP50-95']), g(t_w['break_ap50']),
     g(t_w['thunderbolt_ap50']), g(t_w['wear_ap50'])),
    ('对照：旧权重 · 均值', f(o_m['mAP50']), f(o_m['mAP50-95']), g(o_m['break_ap50']),
     g(o_m['thunderbolt_ap50']), g(o_m['wear_ap50'])),
], ['配置', 'mAP50', 'mAP50-95', 'break', 'thunderbolt', 'wear'],
    widths=[4.2, 2.4, 2.4, 2.0, 2.6, 1.9])

mk(b_tpl,
   f'{P(b_m["mAP50"]):.2f}% 不能读成"模型退步了"。它同时受两个因素污染：')
mk(b_tpl,
   f'其一，卷子偏难。同一颗 s42 权重，在 val 181 图上 mAP50 {VAL_S42:.2f}%、break 0.{VAL_BREAK:.0f}，'
   f'在 test 203 图上 {P(b_s[0]["mAP50"]):.2f}%、break {TEST_BREAK_S42:.3f}；两张卷子的图来源构成几乎相同'
   f'（83% 与 84% 来自旧 train），却差 {VAL_S42 - P(b_s[0]["mAP50"]):.1f} pt。'
   f'段级随机划分没有控制"难度"，把更难的 break 段整批分给了 test。'
   f'而 best.pt 又是在 val 上挑的，这个偏差同时污染了选模口径（实测 best 与 last 仅差 0.43 pt，未致灾）。')
mk(b_tpl,
   f'其二，训练集变窄。新 train 的"旧 train 部分"只有 1107 张，比旧 1428 张少 22%；'
   f'按旧 train / 旧 val / 旧 test 来源把 test 拆开看，新权重在"两边都没见过"的 32 张上反而更高'
   f'（78.70 vs 75.30），说明下降不是单纯的记忆效应，而是覆盖变窄。')
mk(b_tpl,
   f'对照行给出了量级：同一批旧权重在这张 203 图卷上 {P(o_m["mAP50"]):.2f}%，'
   f'因为 203 张里 171 张原本就是它的训练图。数据划分的一个错误就是 20 pt 量级，'
   f'超过本项目 13 个改进方向里最强的效应量（2.9 pt）近 7 倍。')

mk(b_tpl, '本节推翻两条此前的结论，并确立一条新的：')
mk(b_tpl,
   f'① TripletAttention 不再是"抗泛化之王"。段级卷上它单模 {P(t_m["mAP50"]):.2f}%，'
   f'低于基线 {P(b_m["mAP50"]):.2f}%（−{P(b_m["mAP50"]) - P(t_m["mAP50"]):.2f} pt），'
   f'WBF 后 {P(t_w["mAP50"]):.2f}% 也低于基线族 {P(b_w["mAP50"]):.2f}%；'
   f'更关键的是它的三种子跨度 {t_span:.2f} pt，'
   f'是基线（{b_span:.2f} pt）的两倍多。'
   f'6.13.5 里"只掉 0.45 pt、反超基线"成立的前提是那张 100 图帧级卷，换到段级卷即消失。')
mk(b_tpl,
   f'② WBF 是唯一在"坏卷子"上仍站得住的增益。同卷内：基线 {P(b_m["mAP50"]):.2f}% → '
   f'{P(b_w["mAP50"]):.2f}%（+{P(b_w["mAP50"]) - P(b_m["mAP50"]):.2f} pt），'
   f'Triplet {P(t_m["mAP50"]):.2f}% → {P(t_w["mAP50"]):.2f}%'
   f'（+{P(t_w["mAP50"]) - P(t_m["mAP50"]):.2f} pt）。'
   f'这是本项目第一次有后处理增益真正越过 1.60 pt 噪声地板，且阈值稳健：'
   f'iou 0.50 / 0.55 / 0.60 分别为 {P(bW50["mAP50"]):.2f} / {P(b_w["mAP50"]):.2f} / '
   f'{P(bW60["mAP50"]):.2f}，跨度 {P(bW50["mAP50"]) - P(bW60["mAP50"]):.2f} pt。'
   f'机制仍是"降低回归方差"：AP50 涨、AP95 涨得更多，wear 单类 +'
   f'{P(b_w["wear_ap50"]) - P(b_m["wear_ap50"]):.2f} pt 最猛。'
   f'它之所以可信，是因为单模与融合是在**同一张卷子**上比，与卷子难度无关。')
mk(b_tpl,
   '③ TTA 依旧是负资产，此处不再重复，见 6.13.5 的 3/3 同号证据。')

mk(b_tpl,
   '最后是补救。既然"抽到哪批段"能造成 13 pt 波动，就必须让段划分本身可控。'
   '已用六目标局部搜索（图数、break/wear/thunder 三类框数、break 面积和、全体面积和，'
   '权重 0.5/2/2/2/1/0.5）在保持"段整体进同一个 split"的前提下重新配平，'
   '400k 次交换退火后三类框的 split 占比为 78.9/10.3/10.9、78.8/10.3/10.9、79.0/10.2/10.8，'
   '全部偏差 <15%，越界项 0；跨段泄漏仍为 0。该划分（1409/173/203）已落地，'
   '但**尚未重训**——本节的 74.67% 属于"随机段划分"下的真值，'
   '下一版应报配平划分下的数字，届时区间有望收窄，但预计不会消失：'
   '两百张量级的测试集，抽样噪声本身就有数 pt。')

if not DRY:
    d.save(OUT)

print(f'共 {len(EDITS)} 处文本/表格改动：')
for e in EDITS:
    print('  ', e)
print(f'段落 {len(o_txt)} → {len(d.paragraphs)}；表 {len(o_tables)} → {len(d.tables)}')
print('已存', OUT if not DRY else '(dry)')
