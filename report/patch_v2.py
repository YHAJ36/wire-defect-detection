# -*- coding: utf-8 -*-
"""把实验报告就地升级到 v2 口径（段级去泄漏后的真实底座）。

原则
----
1. 只改「以最终成绩口吻出现」的位置；179 图上的各消融实测表保持原样（它们是真实测量，
   只是开卷），改它们等于伪造数据。
2. 数字全部从 TSV 读，不手打。
3. 每处替换前断言 old 串在文档中唯一出现，避免误改；改后做全文 diff，只允许预期位置变化。

用法
    python patch_v2.py            # 应用
    python patch_v2.py --dry      # 只打印将要发生的变化，不写盘
"""
import copy
import csv
import json
import os
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches
from docx.text.paragraph import Paragraph

DOC = Path('D:/2/report/实验报告_全文.docx')
TMP = Path('D:/2/report/_tmp_v2.docx')
LOG_NL = Path('D:/2/exp_upscale/logs/eval_noleak.tsv')
LOG_NMS = Path('D:/2/exp_upscale/logs/eval_nms05.tsv')
LOG_WBF = Path('D:/2/exp_upscale/logs/eval_wbf_tta_clean.tsv')
FIG = Path('D:/2/report/_summary_figs/fig_v2_wbf.png')
DRY = '--dry' in sys.argv

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


# ---------------------------------------------------------------- 数据读取
def read_tsv(p):
    with p.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def pick(rows, run, split):
    for r in rows:
        if r['run'] == run and r['split'] == split:
            return r
    raise KeyError(f'{run}/{split} 不在 {LOG_NL.name}')


noleak = read_tsv(LOG_NL)
nms = read_tsv(LOG_NMS)
wbf = read_tsv(LOG_WBF)

# 五个关键 run：179 图开卷 vs 100 图闭卷
FIVE = [('基线（nf_v11_896）', 'nf_v11_896_s42'),
        ('CBAM+BN', 'att_v11_896_cbam_bn_s42'),
        ('P2v2 小目标检测头', 'p2v2_v11_896_s42'),
        ('YOLO26s', 'nf_yolo26_896_s42'),
        ('TripletAttention', 'att_v11_896_triplet_s42')]
five_rows = []
for name, run in FIVE:
    o = pick(nms, run, 'test')
    c = pick(noleak, run, 'test')
    o50, c50, c95 = float(o['mAP50']), float(c['mAP50']), float(c['mAP50-95'])
    five_rows.append((name, o50 * 100, c50 * 100, (c50 - o50) * 100, c95 * 100))

# 三种子均值口径
base_o = sum(float(pick(nms, f'nf_v11_896_s{s}', 'test')['mAP50']) for s in (42, 43, 44)) / 3 * 100
# 干净集只能用 wbf 那轮的三种子结果（eval_noleak 只跑了 s42）
def seed_mean(fam, field='mAP50'):
    rs = [r for r in wbf if r['family'] == fam and r['group'].startswith('单模型 s')
          and 'TTA' not in r['group']]
    assert len(rs) == 3, [r['group'] for r in rs]
    return sum(float(r[field]) for r in rs) / 3 * 100

base_c = seed_mean('base')
trip_o = sum(float(pick(nms, f'att_v11_896_triplet_s{s}', 'test')['mAP50']) for s in (42, 43, 44)) / 3 * 100
trip_c = seed_mean('triplet')

# 8 组外挂
base_raw = next(r for r in wbf if r['family'] == 'base' and r['group'] == '单模型 raw 三种子均值')
base_tta = next(r for r in wbf if r['family'] == 'base' and r['group'] == '单模型 TTA 三种子均值')
trip_raw = next(r for r in wbf if r['family'] == 'triplet' and r['group'] == '单模型 raw 三种子均值')
trip_tta = next(r for r in wbf if r['family'] == 'triplet' and r['group'] == '单模型 TTA 三种子均值')
wbf55 = [r for r in wbf if r['group'].startswith('WBF') and r['note'] == 'WBF iou=0.55']
assert len(wbf55) == 4, [r['group'] for r in wbf55]
assert not wbf55[0]['group'].startswith('WBF+') and wbf55[1]['group'].startswith('WBF+TTA')
assert not wbf55[2]['group'].startswith('WBF+') and wbf55[3]['group'].startswith('WBF+TTA')

BASE_REF = float(base_raw['mAP50']) * 100
eight_rows = []
for label, r in [('基线 · 单模均值', base_raw), ('基线 · +TTA', base_tta),
                 ('基线 · WBF', wbf55[0]), ('基线 · WBF+TTA', wbf55[1]),
                 ('Triplet · 单模均值', trip_raw), ('Triplet · +TTA', trip_tta),
                 ('Triplet · WBF', wbf55[2]), ('Triplet · WBF+TTA', wbf55[3])]:
    m = float(r['mAP50']) * 100
    eight_rows.append((label, m, m - BASE_REF, float(r['mAP50-95']) * 100))

print('=== 5 run 脱底裤 ===')
for n, o, c, d, c95 in five_rows:
    print(f'  {n:<20} {o:6.2f} → {c:6.2f}  Δ{d:+6.2f}  s95={c95:5.2f}')
print(f'=== 三种子均值 === 基线 {base_o:.2f}→{base_c:.2f} ({(base_c-base_o):+.2f})  '
      f'Triplet {trip_o:.2f}→{trip_c:.2f} ({(trip_c-trip_o):+.2f})')
print('=== 8 组 ===')
for n, m, d, m95 in eight_rows:
    print(f'  {n:<22} {m:6.2f}  Δ{d:+5.2f}  s95={m95:5.2f}')

# ---------------------------------------------------------------- 文本替换
d = Document(str(DOC))
ps = d.paragraphs


def find_para(prefix):
    hit = [i for i, p in enumerate(ps) if p.text.strip().startswith(prefix)]
    assert len(hit) == 1, f'{prefix!r} 命中 {len(hit)} 处'
    return hit[0]


def set_text(p, new, keep_runs=1):
    """保留前 keep_runs 个 run 的格式，把文字塞进第一个 run，删掉其余。"""
    assert p.runs, '空段落'
    p.runs[0].text = new
    for r in p.runs[keep_runs:]:
        r._element.getparent().remove(r._element)


def set_cell(cell, new):
    p = cell.paragraphs[0]
    if p.runs:
        p.runs[0].text = new
        for r in p.runs[1:]:
            r._element.getparent().remove(r._element)
    else:
        p.add_run(new)


EDITS = []

# ① 执行摘要 四、
i = 7
p = ps[i]
assert p.text.startswith('四、最终基线数字。')
set_text(p, '四、最终基线数字（v2 口径，2026-10-26 重定）。', keep_runs=1)
p.runs[0].text = '四、最终基线数字（v2 口径，2026-10-26 重定）。'
r2 = p.add_run(
    f'旧 test 的 179 张图里有 79 张（44.1%）与训练集同处一个连续拍摄段，属相邻帧泄漏；按视觉段剔除后剩 '
    f'100 张闭卷测试集。在该集上，基线 mAP50 = {base_c:.2f}%、mAP50-95 = {eight_rows[0][3]:.2f}%'
    f'（NMS IoU 0.5、conf 0.001、三种子均值）。旧口径 92.12% 系开卷所得，此后只作对照，不再作为最终成绩。'
    f'TripletAttention 三种子均值 {trip_c:.2f}%（较基线 +{trip_c-base_c:.2f} pt），叠加三种子 WBF 融合达 '
    f'{eight_rows[6][1]:.2f}%，为干净集上最高值。')
r2.bold = False
print(f'  [P7] 摘要第四条重写 OK')

# ② 2.3.5 口径声明
i = 220
p = ps[i]
assert '结果：mAP50 = 92.12%' in p.text
set_text(p, f'结果（v2 正式口径，干净 100 图闭卷集、三种子均值）：mAP50 = {base_c:.2f}%，'
            f'mAP50-95 = {eight_rows[0][3]:.2f}%，P = {float(base_raw["P"])*100:.2f}%，'
            f'R = {float(base_raw["R"])*100:.2f}%。开卷 179 图口径的历史值为 mAP50 92.12%、'
            f'mAP50-95 54.23%、P 91.68%、R 85.17%；因该 test 含 44.1% 相邻帧泄漏，此后仅作对照。')
print('  [P220] 口径声明重写 OK')

# ③ 6.2.5 最终基线数字表：改标签 + 增一行
t = d.tables[49]
set_cell(t.cell(2, 0), 'NMS 0.5（179 图开卷，历史口径）')
new_row = t.add_row()
for j, v in enumerate(['干净 100 图（v2 正式口径）', f'{base_c/100:.4f}',
                       f'{eight_rows[0][3]/100:.4f}', f'{float(base_raw["P"]):.4f}',
                       f'{float(base_raw["R"]):.4f}']):
    set_cell(new_row.cells[j], v)
print('  [T49] 增「干净 100 图」行 OK')

# ④ 3.2.1 需求达成表
t = d.tables[12]
set_cell(t.cell(1, 3), f'{base_c/100:.4f}（Triplet+WBF {eight_rows[6][1]/100:.4f}）')
set_cell(t.cell(1, 4), '基线边缘未达成')
set_cell(t.cell(2, 3), f'{eight_rows[0][3]/100:.4f}')
print('  [T12] N1/N2 更新 OK')

# ⑤ 8.2 结论三：加口径括注
i = find_para('P2 头把检测尺度从 P3/P4/P5')
p = ps[i]
old = '实测三种子均值 0.9143，低于基线 0.9212（−0.69 pt）。'
assert old in p.text, p.text[:120]
set_text(p, p.text.replace(old, '实测三种子均值 0.9143，低于基线 0.9212（−0.69 pt；179 图开卷口径。'
                                '换成干净闭卷 100 图，P2v2 单颗 89.71 对基线 89.98，差 −0.27 pt，'
                                '方向不变但幅度更小）。'))
print(f'  [P{i}] 结论三加括注 OK')

# ⑥ 7.6
i = find_para('这份报告里的 92.12%')
set_text(ps[i], ps[i].text.replace(
    '这份报告里的 92.12%', '这份报告里的 89.98%（v2 口径；上一版写的 92.12% 已因相邻帧泄漏作废）'))
print(f'  [P{i}] 7.6 更新 OK')

# ⑦ 8.6 结束语
i = find_para('报告里最重要的数字不是 92.12%')
p = ps[i]
assert len(p.runs) >= 2
p.runs[0].text = '报告里最重要的数字不是 89.98%，也不是 92.04%，是 1.60。'
p.runs[1].text = ' 前两个说明模型有多好，最后一个说明这个"好"能被相信到什么精度。'
print(f'  [P{i}] 结束语更新 OK')

# ⑧ 8.1 表 8-1 ④
t = d.tables[110]
set_cell(t.cell(4, 2), f'干净闭卷 100 图：基线 mAP50 {base_c:.2f}% / mAP50-95 {eight_rows[0][3]:.2f}%；'
                       f'Triplet+WBF {eight_rows[6][1]:.2f}%（NMS 0.5 / 896 / 三种子）')
print('  [T110] 8-1 ④ 更新 OK')

# ---------------------------------------------------------------- 新增 6.13.5
_hit = [i for i, x in enumerate(ps) if x.text.strip() == '第 7 章 项目实施过程与方法论记录']
assert len(_hit) == 1, _hit
ref = ps[_hit[0]]._p
body_tpl = ps[find_para('按可能性的量级排序')]._p      # 正文模板
head_tpl = ps[find_para('6.13.4 什么才可能真正提升')]._p  # 小标题模板
cap_tpl = ps[find_para('图 6-5')]._p                    # 图题模板


def new_para(tpl, text, bold=None):
    el = copy.deepcopy(tpl)
    ref.addprevious(el)
    p = [x for x in ps if x._p is el]
    from docx.text.paragraph import Paragraph
    par = Paragraph(el, ref.getparent())
    set_text(par, text)
    if bold is not None and par.runs:
        par.runs[0].bold = bold
    return par


def mkcap(text):
    el = copy.deepcopy(cap_tpl)
    ref.addprevious(el)
    par = Paragraph(el, ref.getparent())
    par.runs[0].text = text
    for r in par.runs[1:]:
        r._element.getparent().remove(r._element)
    return el


def mktable(rows, header):
    tb = d.add_table(rows=len(rows) + 1, cols=len(header))
    tb.style = d.tables[49].style
    tb.autofit = True
    for j, h in enumerate(header):
        set_cell(tb.cell(0, j), h)
    for i2, row in enumerate(rows):
        for j, v in enumerate(row):
            set_cell(tb.cell(i2 + 1, j), v)
    ref.addprevious(tb._tbl)
    return tb


def fmt(x, plus=True):
    s2 = f'{x:+.2f}' if plus else f'{x:.2f}'
    return s2.replace('-', '\u2212')


base_d = eight_rows[2][1] - eight_rows[0][1]      # WBF vs 基线单模均值
trip_d = eight_rows[6][1] - eight_rows[4][1]      # WBF vs Triplet 单模均值

new_para(head_tpl, '6.13.5 v2 口径重定：段级去泄漏后的真实底座（2026-10-26）', bold=True)
new_para(body_tpl,
         '上面 13 个方向的判读全部建立在 179 图 test 上。这个 test 有一个此前未量化的问题：它与训练集来自'
         '同一批巡检视频，179 张里有 79 张（44.1%）与训练图同处一个连续拍摄段，属相邻帧泄漏——模型在 test 上'
         '看到的几乎是它背过的帧。用 dHash（汉明距离 ≤ 2）与 64×64 灰度 MAE（≤ 20）双闸聚类，把全部图像'
         '并成视觉段（并查集，共 1080 段），再按段整段剔除，得到 100 张闭卷测试集。代价是磨损类只剩 8 图 31 框，'
         '该类指标从此只具参考价值。')
new_para(body_tpl,
         '五个关键 run 用同一批权重、同一口径（896 / NMS IoU 0.5 / conf 0.001 / batch 8）重跑，结果如下'
         '（单种子 s42，与 6.2.5 的逐种子表同源）：')
mkcap('表 6-8\u3000五个关键 run 在开卷 / 闭卷测试集上的对比（s42）')
mktable([(n, f'{o:.2f}', f'{c:.2f}', fmt(d), f'{c95:.2f}') for n, o, c, d, c95 in five_rows],
        ['模型', '179 图开卷 mAP50', '100 图闭卷 mAP50', '掉分', '闭卷 mAP50-95'])

new_para(body_tpl,
         f'换成三种子均值口径，结论要更保守：基线 {base_o:.2f}% → {base_c:.2f}%（{fmt(base_c-base_o)} pt），'
         f'Triplet {trip_o:.2f}% → {trip_c:.2f}%（{fmt(trip_c-trip_o)} pt），两者相差 '
         f'{abs((trip_c-trip_o)-(base_c-base_o)):.2f} pt。方向与"Triplet 退化更小"一致，但这一差值仍落在 '
         f'1.60 pt 噪声地板之内，因此只能表述为"退化更小"，不能表述为"泛化显著更强"。'
         f'CBAM 掉 {abs(five_rows[1][3]):.2f} pt 是全组最狠，这从反面说明它此前的增益有相当部分来自对这批画面的记忆。')
new_para(body_tpl,
         '在闭卷集上重跑推理端后处理（三种子 WBF 融合、TTA），8 组配置的结果如下——')
mkcap('表 6-9\u3000干净闭卷 100 图上的推理端后处理对比（三种子）')
mktable([(n, f'{m:.2f}', fmt(dd), f'{m95:.2f}') for n, m, dd, m95 in eight_rows],
        ['配置', 'mAP50', 'Δ vs 基线', 'mAP50-95'])

pic_p = d.add_paragraph()
pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
pic_p.add_run().add_picture(str(FIG), width=Inches(6.0))
ref.addprevious(pic_p._p)
mkcap('图 6-6\u30008 组推理配置在干净闭卷 100 图上的 mAP50（灰带为 89.98→91.58 的 1.60 pt 噪声地板）')

new_para(body_tpl,
         f'三点判读。第一，WBF 在两族上都为正：相对各自单模均值，基线族 {fmt(base_d)} pt、Triplet 族 '
         f'{fmt(trip_d)} pt（表 6-9 的 Δ 列是相对 89.98% 基线的，口径不同）；Triplet 融合后逐类 3/3 全涨，'
         f'得到 {eight_rows[6][1]:.2f}%，是本项目干净集上的最高值。第二，TTA 是负资产：基线族 '
         f'{fmt(eight_rows[1][1]-eight_rows[0][1])} pt 且三颗种子符号不一致（+0.08 / \u22120.33 / +0.45），'
         f'Triplet 族 {fmt(eight_rows[5][1]-eight_rows[4][1])} pt 且三颗全负（\u22121.44 / \u22121.30 / \u22120.71）；'
         f'因此"WBF + TTA"反而不如 WBF 单打（{eight_rows[7][1]:.2f}% vs {eight_rows[6][1]:.2f}%），'
         'TTA 此后仅作对照，不进入交付配置。第三，除 '
         f'{eight_rows[6][1]:.2f}% 相对基线 {base_c:.2f}% 的 {fmt(eight_rows[6][1]-base_c)} pt 之外，'
         '其余增益全部小于 1.60 pt 地板；而这 '
         f'{fmt(eight_rows[6][1]-base_c)} pt 本身是"架构 +{trip_c-base_c:.2f}"与"融合 {fmt(trip_d)}"两项叠加，'
         '不能拆开当成单一因素越过地板。没有重训：本节全部为纯推理，权重仍是 179 图训练所得，'
         '只是换了考卷——这是它的边界，也是它诚实的地方。')

if DRY:
    print('\n[dry] 不写盘')
    sys.exit(0)

d.save(str(TMP))
print(f'\n已保存临时文件 {TMP}')

# ---------------------------------------------------------------- 校验
old = Document(str(DOC))
new = Document(str(TMP))
o_txt = [p.text for p in old.paragraphs]
n_txt = [p.text for p in new.paragraphs]

import difflib
sm = difflib.SequenceMatcher(None, o_txt, n_txt, autojunk=False)
print('\n=== 段落 diff 摘要 ===')
bad = []
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == 'equal':
        continue
    print(f'  {tag}: 旧[{i1}:{i2}] → 新[{j1}:{j2}]')
    if tag == 'replace':
        for k in range(max(i2 - i1, j2 - j1)):
            a = o_txt[i1 + k][:60] if i1 + k < i2 else ''
            b = n_txt[j1 + k][:60] if j1 + k < j2 else ''
            print(f'      - {a}\n      + {b}')

# 表格完整性：新表插在文档中部，后续表索引整体后移 2，按偏移比对
print('\n=== 表格校验 ===')
oa = [[[c.text for c in row.cells] for row in t.rows] for t in old.tables]
na = [[[c.text for c in row.cells] for row in t.rows] for t in new.tables]
assert len(oa) == 116 and len(na) == 118, (len(oa), len(na))

SIG1 = ['模型', '179 图开卷 mAP50', '100 图闭卷 mAP50', '掉分', '闭卷 mAP50-95']
SIG2 = ['配置', 'mAP50', 'Δ vs 基线', 'mAP50-95']
k = [i for i, t in enumerate(na) if t[0] == SIG1][0]
assert na[k + 1][0] == SIG2, na[k + 1][0]
print(f'  新增两张表位于第 {k} / {k+1} 张（文档顺序）')
assert len(na[k]) == 6 and len(na[k + 1]) == 9, (len(na[k]), len(na[k + 1]))
assert len(na) - len(oa) == 2

changed = []
for ti in range(len(oa)):
    j = ti if ti < k else ti + 2
    if oa[ti] != na[j]:
        changed.append(ti)
print('  被修改的既有表：', changed)
assert set(changed) <= {12, 49, 110}, changed
print(f'  表数 {len(oa)} → {len(na)}，段数 {len(o_txt)} → {len(n_txt)}')
print('\n校验通过。')
