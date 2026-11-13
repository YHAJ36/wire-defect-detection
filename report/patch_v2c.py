# -*- coding: utf-8 -*-
"""v2 同步第三遍：修掉被段级去泄漏实验直接推翻的表述。

不重建文件，逐段 / 逐单元格就地替换；每处先断言唯一匹配。
"""
from docx import Document

SRC = 'D:/2/report/实验报告_全文.docx'
OUT = 'D:/2/report/_tmp_v2c.docx'

d = Document(SRC)
ps = d.paragraphs


def cell(ti, ri, ci, new):
    c = d.tables[ti].cell(ri, ci)
    p = c.paragraphs[0]
    if p.runs:
        p.runs[0].text = new
        for r in p.runs[1:]:
            r._element.getparent().remove(r._element)
    else:
        p.add_run(new)


def para(idx, new=None, old=None, rep=None):
    p = ps[idx]
    if new is not None:
        assert p.runs
        p.runs[0].text = new
        for r in p.runs[1:]:
            r._element.getparent().remove(r._element)
        print(f'  [P{idx}] 整段重写')
    else:
        assert old in p.text, f'P{idx} 不含 {old[:30]!r}\n实际 {p.text[:90]!r}'
        p.runs[0].text = p.text.replace(old, rep)
        for r in p.runs[1:]:
            r._element.getparent().remove(r._element)
        print(f'  [P{idx}] 局部替换')


# ① 需求表 N1：拆成两格，避免 4 行挤压
cell(12, 1, 3, '0.8998')
cell(12, 1, 4, '基线边缘未达成；Triplet+WBF 0.9204')
print('  [T12] N1 拆写')

# ② 6.13.5 里 TTA 那句的双括号
p = [x for x in ps if x.text.startswith('三点判读')][0]
old = '（等同无效）（+0.08 / −0.33 / +0.45）'
assert old in p.text
p.runs[0].text = p.text.replace(old, '（+0.08 / −0.33 / +0.45，等同无效）')
for r in p.runs[1:]:
    r._element.getparent().remove(r._element)
print('  [三点判读] 双括号已合并')

# ③ 1.2.4 六篇对照表：泄漏披露口径
cell(3, 7, 2, '是（段级判定 44.1%）')
print('  [T3] 泄漏披露 → 44.1%')

# ④ 7.6 九项检查：泄漏声明
cell(111, 8, 1, '✅ 44.1%（79/179，段级双闸），并给出闭卷重评')
print('  [T111] 九项检查泄漏行 → 44.1%')

# ⑤ 表 8-1 ① 加口径限定
cell(112, 1, 2, '896 三种子 test mAP50 = 0.9197 / 0.9217 / 0.9221，均值 0.9212（179 图开卷口径）；'
                '同种子重跑差 1.04 pt，"完全可复现"不成立')
print('  [T112] 8-1 ① 补口径限定')

# ⑥ 表 8-2 局限性第 3 条：已部分补齐
cell(116, 3, 2, '段级隔离已补（2026-10-26）：基线 −2.14 pt、CBAM −2.83 pt；闭卷集仅 100 图，'
                '统计功效比 179 图更弱')
cell(116, 3, 3, '179 图指标偏乐观约 2.1 pt；闭卷集的结论方向可用，单点数值仍需谨慎')
print('  [T116] 8-2 #3 已部分补齐')

# ⑦ 结论七：整段重写（原表述已被推翻）
para(1345, new='结论七：数据存在 44.1% 的相邻帧泄漏，按视觉段重划分后基线掉 2.14 pt，'
               '指标只代表"同场景泛化"。')
para(1346, new='test 179 张里有 79 张（44.1%）与训练图同处一个连续拍摄段——这是 dHash（汉明 ≤2）'
               '与 64×64 灰度 MAE（≤20）双闸聚类、把图像并成 1080 个视觉段之后的判定。按段整段剔除，'
               '剩 100 张闭卷集：基线 mAP50 由 92.12% 降到 89.98%，Triplet 由 91.83% 降到 90.79%。'
               '所以"干净子集"的准确含义是"同段隔离且无近复制品"，仍不等于"未见过的新场景"——'
               '全部图像来自同一批巡检视频。')
para(1347, new='更关键的是，泄漏的影响此前"测不出"（置换检验落在随机范围内），换成段级隔离后就被量化了：'
               '基线 −2.14 pt、CBAM −2.83 pt。这说明"测不出"往往不是"没有影响"，而是"隔离粒度不够"。'
               '这一数据工程缺陷已于 2026-10-26 补齐，过程见 6.13.5。')

# ⑧ 8.5 未来工作：段重划分已完成
para(1374, new='为什么"按视频段重划分"曾排第二：它不会提升指标，反而会让指标下降——因为去掉了'
               '近重复带来的乐观偏差。但下降后的数字才是真的。该项已于 2026-10-26 完成（见 6.13.5），'
               '基线因此从 92.12% 下调到 89.98%；它把一个"测不出"的问题变成了可交付的结论，'
               '本报告的 v2 口径即来源于此。')

d.save(OUT)
print('saved', OUT)

# 复核
n = Document(OUT)
print('\n复核：')
print('  段落数', len(n.paragraphs), '表数', len(n.tables), '图数', len(n.inline_shapes))
print('  结论七首句:', n.paragraphs[1345].text[:40])
print('  N1:', n.tables[12].cell(1, 3).text, '|', n.tables[12].cell(1, 4).text)
print('  8-2#3:', n.tables[116].cell(3, 2).text[:40])
