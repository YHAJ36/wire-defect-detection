# -*- coding: utf-8 -*-
"""主报告补丁第三步：正文引用图 4-2 + 表 7-2 补一条口径订正"""
import copy
import docx
from docx.oxml.ns import qn

P = 'D:/2/report/实验报告_全文.docx'
doc = docx.Document(P)

# ---------- 1) 4.2.3 正文引用图 4-2 ----------
hit = 0
for p in doc.paragraphs:
    for r in p.runs:
        if '宽图会有 43.75% 的面积是灰边。' in r.text:
            r.text = r.text.replace('宽图会有 43.75% 的面积是灰边。',
                                    '宽图会有 43.75% 的面积是灰边（示意见图 4-2）。')
            hit += 1
print('图 4-2 引用插入', hit)

# ---------- 2) 表 7-2 追加一行 ----------
tb = doc.tables[106]
before = len(tb.rows)
src = tb.rows[-1]
newtr = copy.deepcopy(src._tr)
src._tr.addnext(newtr)
row = tb.rows[-1]

vals = ['10-26',
        '4.1.1 与 3.1.2 表：数据集规模原先只写"1800 张 → 1785 张"，与教师资料记的 3040 张并存不解释',
        '教师资料记 3040 张（2432 / 305 / 303），实际交付 1800 张。两者比值 1.703，'
        '三份分别为 1.703 / 1.713 / 1.693，比例完全一致——3040 是增强后口径',
        '改为两种口径并列写明：3040 为增强后计数，1785 为可逐张核验的实文件；'
        '全文指标一律以 1785 为准']

for c, v in zip(row.cells, vals):
    ps_ = c.paragraphs
    for k, pp in enumerate(ps_):
        if k == 0:
            if pp.runs:
                pp.runs[0].text = v
                for r in pp.runs[1:]:
                    r.text = ''
            else:
                pp.add_run(v)
        else:
            for r in pp.runs:
                r.text = ''
print('表 7-2 行数', before, '->', len(tb.rows))

doc.save(P)

# ---------- 复核 ----------
d2 = docx.Document(P)
print('paras', len(d2.paragraphs), 'tables', len(d2.tables), 'images', len(d2.inline_shapes))
print('表 7-2 末行:', ' | '.join(x.text.strip()[:34] for x in d2.tables[106].rows[-1].cells))
