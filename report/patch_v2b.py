# -*- coding: utf-8 -*-
"""v2 补丁的第二遍：修为排版细节（图宽、表格列宽、图题孤行、一句措辞）。

就地改，不重建；改完输出到 _tmp_v2b.docx，由外层确认后再覆盖。
"""
from docx import Document
from docx.shared import Inches, Cm

SRC = 'D:/2/report/实验报告_全文.docx'
OUT = 'D:/2/report/_tmp_v2b.docx'

d = Document(SRC)
ps = d.paragraphs


def set_para(prefix, new=None, old=None, rep=None):
    hit = [p for p in ps if p.text.strip().startswith(prefix)]
    assert len(hit) == 1, (prefix, len(hit))
    p = hit[0]
    if new is not None:
        p.runs[0].text = new
        for r in p.runs[1:]:
            r._element.getparent().remove(r._element)
    elif rep:
        assert old in p.text, p.text[:80]
        p.runs[0].text = p.text.replace(old, rep)
        for r in p.runs[1:]:
            r._element.getparent().remove(r._element)
    return p


# ① 图题缩短，避免"板）"孤行
set_para('图 6-6', new='图 6-6\u30008 组推理配置在干净闭卷 100 图上的 mAP50（灰带为 1.60 pt 噪声地板）')

# ② TTA 那句措辞：+0.07 说成"负资产"读起来矛盾
set_para('三点判读', old='基线族 +0.07 pt 且三颗种子符号不一致',
         rep='基线族仅 +0.07 pt、三颗种子涨跌互现（等同无效）')

# ③ 图宽收到版心内（版心 5.906 in，原 6.0 越界）
shapes = d.inline_shapes
assert len(shapes) == 10
shapes[-1].width = Inches(5.85)

# ④ 两张新表列宽
t8, t9 = d.tables[103], d.tables[104]
for tbl, widths in ((t8, (4.5, 2.9, 2.9, 2.0, 2.7)), (t9, (4.4, 3.0, 3.6, 4.0))):
    tbl.autofit = False
    for j, w in enumerate(widths):
        tbl.columns[j].width = Cm(w)
    for row in tbl.rows:
        for j, w in enumerate(widths):
            row.cells[j].width = Cm(w)

d.save(OUT)
print('saved', OUT)
print('  T103', [round(c.width.cm, 2) for c in d.tables[103].columns])
print('  T104', [round(c.width.cm, 2) for c in d.tables[104].columns])
print('  pic', round(shapes[-1].width.inches, 3), 'in')
