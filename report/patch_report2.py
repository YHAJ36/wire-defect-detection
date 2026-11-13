# -*- coding: utf-8 -*-
"""主报告补丁第二步：加 2 页执行摘要 + 图 4-2 正文引用 + 表 7-2 补一条订正"""
import copy
import os
import shutil
import datetime
import docx
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

P = 'D:/2/report/实验报告_全文.docx'
FIG = 'D:/2/report/_summary_figs'

doc = docx.Document(P)
ps = doc.paragraphs
print('before paras', len(ps), 'images', len(doc.inline_shapes))


def find_par(pred, start=0):
    for i in range(start, len(ps)):
        if pred(ps[i].text.strip()):
            return i
    raise RuntimeError('not found')


TMP_BODY = ps[find_par(lambda t: t.startswith('数据集附带的项目说明'))]
TMP_RICH = ps[find_par(lambda t: t.startswith('第一，断股是主类'))]      # 加粗引导 + 正文，2 个 run
TMP_CAP = ps[find_par(lambda t: t.startswith('表 6-2'))]
TMP_H = ps[find_par(lambda t: t.startswith('第 1 章'))]


def _set(el, text):
    runs = el.findall(qn('w:r'))
    for r in runs[1:]:
        el.remove(r)
    r = runs[0]
    for t in r.findall(qn('w:t'))[1:]:
        r.remove(t)
    t = r.find(qn('w:t'))
    t.text = text
    t.set(qn('xml:space'), 'preserve')


def _pb(el):
    pPr = el.find(qn('w:pPr'))
    if pPr is None:
        pPr = el.makeelement(qn('w:pPr'), {})
        el.insert(0, pPr)
    if pPr.find(qn('w:pageBreakBefore')) is None:
        pPr.insert(0, pPr.makeelement(qn('w:pageBreakBefore'), {}))


def body(text, page_break=False):
    el = copy.deepcopy(TMP_BODY._p)
    _set(el, text)
    if page_break:
        _pb(el)
    return el


def rich(lead, text, page_break=False):
    el = copy.deepcopy(TMP_RICH._p)
    runs = el.findall(qn('w:r'))
    runs[0].find(qn('w:t')).text = lead
    runs[1].find(qn('w:t')).text = text
    for r in runs[2:]:
        el.remove(r)
    if page_break:
        _pb(el)
    return el


def cap(text):
    el = copy.deepcopy(TMP_CAP._p)
    _set(el, text)
    return el


def note(text):
    el = copy.deepcopy(TMP_CAP._p)
    _set(el, text)
    pPr = el.find(qn('w:pPr'))
    jc = pPr.find(qn('w:jc'))
    if jc is None:
        jc = pPr.makeelement(qn('w:jc'), {})
        pPr.append(jc)
    jc.set(qn('w:val'), 'left')
    ind = pPr.makeelement(qn('w:ind'), {'{http://schemas.openxmlformats.org/wordprocessingml/2006/main}left': '312'})
    pPr.append(ind)
    for r in el.findall(qn('w:r')):
        rPr = r.find(qn('w:rPr'))
        if rPr is None:
            rPr = r.makeelement(qn('w:rPr'), {})
            r.insert(0, rPr)
        if rPr.find(qn('w:b')) is not None:
            rPr.remove(rPr.find(qn('w:b')))
        for tag in ('w:sz', 'w:szCs'):
            e = rPr.find(qn(tag))
            if e is None:
                e = rPr.makeelement(qn(tag), {})
                rPr.append(e)
            e.set(qn('w:val'), '18')
    return el


def img(path, width, page_break=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.first_line_indent = Cm(0)
    pf.left_indent = Cm(0)
    pf.right_indent = Cm(0)
    pf.space_before = Pt(8)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.0
    if page_break:
        pf.page_break_before = True
    p.add_run().add_picture(path, width=Cm(width))
    return p._p


def chain(anchor, elems):
    cur = anchor
    for e in elems:
        cur.addnext(e)
        cur = e
    return cur


# ============================================================
# A) 执行摘要（插在副标题之后）
# ============================================================
sub_idx = find_par(lambda t: t.startswith('——基于 YOLOv11s'))
anchor = ps[sub_idx]._p

head = copy.deepcopy(TMP_H._p)
_set(head, '执行摘要')
_pb(head)

elems = [head,
         body('这份报告的全部工作可以用一句话概括：在给定的这份数据上，模型侧的改进已经做完了；'
              '真正卡住的不是模型，是标注。'),
         rich('一、13 个方向，没有一个真正超过基线。',
              '结构类（CBAM、TripletAttention、P2 检测头 v1/v2）、损失类（WIoU v3）、'
              '数据类（mosaic 削弱、close_mosaic 延长、磨损过采样）、代际类（YOLO26s）、'
              '规模类（1024 输入、60→150 轮、三种子 WBF 融合、类级模型选择），逐项做单变量对照，'
              '其余超参逐项一致。除 mosaic 削弱（−2.92 pt）之外，全部落在噪声以内。'),
         rich('二、尺子的最小刻度比被测物还大。',
              '三种子在 357 图合并集上的 mAP50 跨度是 1.60 pt，这是本项目的噪声地板；'
              '而绝大多数改进的真实效应只有 0.2–0.9 pt。用刻度 1.6 的尺子去量 0.5 的差，'
              '量不出结论是必然结果，不是操作失误。'),
         rich('三、真正的瓶颈在数据侧。',
              '相邻帧上同一个断股目标，标注框的跨帧 IoU 中位数只有 0.477；'
              '标注框宽度的抖动标准差是 13.8%，而模型输出框是 14.65%——模型只比标注本身"不准"6%。'
              '它已经学到了标注者的精度上限。'),
         rich('四、最终基线数字。',
              'mAP50 = 92.12%，mAP50-95 = 54.23%（179 图 test、NMS IoU 0.5、conf 0.001、'
              '三种子均值）。三类 AP50 以种子 s42 为例：断股 0.847、雷击 0.980、磨损 0.932。'),
         body('瓶颈分三层，越往下越硬：'),
         rich('第一层，测量精度。',
              'n=3 的符号检验，p 值最小只能是 0.25，永远达不到显著；要做到 p<0.05 需要 6 个种子全部同号。'
              '这是统计设计的极限，不是实验质量的问题。'),
         rich('第二层，靶子在晃。',
              '标注自身的抖动 ±13.8%，模型输出 ±14.65%。此时继续调结构，等于在噪声里捞信号。'),
         rich('第三层，数据本身。',
              '1785 张图、3010 个目标、35 种分辨率、26.8% 的近重复。这个规模撑不起 1 pt 以下的判别。'),
         body('所以本报告的结论不是"我们试了很多方法都没用"，而是"我们证明了这条路的上限在哪里、'
              '以及上限由什么决定"。前者的下一步是"再换一个模块"，后者的下一步是"先修标注"。'),
         body('下面两张图是上述结论的直接证据。', page_break=True),
         img(f'{FIG}/fig7_adjacent_label.png', 13.4),
         cap('摘要图 1　同一个断股目标在相邻两帧上的标注框：跨帧 IoU 仅 0.478（详见 6.9.3 节图 6-4）'),
         note('两帧隔不到 0.1 秒，框没有变大变小，是整个人平移了：中心水平偏了 15.5 px，'
              '相当于框宽的三分之一。'),
         img(f'{FIG}/fig5_effect_sizes.png', 15.0),
         cap('摘要图 2　13 个改进方向相对基线的效应量：只有 mosaic 削弱越出 ±1.60 pt 噪声带，'
             '方向为负（详见 6.13.1 节图 6-5）'),
         note('灰带就是噪声地板。带内的 12 条不是"没有效果"，而是"效果小于这把尺子的刻度"。')]

chain(anchor, elems)
print('执行摘要 inserted,', len(elems), '个元素')

# 第 1 章换页
for p in doc.paragraphs:
    if p.text.strip().startswith('第 1 章'):
        p.paragraph_format.page_break_before = True
        print('第 1 章 page_break set')
        break

doc.save(P)
print('saved step A')
