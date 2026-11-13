# -*- coding: utf-8 -*-
"""给 93 页主报告打补丁：补 6 张图 + 修数据集口径 + 加 2 页执行摘要。就地 patch，不重建。"""
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
CONF = 'D:/2/report/_batch2_confusion_896/nf_v11_896_s42'

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('D:/2/_backup', exist_ok=True)
shutil.copy2(P, f'D:/2/_backup/实验报告_全文.docx.{ts}.pre_fig.bak')
print('backup ok', ts)

doc = docx.Document(P)
ps = doc.paragraphs
N0, T0 = len(ps), len(doc.tables)
print('before: paras', N0, 'tables', T0, 'images', len(doc.inline_shapes))

TMP_BODY = ps[316]
TMP_CAP = ps[693]
TMP_H = ps[2]


def _set_text(el, text):
    runs = el.findall(qn('w:r'))
    for r in runs[1:]:
        el.remove(r)
    r = runs[0]
    for t in r.findall(qn('w:t'))[1:]:
        r.remove(t)
    t = r.find(qn('w:t'))
    t.text = text
    t.set(qn('xml:space'), 'preserve')


def clone(tmpl, text, page_break=False):
    el = copy.deepcopy(tmpl._p)
    _set_text(el, text)
    if page_break:
        pPr = el.find(qn('w:pPr'))
        if pPr is None:
            pPr = el.makeelement(qn('w:pPr'), {})
            el.insert(0, pPr)
        pb = pPr.find(qn('w:pageBreakBefore'))
        if pb is None:
            pb = pPr.makeelement(qn('w:pageBreakBefore'), {})
            pPr.insert(0, pb)
    return el


def img_par(path, width_cm, page_break=False):
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
    p.add_run().add_picture(path, width=Cm(width_cm))
    return p._p


def note_par(text):
    el = copy.deepcopy(TMP_CAP._p)
    _set_text(el, text)
    pPr = el.find(qn('w:pPr'))
    if pPr is None:
        pPr = el.makeelement(qn('w:pPr'), {})
        el.insert(0, pPr)
    jc = pPr.find(qn('w:jc'))
    if jc is None:
        jc = pPr.makeelement(qn('w:jc'), {})
        pPr.append(jc)
    jc.set(qn('w:val'), 'left')
    ind = pPr.find(qn('w:ind'))
    if ind is None:
        ind = pPr.makeelement(qn('w:ind'), {})
        pPr.append(ind)
    ind.set(qn('w:left'), '312')
    ind.set(qn('w:right'), '312')
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


def chain(anchor, elems):
    cur = anchor
    for e in elems:
        cur.addnext(e)
        cur = e
    return cur


def add_fig(anchor, items):
    """items: [(kind, payload)]，kind in body/img/cap/note；返回最后一个元素"""
    elems = []
    for kind, payload in items:
        if kind == 'body':
            elems.append(clone(TMP_BODY, payload))
        elif kind == 'cap':
            elems.append(clone(TMP_CAP, payload))
        elif kind == 'note':
            elems.append(note_par(payload))
        elif kind == 'img':
            path, w, pb = payload
            elems.append(img_par(path, w, page_break=pb))
    return chain(anchor, elems)


# ============================================================
# 1) 4.1.1 数据集口径
# ============================================================
p315 = ps[315]
r = p315.runs
r[0].text = ('数据集由指导教师提供。教师提供的原始资料把数据集记为 3040 张有效图像，'
             '并按 8∶1∶1 划分为 2432 / 305 / 303；随数据实际交付、可逐张核验的文件是 1800 张，'
             '剔除 15 张没有任何目标的图像后为 ')
r[1].text = '1785 张'
r[2].text = ('（训练 1428 / 验证 178 / 测试 179）。这 1785 张是 SAM 辅助生成多边形掩膜、'
             '转成矩形框并人工复核后的结果。两组数字并不矛盾，而是两种口径：3040 ÷ 1785 = 1.703，'
             '且三份的比值分别为 1.703 / 1.713 / 1.693，比例完全一致，说明 3040 是按同一划分比例'
             '放大后的计数——对应教师资料中"数据增强扩充训练样本"这一步；本项目的增强（mosaic、'
             '旋转、色彩抖动等）全部在训练时在线完成、不落盘，所以磁盘上只有增强前的 1785 张。'
             '本报告全部实验、图表与指标，一律以 1785 张为准。')
print('4.1.1 rewritten, new len', len(p315.text))

tb = doc.tables[5]
done = 0
for row in tb.rows:
    for c in row.cells:
        if '共 1800 张' in c.text:
            for pp in c.paragraphs:
                if '1800' in pp.text and pp.runs:
                    pp.runs[0].text = ('数据集由教师提供；可核验的实文件 1800 张，剔除 15 张无目标图后为 '
                                       '1785 张（教师资料另记 3040 张，为增强后口径，见 4.1.1）')
                    for rr in pp.runs[1:]:
                        rr.text = ''
                    done += 1
print('table cell patched', done)

# ============================================================
# 2) 图 4-1  三类缺陷样例（4.1.3 末）
# ============================================================
add_fig(ps[336]._p, [
    ('body', '三类缺陷在图像中的实际形态如图 4-1。每类各取三张，左半为原图与人工标注框，'
             '右半为框内区域放大——这样才看得出模型真正要分辨的是什么。'),
    ('img', (f'{FIG}/fig1_samples_gt.jpg', 14.6, False)),
    ('cap', '图 4-1　三类缺陷在 test 集上的典型形态与人工标注框（左：原图与标框；右：框内区域放大）'),
    ('note', '断股是散开的钢丝，标注边界没有客观答案；雷击是表面灼伤，边界相对明确；'
             '磨损是成片区域，一张图常含多个框。三类缺陷的"边界确定性"不同，'
             '这正是 6.9 节量化出的标注噪声的来源。'),
])

# ============================================================
# 3) 图 4-2  letterbox 灰边（4.2.3 末）
# ============================================================
add_fig(ps[370]._p, [
    ('img', (f'{FIG}/fig_letterbox.png', 15.0, False)),
    ('cap', '图 4-2　1920×1080 的图像 letterbox 到 896×896 之后：有效画面只占 56.25%，灰边占 43.75%'),
    ('note', '灰边是像素值恒为 114 的纯色块，但它照常参与卷积与 BN 统计；按面积算，'
             '每个 batch 有 43.75% 的输入像素是同一个常数。这就是"letterbox 的代价不是算力、'
             '是像素"的直观含义。'),
])

# ============================================================
# 4) 图 6-1 训练曲线 + 图 6-2 PR 曲线（6.2.5 末）
# ============================================================
last = add_fig(ps[724]._p, [
    ('body', '基线（nf_v11_896_s42）的训练曲线、验证曲线与测试集 PR 曲线如下。'),
    ('img', (f'{FIG}/fig2_train_curves.png', 15.0, False)),
    ('cap', '图 6-1　基线（YOLOv11s / 896 / 60 轮 / seed 42）的训练与验证曲线'),
    ('note', '60 轮结束时 val/cls_loss 仍在缓慢下降，但 val/box_loss 与 mAP 已基本平走；'
             '6.7 节据此判断"已经过峰"，并用 150 轮对照验证了这个判断。'),
])
last = add_fig(last, [
    ('img', (f'{FIG}/fig3_pr_curve.png', 12.6, False)),
    ('cap', '图 6-2　基线在测试集上的 P-R 曲线（break 0.866 / thunderbolt 0.945 / wear 0.909，三类均值 0.907）'),
    ('note', '这是 val 口径的 AP（NMS IoU 0.7）；主表用的是 test 口径、NMS IoU 0.5，'
             '两者不可直接比较，图仅用于看曲线的形状——断股类的曲线在召回 0.7 之后最先塌下去。'),
])

# ============================================================
# 5) 图 6-3 混淆矩阵（6.3.5 末）
# ============================================================
add_fig(ps[774]._p, [
    ('body', '作为参照，图 6-3 给出基线模型在测试集上的归一化混淆矩阵。'
             '注意它说明的是"错误是怎么分布的"，而不是"加了注意力能改善哪个格子"——'
             '因为 6.3.2 已经证明这两种注意力都没能越过噪声地板。'),
    ('img', (f'{CONF}/confusion_matrix_normalized.png', 12.4, False)),
    ('cap', '图 6-3　基线模型的归一化混淆矩阵（列归一化：每列表示该类真实框被判成各类的比例）'),
    ('note', '断股列：0.83 正确、0.15 被判成背景（漏检）；雷击列 0.91 正确；磨损列 0.84 正确、'
             '0.16 判成背景。末列 background 是把背景区域判成各类的误报分布——'
             '断股类在这一列的 0.71 说明它同时也是"多报"最多的类，与 6.8.4 节"冗余框太多"'
             '的观察一致。'),
])

# ============================================================
# 6) 图 6-4 相邻帧标注差异（6.9.3 末）
# ============================================================
add_fig(ps[1016]._p, [
    ('body', '图 6-4 是这条结论的直观版本：训练集里编号相邻的 1633 与 1634 两张图，'
             '是同一段视频的连续帧，画面里是同一处断股。两帧的标注框尺寸只差 2%，'
             '但中心水平偏了 15.5 px——相当于框宽的三分之一。'),
    ('img', (f'{FIG}/fig7_adjacent_label.png', 13.6, False)),
    ('cap', '图 6-4　同一个断股目标在相邻两帧上的标注框：跨帧 IoU 仅 0.478'),
    ('note', '两帧之间隔不到 0.1 秒，物理形变可以忽略；框没有变大变小，是整个人平移了。'
             '这不是个别现象：全库 1131 对相邻帧、同一目标的跨帧 IoU 中位数是 0.477，'
             'p10 只有 0.328——即使模型完美复现某一帧的标注，在相邻帧上也够不着 IoU 0.5。'),
])

# ============================================================
# 7) 图 6-5 效应量（6.13.1 总表之后）
# ============================================================
add_fig(ps[1140]._p, [
    ('img', (f'{FIG}/fig5_effect_sizes.png', 15.0, False)),
    ('cap', '图 6-5　13 个改进方向相对基线的 mAP50 效应量（灰带为 ±1.60 pt 噪声地板）'),
    ('note', '只有 mosaic 削弱一条越出灰带，方向为负。其余 12 条全部落在带内——'
             '不是"没有效果"，而是"效果小于这把尺子的刻度"。'),
])

doc.save(P)
print('saved')

# ---------- 复核 ----------
d2 = docx.Document(P)
print('after: paras', len(d2.paragraphs), 'tables', len(d2.tables),
      'images', len(d2.inline_shapes))
assert len(d2.tables) == T0, '表格数变化了'
