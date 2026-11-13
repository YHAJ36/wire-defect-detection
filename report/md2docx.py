# -*- coding: utf-8 -*-
"""
把 report/实验报告_全文.md 转成课程设计报告格式的 .docx
样式参照 D:/3-DroneVision-YOLOv11/report 里的成品：
  章标题 14pt 黑体加粗，段前后 21pt，居中
  节标题 12pt 黑体加粗，段前后 6pt
  正文   12pt 宋体，西文 Times New Roman，行距 1.5，首行缩进 2 字符
  表标题 10.5pt，表体 9pt
无封面、无目录（用户指定）。
"""
import io
import re
import sys

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SRC = 'D:/2/report/实验报告_全文.md'
OUT = 'D:/2/report/实验报告_全文.docx'

CN_BODY = '宋体'
CN_HEAD = '黑体'
EN_BODY = 'Times New Roman'
EN_CODE = 'Consolas'


# ---------- 基础工具 ----------

def set_run_font(run, cn=CN_BODY, en=EN_BODY, size=12, bold=False, color=None):
    run.font.name = en
    run.font.size = Pt(size)
    run.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.append(rfonts)
    rfonts.set(qn('w:ascii'), en)
    rfonts.set(qn('w:hAnsi'), en)
    rfonts.set(qn('w:eastAsia'), cn)
    if color is not None:
        run.font.color.rgb = color


def shade(el, hexcolor):
    """给段落或单元格加底纹"""
    pr = el._element.get_or_add_tcPr() if hasattr(el._element, 'get_or_add_tcPr') else el._element.get_or_add_pPr()
    sh = OxmlElement('w:shd')
    sh.set(qn('w:val'), 'clear')
    sh.set(qn('w:color'), 'auto')
    sh.set(qn('w:fill'), hexcolor)
    pr.append(sh)


INLINE = re.compile(r'(\*\*.+?\*\*|`[^`]+`)')


def add_inline(par, text, size=12, cn=CN_BODY, en=EN_BODY, base_bold=False):
    """把 **粗体** 和 `代码` 解析成多个 run"""
    for seg in INLINE.split(text):
        if not seg:
            continue
        if seg.startswith('**') and seg.endswith('**') and len(seg) > 4:
            r = par.add_run(seg[2:-2])
            set_run_font(r, cn=cn, en=en, size=size, bold=True)
        elif seg.startswith('`') and seg.endswith('`') and len(seg) > 2:
            r = par.add_run(seg[1:-1])
            set_run_font(r, cn=CN_BODY, en=EN_CODE, size=size - 1, bold=base_bold)
            r.font.color.rgb = RGBColor(0xA0, 0x30, 0x30)
        else:
            r = par.add_run(seg)
            set_run_font(r, cn=cn, en=en, size=size, bold=base_bold)


# ---------- 段落构造 ----------

def para(doc, text, *, size=12, cn=CN_BODY, en=EN_BODY, bold=False,
         align=None, indent=True, before=0, after=0, spacing=1.5,
         left=0, keep_with_next=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    if align is not None:
        p.alignment = align
    if indent:
        pf.first_line_indent = Pt(24)
        # 2 字符 = 24pt（12pt 字号的 2 倍）
    if left:
        pf.left_indent = Pt(left)
    pf.line_spacing = spacing
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if keep_with_next:
        pf.keep_with_next = True
    add_inline(p, text, size=size, cn=cn, en=en, base_bold=bold)
    return p


def chapter(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(21)
    pf.space_after = Pt(21)
    pf.line_spacing = 1.5
    pf.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, cn=CN_HEAD, en=EN_BODY, size=14, bold=True)
    return p


def section(doc, text, level=2):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(12 if level == 2 else 6)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.5
    pf.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, cn=CN_HEAD, en=EN_BODY, size=12, bold=True)
    return p


def code_block(doc, lines):
    for i, ln in enumerate(lines):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = 1.0
        pf.space_before = Pt(6 if i == 0 else 0)
        pf.space_after = Pt(6 if i == len(lines) - 1 else 0)
        pf.left_indent = Pt(24)
        r = p.add_run(ln if ln else ' ')
        set_run_font(r, cn=CN_BODY, en=EN_CODE, size=9)
        shade(p, 'F4F4F4')


def cjk_len(s):
    """中文按 2 个宽度计，便于估算列宽"""
    return sum(2 if '\u4e00' <= c <= '\u9fff' else 1 for c in s)


def col_widths(rows, total_cm=15.0):
    n = max(len(r) for r in rows)
    w = []
    for j in range(n):
        m = max((cjk_len(r[j]) if j < len(r) else 0) for r in rows)
        w.append(max(m, 4) ** 0.5)
    s = sum(w)
    return [max(1.4, total_cm * x / s) for x in w]


def add_table(doc, rows):
    """rows: 已切分的单元格文本二维数组（第一行为表头）"""
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    # 固定布局，否则 Word 会自己重排列宽
    layout = OxmlElement('w:tblLayout')
    layout.set(qn('w:type'), 'fixed')
    t._tbl.tblPr.append(layout)

    widths = col_widths(rows)
    for j, cm in enumerate(widths):
        t.columns[j].width = Cm(cm)

    for i, row in enumerate(rows):
        # 单行不跨页断开
        trPr = t.rows[i]._tr.get_or_add_trPr()
        cs = OxmlElement('w:cantSplit')
        trPr.append(cs)
        if i == 0:
            th = OxmlElement('w:tblHeader')
            trPr.append(th)
        for j in range(ncol):
            cell = t.cell(i, j)
            cell.width = Cm(widths[j])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            txt = row[j] if j < len(row) else ''
            cell.text = ''
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            add_inline(p, txt, size=9, base_bold=(i == 0))
            if i == 0:
                shade(cell, 'EDEDED')
    # 表后留一点空
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(0)
    sp.paragraph_format.line_spacing = 1.0
    r = sp.add_run('')
    set_run_font(r, size=6)


def split_row(line):
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    return [c.strip() for c in s.split('|')]


def is_sep(line):
    s = line.strip()
    return bool(re.fullmatch(r'\|?[\s:\-|]+\|?', s)) and '-' in s


def add_page_number(section_obj):
    footer = section_obj.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    set_run_font(r, size=9)
    fld1 = OxmlElement('w:fldChar')
    fld1.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText')
    instr.set(qn('xml:space'), 'preserve')
    instr.text = ' PAGE '
    fld2 = OxmlElement('w:fldChar')
    fld2.set(qn('w:fldCharType'), 'end')
    r._r.append(fld1)
    r._r.append(instr)
    r._r.append(fld2)


# ---------- 主流程 ----------

def main():
    src = io.open(SRC, encoding='utf-8').read().split('\n')
    doc = Document()

    sec = doc.sections[0]
    sec.page_height = Cm(29.7)
    sec.page_width = Cm(21.0)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(3.0)
    sec.right_margin = Cm(3.0)
    add_page_number(sec)

    # 默认样式
    st = doc.styles['Normal']
    st.font.name = EN_BODY
    st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), CN_BODY)
    # 关闭"中英文之间自动加空格"，否则 320×320 之类会被拆开
    st_pPr = st.element.get_or_add_pPr()
    for tag in ('w:autoSpaceDE', 'w:autoSpaceDN'):
        el = OxmlElement(tag)
        el.set(qn('w:val'), '0')
        st_pPr.append(el)

    i = 0
    n = len(src)
    title_done = False
    while i < n:
        raw = src[i]
        line = raw.rstrip()
        s = line.strip()

        # 空行
        if not s:
            i += 1
            continue

        # 分隔线
        if s == '---':
            i += 1
            continue

        # 代码块
        if s.startswith('```'):
            i += 1
            buf = []
            while i < n and not src[i].strip().startswith('```'):
                buf.append(src[i].rstrip())
                i += 1
            i += 1
            code_block(doc, buf)
            continue

        # 表格
        if s.startswith('|'):
            rows = []
            while i < n and src[i].strip().startswith('|'):
                ln = src[i].strip()
                if not is_sep(ln):
                    rows.append(split_row(ln))
                i += 1
            if rows:
                add_table(doc, rows)
            continue

        # 标题
        if s.startswith('# '):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf = p.paragraph_format
            pf.space_before = Pt(36)
            pf.space_after = Pt(6)
            r = p.add_run(s[2:].strip())
            set_run_font(r, cn=CN_HEAD, en=EN_BODY, size=22, bold=True)
            title_done = True
            i += 1
            continue

        if s.startswith('## '):
            txt = s[3:].strip()
            if txt.startswith('——') or txt.startswith('——'):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                pf = p.paragraph_format
                pf.space_before = Pt(0)
                pf.space_after = Pt(24)
                r = p.add_run(txt)
                set_run_font(r, cn=CN_BODY, en=EN_BODY, size=14, bold=False)
            else:
                chapter(doc, txt)
            i += 1
            continue

        if s.startswith('#### '):
            section(doc, s[5:].strip(), level=3)
            i += 1
            continue

        if s.startswith('### '):
            section(doc, s[4:].strip(), level=2)
            i += 1
            continue

        # 列表项
        m = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)$', raw)
        if m:
            lead = '　' * (len(m.group(1)) // 2)
            marker = '· ' if m.group(2) in ('-', '*') else m.group(2) + ' '
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.line_spacing = 1.5
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
            pf.left_indent = Pt(24)
            pf.first_line_indent = Pt(-12)
            add_inline(p, lead + marker + m.group(3), size=12)
            i += 1
            continue

        # 引用块
        if s.startswith('>'):
            txt = s.lstrip('>').strip()
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.line_spacing = 1.5
            pf.left_indent = Pt(24)
            pf.space_before = Pt(6)
            pf.space_after = Pt(6)
            add_inline(p, txt, size=10.5)
            for r in p.runs:
                r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            i += 1
            continue

        # 表题 / 图题：10.5pt 居中，与后面的表格绑定
        if re.match(r'^\*\*表\s*[\d]', s) or re.match(r'^表\s*\d+[-—]\d+', s):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf = p.paragraph_format
            pf.space_before = Pt(8)
            pf.space_after = Pt(3)
            pf.line_spacing = 1.0
            pf.keep_with_next = True
            add_inline(p, s, size=10.5)
            i += 1
            continue

        # 正文段落
        para(doc, s)
        i += 1

    doc.save(OUT)
    print('saved:', OUT)
    print('paragraphs:', len(doc.paragraphs), 'tables:', len(doc.tables))


if __name__ == '__main__':
    main()
