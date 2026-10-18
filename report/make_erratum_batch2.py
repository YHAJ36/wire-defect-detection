# -*- coding: utf-8 -*-
"""勘误：第二批 CBAM 修复实验 @896 探索报告 v1 —— §七 判据读法更正

输出：D:/2/report/第二批CBAM修复实验_896_探索报告_v1_勘误_v1.pdf

════════════════════════════════════════════════════════════════════════════
【硬约束】本脚本**绝不 import make_report_v5 / make_report_batch2 / make_report_batch3**
  三者都是脚本不是模块（没有 __main__ 保护，import 即执行、直接重生成各自 PDF）。
  Doc 容器 / 字体 / wrap_cjk 仍是**复制**，版面改动只改本文件。
  本勘误**不重生成** batch2 v1 PDF，也不改动其任何一页。
════════════════════════════════════════════════════════════════════════════

数据来源（现读，不手抄）：
  · 640 地板 / CBAM+BN 双臂 ← exp_upscale/logs/batch2_eval.tsv
  · 判据原文               ← WORKLOG A.9.19（预登记）、A.9.23 §七（更正）

用法：
  D:/yolo-env/Scripts/python.exe D:/2/report/make_erratum_batch2.py
  REPORT_PNG=D:/2/report/_preview_erratum_batch2 ... make_erratum_batch2.py
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

OUT = 'D:/2/report/第二批CBAM修复实验_896_探索报告_v1_勘误_v1.pdf'
assert 'v_final' not in OUT, '不允许把本脚本的输出指向 v_final'
assert OUT.endswith('_v1.pdf')

ROOT = 'D:/2'
B2 = f'{ROOT}/exp_upscale/logs/batch2_eval.tsv'
DATE = '2026-10-17'

# ================================================================
#                        字体（照抄已交付报告）
# ================================================================
F_REG = r'C:\Windows\Fonts\msyh.ttc'
F_BOLD = r'C:\Windows\Fonts\msyhbd.ttc'
font_manager.fontManager.addfont(F_REG)
font_manager.fontManager.addfont(F_BOLD)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fp_title = FontProperties(fname=F_BOLD, size=16)
fp_sub = FontProperties(fname=F_REG, size=10)
fp_h1 = FontProperties(fname=F_BOLD, size=12.5)
fp_h2 = FontProperties(fname=F_BOLD, size=10.5)
fp_body = FontProperties(fname=F_REG, size=9.5)
fp_note = FontProperties(fname=F_REG, size=8.5)
fp_th = FontProperties(fname=F_BOLD, size=9)
fp_td = FontProperties(fname=F_REG, size=9)

PAGE_W, PAGE_H = 8.27, 11.69
TOP, BOTTOM = 0.945, 0.055
X0, WIDTH = 0.10, 0.80
LINE = 0.0175
ROW = 0.026
DEBUG = bool(os.environ.get('REPORT_DEBUG'))


class Doc:
    def __init__(self):
        self.pages = []
        self.fig = None
        self._new_page()

    def _new_page(self):
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        self.ax = ax
        self.y = TOP
        self.pages.append(self.fig)

    def _ensure(self, h):
        if self.y - h < BOTTOM:
            self._new_page()
            return True
        return False

    def title(self, text):
        self.ax.text(0.5, self.y, text, ha='center', va='top',
                     fontproperties=fp_title, transform=self.ax.transAxes)
        self.y -= 0.036

    def subtitle(self, text):
        for ln in wrap_cjk(text, 72):
            self.ax.text(0.5, self.y, ln, ha='center', va='top',
                         fontproperties=fp_sub, color='#555555',
                         transform=self.ax.transAxes)
            self.y -= 0.026

    def h1(self, text):
        self._ensure(0.060)
        self.y -= 0.008
        self.ax.text(X0, self.y, text, ha='left', va='top',
                     fontproperties=fp_h1, transform=self.ax.transAxes)
        self.y -= 0.028

    def para(self, text, indent=True):
        lines = wrap_cjk(text, 88)
        for i, ln in enumerate(lines):
            self._ensure(LINE * 1.2)
            s = ('　　' + ln) if (indent and i == 0) else ln
            self.ax.text(X0 + (0.018 if (indent and i > 0) else 0), self.y, s,
                         ha='left', va='top', fontproperties=fp_body,
                         transform=self.ax.transAxes)
            self.y -= LINE
        self.y -= 0.003

    def bullet(self, text, mark='·'):
        lines = wrap_cjk(text, 84)
        for i, ln in enumerate(lines):
            self._ensure(LINE * 1.2)
            self.ax.text(X0 + 0.014, self.y, mark if i == 0 else ' ',
                         ha='left', va='top', fontproperties=fp_body,
                         transform=self.ax.transAxes)
            self.ax.text(X0 + 0.034, self.y, ln, ha='left', va='top',
                         fontproperties=fp_body, transform=self.ax.transAxes)
            self.y -= LINE
        self.y -= 0.002

    def table(self, col_labels, cell_text, col_widths=None, height=None, note=None):
        h = height or (len(cell_text) + 1) * ROW
        pad = 0.06
        if DEBUG:
            print(f'  [table {len(cell_text)}r] y={self.y:.3f} need={h + pad:.3f} '
                  f'{"-> new page" if self.y - (h + pad) < BOTTOM else "ok"}')
        self._ensure(h + pad)
        y0 = self.y - h
        tbl = self.ax.table(cellText=cell_text, colLabels=col_labels,
                            bbox=(X0, y0, WIDTH, h), cellLoc='center',
                            colWidths=col_widths)
        tbl.auto_set_font_size(False)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor('#333333')
            cell.set_linewidth(0.7)
            if r == 0:
                cell.set_facecolor('#dce6f1')
                cell.get_text().set_fontproperties(fp_th)
            else:
                cell.get_text().set_fontproperties(fp_td)
                if r % 2 == 0:
                    cell.set_facecolor('#f2f6fb')
        self.y = y0 - 0.012
        if note:
            self.note(note)

    def note(self, text):
        for ln in wrap_cjk(text, 92):
            self._ensure(LINE)
            self.ax.text(X0, self.y, ln, ha='left', va='top',
                         fontproperties=fp_note, color='#444444',
                         transform=self.ax.transAxes)
            self.y -= LINE * 0.92
        self.y -= 0.006


PUNCT_NO_START = '，。；：、）》」”％…！？,.;:)]}%'
PUNCT_NO_END = '（《「“‘([{'


def _tokenize(text):
    tokens, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ord(ch) > 0x2E80:
            tokens.append((ch, 2))
            i += 1
        elif ch == ' ':
            tokens.append((' ', 1))
            i += 1
        else:
            j = i
            while j < n and ord(text[j]) <= 0x2E80 and text[j] != ' ':
                j += 1
            tokens.append((text[i:j], j - i))
            i = j
    return tokens


def wrap_cjk(text, max_units):
    text = text.replace('**', '').replace('`', '')
    lines, cur, u = [], '', 0
    for tok, w in _tokenize(text):
        if not cur and tok and tok[0] in PUNCT_NO_START and lines:
            lines[-1] += tok
            continue
        if cur and u + w > max_units:
            if tok and tok[0] in PUNCT_NO_START:
                lines.append(cur + tok)
                cur, u = '', 0
                continue
            if cur and cur[-1] in PUNCT_NO_END:
                moved = cur[-1]
                lines.append(cur[:-1])
                cur, u = moved + tok, 2 + w
                continue
            lines.append(cur)
            cur, u = ('' if tok == ' ' else tok), (0 if tok == ' ' else w)
        else:
            cur += tok
            u += w
    if cur:
        lines.append(cur)
    return lines


# ================================================================
#                     读 TSV 并重算区间（不手抄）
# ================================================================
def load_tsv(path):
    lut = {}
    with open(path, encoding='utf-8', newline='') as f:
        for d in csv.DictReader(f, delimiter='\t'):
            if d.get('run'):
                lut[(d['run'], d['split'])] = d
    return lut


LUT = load_tsv(B2)


def m(run, split, key='mAP50'):
    return float(LUT[(run, split)][key])


def rng(runs, split, key='mAP50'):
    v = [m(r, split, key) for r in runs]
    return min(v), max(v), sum(v) / len(v)


FLOOR = [f'nf_v11_640_s{s}' for s in (42, 43, 44)]
ARM = [f'att_v11_640_cbam_bn_s{s}' for s in (42, 43, 44)]
F_TEST = rng(FLOOR, 'test')
A_TEST = rng(ARM, 'test')
F_VAL = rng(FLOOR, 'val')
A_VAL = rng(ARM, 'val')
F896 = rng([f'nf_v11_896_s{s}' for s in (42, 43, 44)], 'test')
A896 = rng([f'att_v11_896_cbam_bn_s{s}' for s in (42, 43, 44)], 'test')


def sp(x):
    return f'{x * 100:.2f}'


def pt(x):
    return f'{x * 100:+.2f}'


def pair(r):
    return f'{sp(r[0])} – {sp(r[1])}'


# ---- 自检：勘误所依据的数字必须与 TSV 一致 ----
assert abs((A_TEST[2] - F_TEST[2]) * 100 - 1.62) < 0.01, A_TEST[2] - F_TEST[2]
assert abs((A_VAL[2] - F_VAL[2]) * 100 - 0.34) < 0.01, A_VAL[2] - F_VAL[2]
assert A_TEST[0] > F_TEST[1], '640 test 应错开'
assert A_VAL[0] < F_VAL[1] and A_VAL[1] > F_VAL[0], '640 val 应重叠'
assert abs((A896[2] - F896[2]) * 100 + 0.67) < 0.01, A896[2] - F896[2]

# ================================================================
#                        正文
# ================================================================
doc = Doc()
doc.title('勘 误 说 明')
doc.subtitle('《第二批 CBAM 修复实验 @896 探索报告 v1》§七 判据读法更正')
doc.note(f'勘误对象：report/第二批CBAM修复实验_896_探索报告_v1.pdf（2026-10-16 13:00 生成）　'
         f'勘误日期：{DATE}　性质：文字层读法更正；数值、图、表均不变。')

doc.h1('一、一句话结论')
doc.para('原文 §七 用「val 也必须错开」作为否决条件，严于出数前预登记的'
         '「test 区间错开，且 val 不反向矛盾」。按登记规则重读，640 CBAM+BN 的'
         '名义效应成立，但幅度临界、方向不稳，不足以采信；净结论（无可采信增益）不变。')

doc.h1('二、原文错在哪（只错在读法，不错在数字）')
doc.bullet('预登记原文（WORKLOG A.9.19）：判为真实效应的条件是 test 区间与地板区间错开，'
           '且 val 不反向矛盾；只有 val 反向错开（臂在 val 明显更差却在 test 更好）才否决。'
           'val 重叠本身不否决。')
doc.bullet('原文写法：要求 val 也必须错开，否则不构成效应。这把「val 重叠」误当成了否决条件。')
doc.bullet('来源：A.9.17 的「单边错开即噪声」是 n=1 单种子镜像情形的规则；n≥2 的区间判读不适用。')
doc.bullet('性质：文字层的判据读法错误，不是数值或计算错误；所有 mAP 数字均可复算。')

doc.h1('三、按正确读法重读 640 CBAM+BN')
doc.table(
    ['口径', '地板 [min, max]', 'CBAM+BN [min, max]', '区间关系', '均值差'],
    [
        ['test mAP50', pair(F_TEST), pair(A_TEST), '错开（余量 0.13 pt）', pt(A_TEST[2] - F_TEST[2]) + ' pt'],
        ['val mAP50', pair(F_VAL), pair(A_VAL), '重叠（同向，不反向）', pt(A_VAL[2] - F_VAL[2]) + ' pt'],
    ],
    col_widths=[0.18, 0.23, 0.23, 0.24, 0.12],
    note='数值由 batch2_eval.tsv 现读；两项均值差的小数位按四舍五入显示。')

doc.para('按登记规则：test 错开且 val 同向不矛盾 → 名义效应成立（+1.62 pt）。'
         '但「名义成立」不等于「可采信」，理由有三：')
doc.bullet('余量只有 0.13 pt，是地板跨度 2.25 pt 的 5.8%，属临界。')
doc.bullet('同臂在 896 的方向为负：区间 [0.8974, 0.9080] 与地板 [0.9044, 0.9122] 重叠，'
           f'均值差 {pt(A896[2] - F896[2])} pt。同一修改在另一分辨率不复制，方向不稳。')
doc.bullet('A.9.21 已证实同种子重跑可差 1.04 pt，大于 640 地板跨度 0.78 pt；'
           '因此三种子区间只是不稳定量的下界，0.13 pt 的错开不足以越过噪声。')
doc.para('→ 结论不变：640 CBAM+BN 没有可采信的增益。原文结论正确，理由需按上述更正。')

doc.h1('四、哪些内容不变')
doc.bullet('batch2 v1 PDF 的 §一至 §六、§八，以及全部数字、图、表，均不修改；'
           '本文件是其附属勘误，不重生成、不覆盖原 PDF。')
doc.bullet('WORKLOG A.9.23 §七已按正确读法登记；第三批报告（v1）§三已引用该更正。')
doc.bullet('评估脚本 eval_att_batch2.py 的旧警告行已于 2026-10-17 改为正确表述。')

doc.h1('五、关联的未决项（已暂停，不再追加实验）')
doc.bullet('mAP50-95 名义擦边偏低（Triplet −0.65 / CBAM+BN −1.20 pt，余量 0.15 / 0.18 pt）：'
           '0.15 pt 级低于同种子不稳定量 1.04 pt，不可信；要定它需 seed 45/46 或同种子重跑。')
doc.bullet('用户 2026-10-17 决定暂不追加实验；该项保持登记备查，不进入任何结论。')
doc.note(f'口径声明：探索阶段记录，未做显著性检验，不得进入 v_final。'
         f'数据来源 exp_upscale/logs/batch2_eval.tsv，生成日期 {DATE}。')

# ================================================================
#                        页码 / 落盘
# ================================================================
n = len(doc.pages)
for i, fig in enumerate(doc.pages):
    fig.text(0.5, 0.035, f'第 {i + 1} 页 / 共 {n} 页', ha='center', va='bottom',
             fontproperties=fp_note, color='#666666')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
PNG_DIR = os.environ.get('REPORT_PNG')
if PNG_DIR:
    os.makedirs(PNG_DIR, exist_ok=True)
    for i, fig in enumerate(doc.pages):
        fig.savefig(os.path.join(PNG_DIR, f'erratum_p{i + 1}.png'), dpi=110)
with PdfPages(OUT) as pdf:
    for fig in doc.pages:
        pdf.savefig(fig)
        plt.close(fig)

size = os.path.getsize(OUT)
print(f'OK  {OUT}')
print(f'pages = {n}  size = {size} bytes ({size / 1024:.1f} KB)')
print(f'640 test  floor={pair(F_TEST)}  arm={pair(A_TEST)}  d={pt(A_TEST[2]-F_TEST[2])} pt')
print(f'640 val   floor={pair(F_VAL)}  arm={pair(A_VAL)}  d={pt(A_VAL[2]-F_VAL[2])} pt')
print(f'896 test  floor={pair(F896)}  arm={pair(A896)}  d={pt(A896[2]-F896[2])} pt')
