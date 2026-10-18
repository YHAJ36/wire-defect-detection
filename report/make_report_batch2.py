# -*- coding: utf-8 -*-
"""第二批 CBAM 修复实验 —— 896 线探索报告生成器（独立脚本）

输出：D:/2/report/第二批CBAM修复实验_896_探索报告_v1.pdf
混淆矩阵：D:/2/report/_batch2_confusion_896/<run>/confusion_matrix[_normalized].png

════════════════════════════════════════════════════════════════════════════
【注意】硬约束：本脚本**绝不 import make_report_v5**
   `report/make_report_v5.py` 是**脚本不是模块** —— 它没有 `if __name__ == '__main__'`
   保护，顶部就写死 `OUT = '.../基于YOLOv11+CBAM的缺陷检测算法_v_final.pdf'`，
   文件末尾无条件执行 `with PdfPages(OUT) as pdf: ...`。
   一旦 import，它会**一路执行到底、直接重生成已封版的 v_final.pdf**。
   故本文件里的 Doc 容器类 / 字体设置 / wrap_cjk / _tokenize 全部是**复制**而非引用。
   要改版面请直接改本文件，不要回头去动 v_final 那份。
   （末尾另有 assert，防止 OUT 被误改到 v_final 路径上。）
════════════════════════════════════════════════════════════════════════════

数据来源（本报告所有数字均可追溯，不手抄）：
  · 判读数字       ← exp_upscale/logs/batch2_eval.tsv（由 eval_att_batch2.py 落盘）
  · 归档对照数字   ← exp_upscale/logs/eval_batch1.txt（wire_v11_896 / wire_v11_640）
  · 混淆矩阵       ← 本脚本现场用 YOLO(best.pt).val(test, imgsz=896, plots=True) 生成

用法：
  D:/yolo-env/Scripts/python.exe D:/2/report/make_report_batch2.py
  REPORT_PNG=1 ... make_report_batch2.py     # 另存逐页 PNG（默认不导出，见 AI 规则 8）
  REPORT_DEBUG=1 ... make_report_batch2.py   # 打印翻页/占位调试
"""
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

OUT = 'D:/2/report/第二批CBAM修复实验_896_探索报告_v1.pdf'
assert 'v_final' not in OUT, '不允许把本脚本的输出指向 v_final'
assert OUT.endswith('_v1.pdf')

ROOT = 'D:/2'
TSV = f'{ROOT}/exp_upscale/logs/batch2_eval.tsv'
CM_DIR = f'{ROOT}/report/_batch2_confusion_896'
RUNS_DIR = f'{ROOT}/exp_upscale/runs'
DATA = f'{ROOT}/dataset_final_bbox/data.yaml'
DATE = '2026-10-16'

# ================================================================
#                        字体（照抄 v_final 的设定）
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
fp_th_s = FontProperties(fname=F_BOLD, size=7.5)
fp_td = FontProperties(fname=F_REG, size=9)
fp_td_s = FontProperties(fname=F_REG, size=7.5)

# ================================================================
#                        版面常量 + Doc 容器（复制自 v_final）
# ================================================================
PAGE_W, PAGE_H = 8.27, 11.69          # A4
TOP, BOTTOM = 0.945, 0.065
X0, WIDTH = 0.10, 0.80
LINE = 0.0185
ROW = 0.028
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

    def need(self, h):
        """为"接下来高约 h 的整块内容"预留空间，不够就换页。

        用途：小标题与其配图不能被分页切开（否则标题会孤零零留在上一页页脚）。
        figure() 自带的 _ensure 只管图本身，管不到它上面的标题。
        """
        return self._ensure(h)

    def title(self, text):
        self.ax.text(0.5, self.y, text, ha='center', va='top',
                     fontproperties=fp_title, transform=self.ax.transAxes)
        self.y -= 0.042

    def subtitle(self, text):
        for ln in wrap_cjk(text, 72):
            self.ax.text(0.5, self.y, ln, ha='center', va='top',
                         fontproperties=fp_sub, color='#555555',
                         transform=self.ax.transAxes)
            self.y -= 0.030

    def h1(self, text):
        self._ensure(0.065)
        self.y -= 0.010
        self.ax.text(X0, self.y, text, ha='left', va='top',
                     fontproperties=fp_h1, transform=self.ax.transAxes)
        self.y -= 0.030

    def h2(self, text):
        self._ensure(0.055)
        self.ax.text(X0, self.y, text, ha='left', va='top',
                     fontproperties=fp_h2, transform=self.ax.transAxes)
        self.y -= 0.026

    def para(self, text, indent=True):
        lines = wrap_cjk(text, 88)
        indent_str = '　　'
        for i, ln in enumerate(lines):
            self._ensure(LINE * 1.2)
            s = (indent_str + ln) if (indent and i == 0) else ln
            self.ax.text(X0 + (0.018 if (indent and i > 0) else 0), self.y, s,
                         ha='left', va='top', fontproperties=fp_body,
                         transform=self.ax.transAxes)
            self.y -= LINE
        self.y -= 0.004

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
        self.y -= 0.003

    def table(self, col_labels, cell_text, col_widths=None, height=None, note=None,
              left=X0, width=WIDTH, small=False, row_h=None):
        rh = row_h or (ROW * 0.88 if small else ROW)
        h = height or (len(cell_text) + 1) * rh
        pad = 0.06
        if DEBUG:
            print(f'  [table {len(cell_text)}r] y={self.y:.3f} need={h + pad:.3f} '
                  f'{"→ new page" if self.y - (h + pad) < BOTTOM else "ok"}')
        self._ensure(h + pad)
        y0 = self.y - h
        tbl = self.ax.table(
            cellText=cell_text, colLabels=col_labels,
            bbox=(left, y0, width, h), cellLoc='center', colWidths=col_widths,
        )
        tbl.auto_set_font_size(False)
        fth = fp_th_s if small else fp_th
        ftd = fp_td_s if small else fp_td
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor('#333333')
            cell.set_linewidth(0.7)
            if r == 0:
                cell.set_facecolor('#dce6f1')
                cell.get_text().set_fontproperties(fth)
            else:
                cell.get_text().set_fontproperties(ftd)
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

    def caption(self, text):
        for ln in wrap_cjk(text, 74):
            self._ensure(LINE)
            self.ax.text(0.5, self.y, ln, ha='center', va='top',
                         fontproperties=fp_note, color='#333333',
                         transform=self.ax.transAxes)
            self.y -= LINE * 0.95
        self.y -= 0.006

    def figure(self, height, draw, caption, note=None, tick_pad=0.040, title_pad=0.026):
        if DEBUG:
            print(f'  [figure] y={self.y:.3f} need={height + 0.03:.3f}')
        self._ensure(height + 0.03)
        y0 = self.y - height
        ax = self.fig.add_axes([0.115, y0 + tick_pad, 0.80,
                                height - tick_pad - title_pad])
        draw(ax)
        self.y = y0 - 0.024
        self.caption(caption)
        if note:
            self.note(note)


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
    # batch2 局部改动（不影响 v_final 的排版逻辑）：去掉源码里的 "**" 强调标记。
    # 封版 make_report_v5.py 全文 0 处 "**"，其 Doc 容器没有加粗渲染（matplotlib 的
    # fontproperties 是单字体对象，实现加粗要换 msyhbd.ttc 并拆 run），故 "**" 只会
    # 原样印成星号。这里统一剥掉，正文语义不变、只少一层视觉强调。
    text = text.replace('**', '')
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


def img_draw(path, max_px=1400):
    """把 PNG 贴进 axes。先在内存里缩到 max_px，避免把 3000×2250 的原始图
    整幅塞进 PDF（否则单张图就能让 PDF 涨好几 MB）。"""
    from PIL import Image
    im = Image.open(path).convert('RGB')
    w, h = im.size
    k = max_px / max(w, h)
    if k < 1:
        im = im.resize((max(1, int(w * k)), max(1, int(h * k))), Image.LANCZOS)
    arr = np.asarray(im)

    def _d(ax):
        ax.imshow(arr)
        ax.axis('off')
    return _d


def fig_h_for(path, tick_pad=0.040, title_pad=0.026, max_h=0.46):
    """按图片真实长宽比算占位高度，铺满 0.80 页宽（不铺满会留白）。"""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    h_units = (WIDTH * PAGE_W) * h / w / PAGE_H
    return min(h_units, max_h) + tick_pad + title_pad


# ================================================================
#              一、读 batch2_eval.tsv → 判读数字（不手抄）
# ================================================================
def load_tsv():
    lut = {}
    with open(TSV, encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        for line in f:
            if not line.strip():
                continue
            p = line.rstrip('\n').split('\t')
            d = dict(zip(header, p))
            lut[(d['run'], d['split'])] = d
    return lut


LUT = load_tsv()

FLOOR_896 = ['nf_v11_896_s42', 'nf_v11_896_s43', 'nf_v11_896_s44']
FLOOR_640 = ['nf_v11_640_s42', 'nf_v11_640_s43', 'nf_v11_640_s44']
CBAM_896 = [f'att_v11_896_cbam_bn_s{s}' for s in (42, 43, 44)]
TRIP_896 = ['att_v11_896_triplet_s42']
CBAM_640 = [f'att_v11_640_cbam_bn_s{s}' for s in (42, 43, 44)]
TRIP_640 = [f'att_v11_640_triplet_s{s}' for s in (42, 43, 44)]

SUB = {'nf': 'baseline', 'att': 'arm'}


def val_of(run, split):
    return float(LUT[(run, split)]['mAP50'])


def rng(runs, split):
    v = [val_of(r, split) for r in runs]
    return min(v), max(v), sum(v) / len(v), len(v)


def pt(x):
    """百分点，带符号（Δ 用）。全篇 "pt" 一律指 ×100 后的百分点。"""
    return f'{x * 100:+.2f}'


def sp(x):
    """跨度（区间宽度），无符号百分点。

    注意：区间端点是 0~1 的小数，直接 `:.2f` 会印成 "0.01 pt"，
    与同一篇里 Δ 的 "0.78 pt" 差 100 倍 —— §七 表 6 曾踩过这个坑。
    凡印 "pt" 必须过 pt() 或 sp()。
    """
    return f'{x * 100:.2f}'


def judge(arm_lo, arm_hi, flo_lo, flo_hi):
    """预登记规则：臂区间与地板区间错开 → 真实效应；否则重叠不可分。"""
    if arm_hi < flo_lo or arm_lo > flo_hi:
        return '★ 错开 → 真实效应'
    return '· 重叠 → 不可分'


F896 = rng(FLOOR_896, 'test')
V896 = rng(FLOOR_896, 'val')
F640 = rng(FLOOR_640, 'test')
V640 = rng(FLOOR_640, 'val')
A_cbam896 = rng(CBAM_896, 'test')
A_trip896 = rng(TRIP_896, 'test')
A_cbam640 = rng(CBAM_640, 'test')
A_trip640 = rng(TRIP_640, 'test')

# 归档对照（数字来自 eval_batch1.txt；见脚本头"数据来源"）
ARCH = [
    ('896', 'runs/wire_v11_896（归档）', 42, 0.9148, 'nf_v11_896_s42'),
    ('640', 'runs/wire_v11_640（归档）', 42, 0.8992, 'nf_v11_640_s42'),
]

doc = Doc()

# ================================================================
#                        封面 / 摘要
# ================================================================
doc.y = 0.80
doc.title('第二批 CBAM 修复实验')
doc.subtitle('896 线探索报告　·　CBAM+BN / Triplet 双臂　·　2026-10-16')
doc.y -= 0.02
doc.para('口径定位：896 线（报告第十一章独立口径；v11 基线 test 0.9148 所在）。'
         '本报告是**探索性**记录，不属于任何交付形态。')
doc.y -= 0.01
doc.h1('摘要')
doc.para('一句话结论：896 线上 CBAM+BN 与 Triplet 两条臂**全部与噪声地板不可分**；'
         'CBAM 读出头退化机理已诊断，修复版在**未训练态**不再退化为常数——'
         '但**训练后是否仍不退化为常数未在本批复测**——且无论机理如何，'
         '修复**没有带来可测的增益**。')
doc.bullet('噪声地板（896 基线 × 3 种子，test mAP50 区间）= '
           f'[{F896[0]:.4f}, {F896[1]:.4f}]，跨度 {sp(F896[1] - F896[0])} pt。')
doc.bullet(f'CBAM+BN@neck（3 种子）区间 [{A_cbam896[0]:.4f}, {A_cbam896[1]:.4f}]，'
           f'Δ vs 地板均值 {pt(A_cbam896[2] - F896[2])} pt → '
           f'{judge(A_cbam896[0], A_cbam896[1], F896[0], F896[1])}。')
doc.bullet(f'Triplet@neck（**n=1**，0.9105）Δ {pt(A_trip896[2] - F896[2])} pt → '
           '不可判（预登记规则要求 n≥2）。')
doc.bullet('附带发现（本报告最重要的一条）：**同种子重跑差 1.04 pt > 地板跨度 0.78 pt** '
           '→ 3 种子地板是**下界**，不是尺度；归档的 0.9148 落在地板区间**之外**。')

# ================================================================
#                        一、背景
# ================================================================
doc.h1('一、背景：要修的是哪一处')
doc.para('已封版的 v_final 第五章定量证明：CBAM 在本数据集上近似恒等映射，'
         '根因定位在**空间注意力（SA）的读出头**——`s = sigmoid(conv(...))` 里，'
         'sigmoid 的输入 z 没有任何东西约束，尺度与偏移全由那个无偏置 7×7 conv 自由决定。'
         '由此长出两个机制：')
doc.bullet('DC 饱和：浅层（层 2/4/6）z 有大正直流（+3.1~+4.9），sigmoid 单边饱和 → 输出恒为 1；')
doc.bullet('幅度衰减：深层（层 8/9）z≈0（−0.23 / −0.08，未饱和）但该处 z 的 std 比输入小 2~5 倍，'
           '→ 输出 std 仅 0.032（层 8/9 实测 0.0315 / 0.0320），趋近常数 0.5。')
doc.para('常数注意力图被其后的 BN 严格吸收（BN(c·x)=BN(x)），网络从这个自由度学不到东西。')
doc.need(0.30)      # 标题 + 表 1（6 行）+ 表注不能被分页切开
doc.h2('两条修复路线（本批双臂）')
doc.table(
    ['', 'CBAM+BN', 'Triplet'],
    [['本质', '原 CBAM，只给 SA 读出头加一层 BN', '官方 TripletAttention（另一套模块，不改 CBAM）'],
     ['读出头', 'sigmoid(BN(conv(cat[mean,max])))', 'sigmoid(BN(conv(ZPool(x))))'],
     ['那层 BN', '我们加的（唯一实质改动）', '官方 BasicConv 里本来就有'],
     ['分支', '通道注意力 × 空间注意力', '三分支 C×W / H×C / H×W，等权平均'],
     ['比基线多', '+32,868 参数', '+300 参数']],
    col_widths=[0.13, 0.44, 0.43], small=True, row_h=0.030,
    note='表 1　两条修复路线对照。两者针对的是**同一处**（sigmoid 的输入 z 未被约束），'
         '但来源不同：一条是"往老模块里补"，一条是"换成天生就带 BN 的模块"——'
         '这正是设计上要它们**互为印证**的原因。')
doc.note('【注意】两条臂不是严格单变量对照：Triplet 相对 CBAM+BN 还同时改了分支数与聚合方式，'
         '故**注入位置是控住的（都在 layer 9 / neck_sppf），模块结构没有控住**。')

# ================================================================
#                        二、实验设置
# ================================================================
doc.h1('二、实验设置')
doc.bullet('数据：dataset_final_bbox，1428 / 178 / 179（train/val/test），3 类'
           '（break 断股 / thunderbolt 雷击 / wear 磨损）。')
doc.bullet('896 线：imgsz=896、batch=8、epochs=60、SGD lr0=0.01、seed 42/43/44；'
           '**workers=0**（与封版管线一致，刻意不改——改了本批就不可比）。')
doc.bullet('注入位置：统一 layer 9（neck_sppf），baseline 不注入。')
doc.bullet('评估：YOLO(best.pt) 每次全新加载 → val(data, split, imgsz=训练 imgsz, '
           'batch=8, workers=0, device=0)，test 为主判据、val 作一致性检查。')
run_rows = []
for r in FLOOR_896 + CBAM_896 + TRIP_896:
    arm = 'baseline' if r.startswith('nf_') else ('CBAM+BN' if 'cbam_bn' in r else 'Triplet')
    seed = r.rsplit('_s', 1)[1]
    run_rows.append([r, arm, seed, 'OK', f'{val_of(r, "test"):.4f}', f'{val_of(r, "val"):.4f}'])
doc.table(['run', '臂', 'seed', '注入闸门', 'test mAP50', 'val mAP50'], run_rows,
          col_widths=[0.34, 0.15, 0.09, 0.14, 0.14, 0.14], small=True, row_h=0.0225,
          note='表 2　896 线 run 清单（7 行）。注入闸门 = 训练后核验注意力层参数量是否'
               '与预期一致（防止 ultralytics 重建模型时把注入层丢掉）。')
doc.h2('判读规则（出数前预登记，未事后更改）')
doc.bullet('噪声地板 = 同分辨率 3 个 baseline 种子的 test mAP50 区间 [min, max]；'
           '臂 = 同分辨率种子的区间。**错开**（臂 min > 地板 max 或 臂 max < 地板 min）'
           '→ 真实效应；**重叠** → 不可分，诚实记负。')
doc.bullet('**test 为主判据，val 只作一致性检查**：best.pt 由 ultralytics 在 val 上按 fitness '
           '从 60 轮中挑出 → val 是选择集、带向上偏置。规则是'
           '"**test 区间错开 且 val 不反向矛盾**"才记真实效应。')
doc.bullet('两臂互相矛盾 → 两次独立噪声，两臂均记不可分；不择一采信、不合并两臂凑 n。')
doc.bullet('n<2 只印数、**不下判读**。争议只能靠**加种子**解决。')

# ================================================================
#                        三、896 噪声地板
# ================================================================
doc.h1('三、896 噪声地板（baseline × 3 种子，仅 seed 不同）')
doc.table(['种子', 'test mAP50', 'val mAP50'],
          [[r.rsplit('_s', 1)[1], f'{val_of(r, "test"):.4f}', f'{val_of(r, "val"):.4f}']
           for r in FLOOR_896] +
          [['区间', f'[{F896[0]:.4f}, {F896[1]:.4f}]', f'[{V896[0]:.4f}, {V896[1]:.4f}]'],
           ['跨度', f'{sp(F896[1] - F896[0])} pt', f'{sp(V896[1] - V896[0])} pt'],
           ['均值', f'{F896[2]:.4f}', f'{V896[2]:.4f}']],
          col_widths=[0.22, 0.39, 0.39], row_h=0.028,
          note='表 3　896 噪声地板。**test 跨度 0.78 pt 是本批一切判读的尺子。**'
               'val 跨度 2.95 pt 明显更宽（预登记时预期 val 会因选择偏置被撑宽，'
               '896 侧与此相符；但 640 侧不符，见第七节末）。')

# ================================================================
#                        四、896 判读结果
# ================================================================
doc.h1('四、896 判读结果')
doc.table(['臂', 'n', 'test 区间', 'Δ vs 地板均值', 'test 判读'],
          [['CBAM+BN@neck', '3', f'[{A_cbam896[0]:.4f}, {A_cbam896[1]:.4f}]',
            pt(A_cbam896[2] - F896[2]), judge(A_cbam896[0], A_cbam896[1], F896[0], F896[1])],
           ['Triplet@neck', '1', f'仅单种子（{A_trip896[0]:.4f}）',
            pt(A_trip896[2] - F896[2]), '不可判（需 ≥2 种子）']],
          col_widths=[0.24, 0.07, 0.26, 0.19, 0.24], row_h=0.030,
          note=f'表 4　896 判读汇总。地板区间 [{F896[0]:.4f}, {F896[1]:.4f}]，'
               f'跨度 {sp(F896[1] - F896[0])} pt。')
doc.para('结论：**两条臂均不可分**。CBAM+BN 的 test 区间 '
         f'[{A_cbam896[0]:.4f}, {A_cbam896[1]:.4f}] 完全落在地板区间内，'
         f'且方向为负（Δ {pt(A_cbam896[2] - F896[2])} pt）——'
         '即"修复了读出头"这件事在 896 线上没有换来任何可测的精度变化。'
         'Triplet 只有单种子，按预登记规则**不作判读**。')
doc.para('这个负结果是有信息量的：两条臂来自**不同来源**（一条补 BN、一条换模块），'
         '若"读出头 BN"是涨分的真原因，两条都该涨；两条都没能在噪声之上涨，'
         '说明**读出头退化不是本数据集上精度受限的主因**——'
         '至少不是靠这一处改动能兑现的因素。')

# ================================================================
#                        五、附带发现
# ================================================================
doc.h1('五、★ 附带发现：同种子重跑差 > 地板跨度')
doc.para('本批最重要的一条发现不在臂上，在**地板本身**。')
doc.table(['线', '归档 run（seed 42）', '归档 test', '本批同种子重跑', '重跑 test', '差'],
          [[a[0], a[1].split('/')[-1], f'{a[3]:.4f}', a[4], f'{val_of(a[4], "test"):.4f}',
            f'{abs(a[3] - val_of(a[4], "test")) * 100:.2f} pt'] for a in ARCH],
          col_widths=[0.07, 0.28, 0.13, 0.24, 0.13, 0.15], small=True, row_h=0.026,
          note='表 5　同种子重跑对照。两个 run 的 args.yaml 逐字 diff **只差 6 项，'
               '全是路径/落盘项（model→last.pt、save_period、project、name、resume、save_dir），'
               '没有一个训练超参**：同 seed、同 imgsz、同 workers=0、同 batch、同全部增广、'
               '同 deterministic=true、同 amp=true。')
doc.para('896 那对的差是 **1.04 pt**，而 896 地板跨度只有 **0.78 pt**；'
         f'且归档的 {ARCH[0][3]:.4f} **高出地板区间上界 {F896[1]:.4f} '
         f'{abs(ARCH[0][3] - F896[1]) * 100:.2f} pt**，落在区间之外。'
         '后果有三条：')
doc.bullet('**3 种子地板低估了真实跑数不确定度**——同种子差异本身就大于种子间跨度。'
           '地板是**下界**，不是尺度。')
doc.bullet(f'**本批 11 次跑没有一次达到 {ARCH[0][3]:.4f}**（最高 '
           f'{max(val_of(r, "test") for r in FLOOR_896 + CBAM_896 + TRIP_896):.4f}，'
           '还是个 baseline 种子）→ 归档值是"运气好的一次"，不能当中心值。')
doc.bullet('**报告里那个头号数字偏乐观**：报告第十一章用 0.9148 与 YOLOv5s 的 0.9310 比，'
           '得"v5 高 1.6 pt（方向性，非量化）"。按本批区间重算，该差距为 1.9~2.7 pt → '
           '**方向不变、量值偏大**。'
           '（注：v5 那条未在本批复核，不能与本批数字直接等价，此处只作量级示意。）')
doc.h2('机制：这条管线不是位级可复现的')
doc.bullet('`deterministic: true` 只把 cuDNN 限制在确定性算法集；但 ultralytics 的 '
           '`init_seeds` 里同时设 **`cudnn.benchmark = True`**——autotune 按**实测耗时**'
           '选核，选中的核可能逐次不同；AMP 亦含非确定性算子。')
doc.bullet('且本批 `nf_v11_896_s42` **在第 2 轮崩过一次**，日志随后出现 '
           '`Resuming training ... from epoch 2 to 60`（其余 10 个 run 无续训）。'
           '而每次训练启动 `_setup_train()` 都会调 `init_seeds(...)` → '
           '**RNG 被重置回初始状态**，于是 epoch 2 起的增广序列与"一口气跑完"的流不同。')
doc.note('诚实边界：本批数据**分不清**这 1.04 pt 是"续训换随机流"还是"AMP/cuDNN 残余'
         '不确定性"。两者都指向同一句话：**同种子重现性弱于地板暗示的水平**。')
doc.para('措辞澄清：这不叫"选择偏置"。选择偏置在本项目里专指"best.pt 由 val 挑出"，'
         '而两侧同构都存在（归档与本批都是 val 挑完测 test）。本条证据指向的是'
         '更朴素也更麻烦的事：**同种子跑数本身不稳定**。')

# ================================================================
#                    六、混淆矩阵（896，3 张）
# ================================================================
doc.h1('六、混淆矩阵（896 · test）')
doc.para('本节三张图**全部取自第二批重跑件**，与归档推荐模型'
         '（`runs/wire_v11_896`，test 0.9148）**不是同一权重**——'
         '它们是第五节讨论的那批重跑件，其中基线那张即 test 0.9044 的那一次。'
         '三者同源同口径，故三张图之间可比；但**不可与报告第十一章的 0.9148 那张直接对应**。')
doc.note('读图方向（必须说明，否则"哪两类易混"会读反）：矩阵**行 = 预测、列 = 真值**'
         '（ultralytics 的 `matrix[dc, gc] += 1  # TP`；轴标 x=True、y=Predicted）。'
         '**归一化是按列归的**（除以列和），故归一化矩阵**列和 = 1**，'
         '读的是"该**真值**类被预测成各类的比例"，对角线即该类**召回率**。'
         '第 4 行/列是 background：**列** = 真值为背景（即误检 FP），'
         '**行** = 预测为背景（即漏检 FN）。')

CM_RUNS = [
    ('nf_v11_896_s42', '图 1　896 基线（batch2 rerun, seed 42, test 0.9044）'),
    ('att_v11_896_cbam_bn_s42', '图 2　896 CBAM+BN@neck（batch2 rerun, seed 42, test 0.8982）'),
    ('att_v11_896_triplet_s42', '图 3　896 Triplet@neck（batch2 rerun, seed 42, test 0.9105）'),
]


def ensure_cm(run):
    """生成（或复用）混淆矩阵：落 CM_DIR/<run>/，并写 confusion_matrix.json 供复用。"""
    d = os.path.join(CM_DIR, run)
    jp = os.path.join(d, 'confusion_matrix.json')
    if os.path.exists(jp):
        with open(jp, encoding='utf-8') as f:
            print(f'  [cm] 复用 {jp}')
            return json.load(f)
    sys.path.insert(0, f'{ROOT}/code/yolov11')     # 反序列化注入层需要
    sys.path.insert(0, f'{ROOT}/code/yolov12')
    from ultralytics import YOLO
    w = os.path.join(RUNS_DIR, run, 'weights', 'best.pt')
    if not os.path.exists(w):
        # 注意：本脚本 stdout 在本机是 GBK，"✓""❌"等符号不在 GBK 内，print 会崩。
        # 故本脚本所有 print 只用 GBK 可编码字符（ASCII + 中文 + ★·×Δ→ 等）。
        raise SystemExit(f'!! 缺权重：{w}')
    print(f'  [cm] 生成 {run} …')
    m = YOLO(w)
    r = m.val(data=DATA, split='test', imgsz=896, batch=8, workers=0, device=0,
              project=CM_DIR, name=run, exist_ok=True, verbose=False,
              plots=True, save_json=False)
    # ultralytics 的 confusion_matrix.plot 会按 title 落两个文件名：
    #   "Confusion Matrix"            → confusion_matrix.png
    #   "Confusion Matrix Normalized" → confusion_matrix_normalized.png
    got = sorted(x for x in os.listdir(d) if x.startswith('confusion_matrix')
                 and x.endswith('.png'))
    b = r.box
    rec = {
        'run': run,
        'names': [r.names[int(i)] for i in sorted(r.names)],
        'matrix': np.asarray(r.confusion_matrix.matrix).astype(int).tolist(),
        'metrics': {'test_map50': float(b.map50), 'test_map50_95': float(b.map),
                    'P': float(b.mp), 'R': float(b.mr)},
        'png': got,
    }
    os.makedirs(d, exist_ok=True)
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(f'      落盘 {got}；test mAP50={rec["metrics"]["test_map50"]:.4f}')
    return rec


doc.pages[-1]  # noqa
doc.h2('生成与口径')
doc.note('混淆矩阵由本脚本现场生成：`YOLO(best.pt).val(split=test, imgsz=896, batch=8, '
         'workers=0, device=0, plots=True)`，产物落 `report/_batch2_confusion_896/<run>/`。'
         '【注意】`plots=True` 除两张混淆矩阵外，**还会连带写** '
         '`BoxPR_curve.png` / `BoxF1_curve.png` / `BoxP_curve.png` / `BoxR_curve.png` '
         '与 `val_batch0_labels.jpg` / `val_batch0_pred.jpg`（896 尺寸，体积不小）——'
         '目录里文件的数目多于两张，属正常。'
         '本节正文给出**计数版**（表格式），图中嵌的是**归一化版**（易混比例读它更直观）。')

cm_recs = []
for run, cap in CM_RUNS:
    rec = ensure_cm(run)
    cm_recs.append((run, cap, rec))
    # 与 TSV 对账（同一权重同 split 的复评应逐位一致）
    exp = val_of(run, 'test')
    got = rec['metrics']['test_map50']
    flag = 'OK' if abs(got - exp) < 5e-4 else '★ 与 TSV 不一致'
    print(f'  [check] {run}: 现场 {got:.4f} vs TSV {exp:.4f} {flag}')

doc.pages[-1]
for run, cap, rec in cm_recs:
    nrm = os.path.join(CM_DIR, run, 'confusion_matrix_normalized.png')
    raw = os.path.join(CM_DIR, run, 'confusion_matrix.png')
    if not os.path.exists(nrm):
        raise SystemExit(f'!! 缺归一化混淆矩阵图：{nrm}')   # 不用 ❌：GBK stdout 会崩
    names = rec['names']
    M = np.asarray(rec['matrix'])
    nc = len(names)
    h_fig = fig_h_for(nrm)
    # 标题 + 图 + 图注整体预留（h_fig + 图备注/图注约 0.14），避免标题被分页切开
    doc.need(h_fig + 0.14)
    doc.h2(f'{run}')
    doc.figure(h_fig, img_draw(nrm), caption=cap,
               note='读法：行 = 预测、列 = 真值；按**列**归一化 → 列和 = 1，'
                    '对角线 = 该类召回率。左起第 4 列为 background（误检），'
                    '第 4 行为 background（漏检）。'
                    f'（同一 run 的计数版见 `_batch2_confusion_896/{run}/confusion_matrix.png`）')
    # 计数表：行=预测（含 background），列=真值（含 background）
    hdr = ['预测 \\ 真值'] + names + ['background']
    rows = []
    for i in range(nc + 1):
        lab = names[i] if i < nc else 'background'
        rows.append([lab] + [str(int(M[i, j])) for j in range(nc + 1)])
    doc.table(hdr, rows, col_widths=[0.16] + [0.14] * (nc + 1), small=True, row_h=0.024,
              note=f'计数矩阵（{run}）。行和 = 该类被预测出去的总数；'
                   f'列和 = 该类真值实例总数。')
    # 分析：召回、最易混、FP/FN
    # 召回率分母必须是"该类真值实例总数" = 整列和（含第 nc 行 background，
    # 即被判成背景的漏检）。此前误写成 M[:nc, :nc].sum(0) 切掉了 background 行，
    # 分母只剩"判成前景的"部分 → 百分比虚高（曾印出 134/134 = 100.0%）。
    colsum = M[:, :nc].sum(0)
    diag = [M[j, j] for j in range(nc)]
    rec_txt = '、'.join(f'{names[j]} {diag[j]}/{colsum[j]}（{diag[j] / max(colsum[j], 1):.1%}）'
                        for j in range(nc))
    off = [(M[i, j], names[j], names[i]) for j in range(nc) for i in range(nc) if i != j]
    off.sort(reverse=True)
    top = off[0]
    fp = [int(M[i, nc]) for i in range(nc)]
    fn = [int(M[nc, j]) for j in range(nc)]
    note = ''.join(f'{names[j]} {fn[j]}、' for j in range(nc)).rstrip('、')
    doc.para(f'对角线（真值→正确预测）：{rec_txt}。'
             f'**最易混**：真值 {top[1]} 被预测成 {top[2]} 共 {top[0]} 例。'
             f'误检（判成某类但真值为背景）：'
             + '、'.join(f'{names[i]} {fp[i]}' for i in range(nc)) + '。'
             f'漏检（判成背景）：{note}。', indent=True)

# ================================================================
#                    七、640 线判读（备查）
# ================================================================
doc.h1('七、640 线判读（备查）')
doc.para('本报告主体是 896 线，但第二批**同时补了 640 的种子**。'
         '其中 **640 CBAM+BN 的 +1.62 pt 是本批唯一"擦边"的数字**，'
         '故一并列出备查——不列会让"这批到底有没有看到一个接近显著的东西"'
         '这个问题无从回答。')
doc.table(['分辨率', '地板区间（跨度）', '臂', 'n', '臂区间', 'Δ vs 地板均值', '判读'],
          [['896', f'[{F896[0]:.4f}, {F896[1]:.4f}]（{sp(F896[1] - F896[0])} pt）',
            'CBAM+BN', '3', f'[{A_cbam896[0]:.4f}, {A_cbam896[1]:.4f}]',
            pt(A_cbam896[2] - F896[2]), '重叠 → 不可分'],
           ['896', '同上', 'Triplet', '1', f'{A_trip896[0]:.4f}',
            pt(A_trip896[2] - F896[2]), '不可判（n=1）'],
           ['640', f'[{F640[0]:.4f}, {F640[1]:.4f}]（{sp(F640[1] - F640[0])} pt）',
            'CBAM+BN', '3', f'[{A_cbam640[0]:.4f}, {A_cbam640[1]:.4f}]',
            pt(A_cbam640[2] - F640[2]), '★ 擦边，但规则判不成立'],
           ['640', '同上', 'Triplet', '3', f'[{A_trip640[0]:.4f}, {A_trip640[1]:.4f}]',
            pt(A_trip640[2] - F640[2]), '重叠 → 不可分']],
          col_widths=[0.08, 0.24, 0.13, 0.05, 0.19, 0.14, 0.17], small=True, row_h=0.030,
          note='表 6　640 线判读（备查）。**跨分辨率数字不可直接相比**'
               '——896 与 640 是两条线，各自比各自的噪声地板。')
lo_gap = (A_cbam640[0] - F640[1]) * 100
doc.para(f'640 CBAM+BN 之所以只判"擦边"而非"真实效应"：它的 test 下界 '
         f'{A_cbam640[0]:.4f} 只高出地板上界 {F640[1]:.4f} **{lo_gap:.2f} pt '
         f'= 地板跨度 {sp(F640[1] - F640[0])} pt 的 '
         f'{lo_gap / ((F640[1] - F640[0]) * 100):.1%}**，'
         '属于临界错开；而预登记规则要求"test 错开**且** val 不反向矛盾"，'
         '本臂在 val 侧重叠 → **规则判不成立**。'
         '这条规则正是为了拦住"只看 test 的 0.13 pt 就宣布涨了"这种读法而写的'
         '（A.9.17 已出现过同型的镜像情形）。', indent=True)
doc.para('另两条备查观察：（1）两臂方向互相矛盾——640 CBAM+BN '
         f'{pt(A_cbam640[2] - F640[2])} pt 而 896 CBAM+BN {pt(A_cbam896[2] - F896[2])} pt，'
         '按预登记口径这是**两次独立噪声**，不择一采信。'
         '（2）跨分辨率地板均值差 '
         f'{abs(F896[2] - F640[2]) * 100:.2f} pt（896 {F896[2]:.4f} vs 640 {F640[2]:.4f}），'
         '两区间不重叠——**仅记录，不解读**，因为跨分辨率数字不可直接相比。', indent=True)
doc.note('"val 被撑宽"这条预登记推理目前只有弱支持：896 侧支持'
         f'（val 跨度 {sp(V896[1] - V896[0])} pt 远大于 test {sp(F896[1] - F896[0])} pt），'
         f'640 侧**不支持**（val {sp(V640[1] - V640[0])} pt vs test {sp(F640[1] - F640[0])} pt，'
         '几乎一样宽）→ 1/2 支持，不足以当既定事实。'
         '但它只影响"val 只作一致性检查"这一用法，而该用法本身偏保守，故不影响结论。')

# ================================================================
#                        八、口径声明（文末，必须带）
# ================================================================
doc.h1('八、口径声明（引用本报告任何数字时必须一并带上）')
doc.note('探索阶段：地板 3 种子 / CBAM+BN 3 种子 / **Triplet 896 线 n=1（不可判）**；'
         '只报方向与幅度，未做显著性检验。')
doc.note('评估 imgsz = 训练 imgsz = 896；跨分辨率数字不可直接相比'
         '（896 与 640 是两条线，各自比各自的噪声地板）。')
doc.note('**本批数字不得进入 v_final 或任何交付形态。**')
doc.note('本报告为探索性记录，其结论（两臂不可分）不改变已封版 v_final 的任何结论；'
         '第五节关于"归档 0.9148 偏乐观"的发现已登记为待决事项，'
         '是否重估报告数字由后续决定。')
doc.note(f'数据来源：exp_upscale/logs/batch2_eval.tsv（判读）、'
         f'exp_upscale/logs/eval_batch1.txt（归档对照）、'
         f'report/_batch2_confusion_896/（混淆矩阵）。生成日期 {DATE}。')

# ================================================================
#                        页码 / 落盘
# ================================================================
n = len(doc.pages)
for i, fig in enumerate(doc.pages):
    fig.text(0.5, 0.035, f'第 {i + 1} 页 / 共 {n} 页', ha='center', va='bottom',
             fontproperties=fp_note, color='#666666')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
# 预览图门控（AI 规则 8）：只在设了 REPORT_PNG 时才导出逐页 PNG。
# 不设 = 不刷新旧预览图 → "看预览图"不能证明 PDF 已更新，必须同时看 stdout 的页数/大小。
PNG_DIR = os.environ.get('REPORT_PNG')
if PNG_DIR:
    os.makedirs(PNG_DIR, exist_ok=True)
    for i, fig in enumerate(doc.pages):
        fig.savefig(os.path.join(PNG_DIR, f'batch2_p{i + 1}.png'), dpi=110)
with PdfPages(OUT) as pdf:
    for fig in doc.pages:
        pdf.savefig(fig)
        plt.close(fig)

size = os.path.getsize(OUT)
print(f'OK  {OUT}')
print(f'页数 = {n}  大小 = {size} bytes ({size / 1024:.1f} KB)')
print(f'混淆矩阵目录 = {CM_DIR}')
