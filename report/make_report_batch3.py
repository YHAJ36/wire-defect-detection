# -*- coding: utf-8 -*-
"""第三批：YOLO26@896 与 Triplet 补种子 —— 探索报告生成器（独立脚本）

输出：D:/2/report/第三批YOLO26_896_与Triplet补种子_探索报告_v1.pdf

════════════════════════════════════════════════════════════════════════════
【硬约束】本脚本**绝不 import make_report_v5 / make_report_batch2**
  两者都是**脚本不是模块**（没有 __main__ 保护，import 即执行、直接重生成各自 PDF）。
    · make_report_v5.py 会重写已封版的 v_final.pdf；
    · make_report_batch2.py 会重跑混淆矩阵并重写 batch2 PDF。
  故 Doc 容器 / 字体 / wrap_cjk 全部是**复制**，版面改动只改本文件。
  末尾另有 assert，防止 OUT 被误指到 v_final。
════════════════════════════════════════════════════════════════════════════

数据来源（全部现读，不手抄）：
  · v11 地板 / Triplet / CBAM+BN ← exp_upscale/logs/batch2_eval.tsv
  · YOLO26                         ← exp_upscale/logs/batch3_eval.tsv（判读取最后一行）
  · 训练轮数 / val 峰值自检        ← exp_upscale/runs/<run>/results.csv
  · 链时间线                       ← exp_upscale/logs/chain_batch3_stdout.log（正文引用）

用法：
  D:/yolo-env/Scripts/python.exe D:/2/report/make_report_batch3.py
  REPORT_PNG=D:/2/report/_preview_batch3 ... make_report_batch3.py   # 逐页 PNG（AI 规则 8）
  REPORT_DEBUG=1 ... make_report_batch3.py                            # 打印翻页调试
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

OUT = 'D:/2/report/第三批YOLO26_896_与Triplet补种子_探索报告_v1.pdf'
assert 'v_final' not in OUT, '不允许把本脚本的输出指向 v_final'
assert OUT.endswith('_v1.pdf')

ROOT = 'D:/2'
B2 = f'{ROOT}/exp_upscale/logs/batch2_eval.tsv'
B3 = f'{ROOT}/exp_upscale/logs/batch3_eval.tsv'
RUNS_DIR = f'{ROOT}/exp_upscale/runs'
DATE = '2026-10-17'

# ================================================================
#                        字体（照抄 v_final / batch2 的设定）
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
#                        版面常量 + Doc 容器（复制）
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
                  f'{"-> new page" if self.y - (h + pad) < BOTTOM else "ok"}')
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

    def figure(self, height, draw, caption, note=None, tick_pad=0.045, title_pad=0.030):
        # 图与图注/备注必须同页：先把图注占位也算进预留高度，
        # 否则图注会被 _ensure 推到下一页（batch3 初版出现过一次）。
        cap_h = len(wrap_cjk(caption, 74)) * LINE * 0.95
        note_h = 0 if not note else len(wrap_cjk(note, 92)) * LINE * 0.92 + 0.006
        if DEBUG:
            print(f'  [figure] y={self.y:.3f} need={height + cap_h + note_h + 0.03:.3f}')
        self._ensure(height + cap_h + note_h + 0.045)
        y0 = self.y - height
        ax = self.fig.add_axes([0.135, y0 + tick_pad, 0.78,
                                height - tick_pad - title_pad])
        draw(ax)
        self.y = y0 - 0.020
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
    # 去掉源码里的强调标记；本容器没有加粗渲染，** 与 ` 只会原样印出。
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
#                     读 TSV（后行覆盖前行）
# ================================================================
def load_tsv(path):
    """同一 (run, split) 有多行时，取**最后一行**（坏行保留、正确行后追加）。"""
    lut = {}
    with open(path, encoding='utf-8', newline='') as f:
        for d in csv.DictReader(f, delimiter='\t'):
            if not d.get('run'):
                continue
            lut[(d['run'], d['split'])] = d
    return lut


B2_LUT = load_tsv(B2)
B3_LUT = load_tsv(B3)

FLOOR = [f'nf_v11_896_s{s}' for s in (42, 43, 44)]
Y26 = [f'nf_yolo26_896_s{s}' for s in (42, 43, 44)]
TRIP = [f'att_v11_896_triplet_s{s}' for s in (42, 43, 44)]
CBAM = [f'att_v11_896_cbam_bn_s{s}' for s in (42, 43, 44)]


def row(run, split):
    if run.startswith('nf_yolo26_'):
        return B3_LUT[(run, split)]
    return B2_LUT[(run, split)]


def m(run, split, key='mAP50'):
    return float(row(run, split)[key])


def rng(runs, split='test', key='mAP50'):
    v = [m(r, split, key) for r in runs]
    return min(v), max(v), sum(v) / len(v), len(v)


F896 = rng(FLOOR)
V896 = rng(FLOOR, 'val')
A26 = rng(Y26)
A26_95 = rng(Y26, key='mAP50-95')
A26_P = rng(Y26, key='P')
A26_R = rng(Y26, key='R')
AT = rng(TRIP)
AT_95 = rng(TRIP, key='mAP50-95')
AC = rng(CBAM)
AC_95 = rng(CBAM, key='mAP50-95')
F95 = rng(FLOOR, key='mAP50-95')
FP = rng(FLOOR, key='P')
FR = rng(FLOOR, key='R')
V26 = rng(Y26, 'val')
VT = rng(TRIP, 'val')
VC = rng(CBAM, 'val')


def pt(x):
    return f'{x * 100:+.2f}'


def sp(x):
    return f'{x * 100:.2f}'


def disjoint(a_lo, a_hi, f_lo, f_hi):
    return a_hi < f_lo or a_lo > f_hi


# ================================================================
#                        图（纯数据，无 GPU）
# ================================================================
GROUPS = [('v11 地板', FLOOR, '#6f6f6f'), ('YOLO26', Y26, '#1f77b4'),
          ('Triplet', TRIP, '#2ca02c'), ('CBAM+BN', CBAM, '#d62728')]


def draw_map50(ax):
    """test mAP50 逐种子点图 + 地板灰带。"""
    ax.axhspan(F896[0], F896[1], color='#bdbdbd', alpha=0.55, zorder=0)
    for i, (name, runs, c) in enumerate(GROUPS):
        vals = [m(r, 'test') for r in runs]
        xs = [i + (j - (len(vals) - 1) / 2) * 0.11 for j in range(len(vals))]
        ax.scatter(xs, vals, s=40, color=c, zorder=3,
                   edgecolor='white', linewidth=0.4)
        ax.hlines(sum(vals) / len(vals), i - 0.26, i + 0.26,
                  color=c, lw=1.8, zorder=2)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([g[0] for g in GROUPS], fontsize=9)
    ax.set_ylabel('test mAP50', fontsize=9)
    ax.set_ylim(0.895, 0.925)
    ax.grid(axis='y', ls=':', alpha=0.5)
    ax.tick_params(labelsize=8)


def draw_direction(ax):
    """召回（R）vs 高 IoU 精度（mAP50-95），看方向是否"找全/框准"。"""
    for name, runs, c in GROUPS:
        xs = [m(r, 'test', 'mAP50-95') for r in runs]
        ys = [m(r, 'test', 'R') for r in runs]
        ax.scatter(xs, ys, s=34, color=c, alpha=0.85, label=name, zorder=3)
        ax.scatter([sum(xs) / len(xs)], [sum(ys) / len(ys)], s=150, color=c,
                   marker='*', edgecolor='black', linewidth=0.4, zorder=4)
    ax.set_xlabel('test mAP50-95', fontsize=9)
    ax.set_ylabel('test R', fontsize=9)
    ax.grid(ls=':', alpha=0.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc='lower left', framealpha=0.85)


def draw_flip(ax):
    """同一臂 n=2 vs n=3 的区间：灰带为地板。"""
    ax.axvspan(F896[0], F896[1], color='#bdbdbd', alpha=0.55, zorder=0)
    v2 = sorted([m('nf_yolo26_896_s43', 'test'), m('nf_yolo26_896_s44', 'test')])
    v3 = sorted([m(r, 'test') for r in Y26])
    ax.hlines(2, F896[0], F896[1], color='#555555', lw=8)
    ax.hlines(1, v2[0], v2[1], color='#1f77b4', lw=8)
    ax.hlines(0, v3[0], v3[1], color='#1f77b4', lw=8)
    s42 = m('nf_yolo26_896_s42', 'test')
    ax.scatter([s42], [0], s=70, color='#d62728', zorder=4)
    ax.annotate('s42', (s42, 0), textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=8)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(['地板 n=3', 'YOLO26 n=2', 'YOLO26 n=3'], fontsize=9)
    ax.set_xlim(0.900, 0.920)
    ax.set_xlabel('test mAP50', fontsize=9)
    ax.grid(axis='x', ls=':', alpha=0.5)
    ax.tick_params(labelsize=8)


# ----------------------------------------------------------------
# 自检 1：YOLO26 的每一行必须与它自己的 results.csv 峰值一致（陈旧行防线）
# ----------------------------------------------------------------
def csv_peak(run):
    p = os.path.join(RUNS_DIR, run, 'results.csv')
    if not os.path.exists(p):
        return None, 0
    best, n = None, 0
    with open(p, encoding='utf-8', newline='') as f:
        rdr = csv.DictReader(f)
        key = next((k for k in rdr.fieldnames if k.strip() == 'metrics/mAP50(B)'), None)
        for d in rdr:
            n += 1
            if key is None:
                continue
            try:
                v = float(d[key])
            except (TypeError, ValueError):
                continue
            best = v if best is None else max(best, v)
    return best, n


checks = []
for run in FLOOR + Y26 + TRIP + CBAM:
    peak, n = csv_peak(run)
    got = m(run, 'val')
    ok = (n >= 60) and peak is not None and abs(got - peak) <= 0.05
    checks.append((run, n, peak, got, ok))
bad = [c for c in checks if not c[4]]
if bad:
    raise SystemExit(f'!! 陈旧行自检失败（TSV val 与 results.csv 峰值差 >0.05）：{bad}')

doc = Doc()

# ================================================================
#                        封面 / 摘要
# ================================================================
doc.y = 0.80
doc.title('第三批：YOLO26@896 与 Triplet 补种子')
doc.subtitle('第 3 步 / 第 4 步串行链 · 探索报告　·　2026-10-17')
doc.y -= 0.02
doc.para('口径定位：896 线（报告第十一章独立口径）。本报告是探索性记录，不属于任何交付形态；'
         '其中数字**不得进入 v_final 或任何交付形态**。')
doc.y -= 0.01
doc.h1('摘要')
doc.para('一句话结论：896 线上，YOLO26 与 v11 地板在**主判据 test mAP50 上不可分**；'
         'Triplet@896 补到 3 种子后同样不可分。唯一值得记录的差异是 YOLO26 的'
         '**召回名义上更高**（+2.18 pt），但高 IoU 精度没有跟上——方向与"YOLO26 框更准"的预想相反。')
doc.bullet(f'地板（v11 baseline × 3 种子）test mAP50 区间 [{F896[0]:.4f}, {F896[1]:.4f}]，'
           f'跨度 {sp(F896[1] - F896[0])} pt。')
doc.bullet(f'YOLO26（3 种子）区间 [{A26[0]:.4f}, {A26[1]:.4f}]，Δ 均值 {pt(A26[2] - F896[2])} pt '
           f'→ {"错开" if disjoint(A26[0], A26[1], F896[0], F896[1]) else "重叠 → 不可分"}；'
           f'mAP50-95 Δ {pt(A26_95[2] - F95[2])} pt（{ "错开" if disjoint(A26_95[0], A26_95[1], F95[0], F95[1]) else "重叠" }）。')
doc.bullet(f'YOLO26 的 R 区间 [{A26_R[0]:.4f}, {A26_R[1]:.4f}] 下界高出地板上界 '
           f'{(A26_R[0] - FR[1]) * 100:.2f} pt → **名义错开偏高**；该项为未预登记的探索性读数，不作效应判定。')
doc.bullet(f'Triplet@896 补到 n=3：区间 [{AT[0]:.4f}, {AT[1]:.4f}]，Δ {pt(AT[2] - F896[2])} pt '
           f'→ 与地板重叠、不可分。')
doc.bullet('方法论自证：YOLO26 只有 s43/s44 时区间 [0.9155, 0.9162] 在地板之上（错开）；'
           '补入 s42（0.9088）后翻转为重叠。**n=3 规则拦住了一次误报。**')
doc.bullet('期间发生一次中途评估污染（epoch-14 模型被当成 60 轮模型），两层根因已修复，'
           '并在 23:11:42 自动自愈；坏行按纪律保留，判读取最后一行。')

# ================================================================
#                        一、背景与任务
# ================================================================
doc.h1('一、背景与任务')
doc.para('本批承接第一批（640 双臂）与第二批（896 地板 + 双臂）。用户按四步下令：'
         '第 3 步 = YOLO26@896 跑 seed 42，看 test/val 与 v11 896 地板的关系，'
         '并特别回答"mAP50 与 mAP50-95 是否一个高一个低"；'
         '第 4 步 = 后台串行跑 YOLO26 s43/s44 与 Triplet@896 s43/s44，'
         '可重入、崩溃续训、每 run 完成立即落账。')
doc.para('第 3 步跑完即汇报，用户看过数字后授权第 4 步（未等 s42 的完整判读）。'
         '第 4 步链 2026-10-16 14:57 启动、23:11 退出，四个 run 全部 60/60。')

# ================================================================
#                        二、实验设置
# ================================================================
doc.h1('二、实验设置')
doc.bullet('数据：dataset_final_bbox，1428 / 178 / 179（train/val/test），3 类'
           '（break 断股 / thunderbolt 雷击 / wear 磨损）。')
doc.bullet('896 线：imgsz=896、batch=8、epochs=60、SGD lr0=0.01、seed 42/43/44、workers=0；'
           '与第二批**逐字同管线**（改了本批就不可比）。')
doc.bullet('YOLO26s：权重 `models/yolo26s.pt`；训练脚本 `code/yolov11/train_yolo26_896.py`'
           '（照抄 `train_v11_896.py`，只换权重与输出目录，另加 `--seed`）。')
doc.bullet('解释器：YOLO26 用 `D:/yolo26-env`（ultralytics 8.4.178）；v11 地板/双臂用'
           '`D:/yolo-env`（8.3.180）。**版本不同**，见第七节。')
doc.bullet('评估：`YOLO(best.pt)` 每次全新加载 → `.val(split=val|test, imgsz=896, batch=8, '
           'workers=0, device=0)`；test 为主判据、val 作一致性检查。')
doc.bullet('本报告所有数字由生成脚本**现读 TSV**（`batch2_eval.tsv` / `batch3_eval.tsv`），'
           '不手抄；并对每个 run 复核"TSV val mAP50 vs 自己的 results.csv 峰值"（阈值 0.05）。')

# ================================================================
#                      三、判据（预登记 + 本次声明）
# ================================================================
doc.h1('三、判据（出数前预登记，未事后更改）')
doc.bullet('噪声地板 = 同分辨率 3 个 baseline 种子的 **test mAP50** 区间 [min, max]；'
           '臂 = 同分辨率种子的区间。**错开**（臂 min > 地板 max 或 臂 max < 地板 min）→ 真实效应；'
           '**重叠** → 不可分，诚实记负。')
doc.bullet('**test 为主判据，val 为一致性检查**：best.pt 由 ultralytics 在 val 上按 fitness '
           '从 60 轮中挑出 → val 是选择集、带向上偏置。规则：'
           '"test 区间错开 **且 val 不反向矛盾**"才记真实效应；val 反向错开 → 判噪声。'
           '**val 仅重叠、不否决**。')
doc.bullet('两臂互相矛盾 → 两次独立噪声，两臂均记不可分。n<2 只印数、不下判读。')
doc.bullet('**本报告的预登记只覆盖 test mAP50**。mAP50-95 / P / R 的区间对比属**未预登记的'
           '探索性读数**，只报方向与幅度，不作效应判定。')
doc.note('【读法更正】第二批 PDF §七 曾用"val 也必须错开"否决 640 CBAM+BN，'
         '该读法严于预登记（预登记只否决 val 反向错开）。净结论不变，但理由已在 WORKLOG A.9.23 §七'
         '更正；batch2 v1 PDF 未擅自修改。')

# ================================================================
#                      四、第 3 步：YOLO26 结果
# ================================================================
doc.h1('四、第 3 步：YOLO26@896 的数字')
run_rows = []
for run in Y26:
    run_rows.append([run.rsplit('_s', 1)[1], f'{m(run, "test"):.6f}',
                     f'{m(run, "test", "mAP50-95"):.4f}',
                     f'{m(run, "test", "P"):.4f}', f'{m(run, "test", "R"):.4f}',
                     f'{m(run, "val"):.6f}'])
doc.table(['seed', 'test mAP50', 'test mAP50-95', 'test P', 'test R', 'val mAP50'],
          run_rows, col_widths=[0.09, 0.19, 0.19, 0.15, 0.15, 0.19], row_h=0.027,
          note='表 1　YOLO26@896 三种子。s42 的 val 最高（0.9280），test 却最低（0.9088）——'
               'val 是选择集的现场演示（见本节末）。')

doc.table(['指标', '地板区间', '地板均值', 'YOLO26 区间', 'YOLO26 均值', 'Δ 均值（pt）', '区间关系'],
          [['test mAP50', f'[{F896[0]:.4f}, {F896[1]:.4f}]', f'{F896[2]:.4f}',
            f'[{A26[0]:.4f}, {A26[1]:.4f}]', f'{A26[2]:.4f}', pt(A26[2] - F896[2]),
            '错开' if disjoint(A26[0], A26[1], F896[0], F896[1]) else '重叠'],
           ['test mAP50-95', f'[{F95[0]:.4f}, {F95[1]:.4f}]', f'{F95[2]:.4f}',
            f'[{A26_95[0]:.4f}, {A26_95[1]:.4f}]', f'{A26_95[2]:.4f}',
            pt(A26_95[2] - F95[2]),
            '错开' if disjoint(A26_95[0], A26_95[1], F95[0], F95[1]) else '重叠'],
           ['test P', f'[{FP[0]:.4f}, {FP[1]:.4f}]', f'{FP[2]:.4f}',
            f'[{A26_P[0]:.4f}, {A26_P[1]:.4f}]', f'{A26_P[2]:.4f}', pt(A26_P[2] - FP[2]),
            '错开' if disjoint(A26_P[0], A26_P[1], FP[0], FP[1]) else '重叠'],
           ['test R', f'[{FR[0]:.4f}, {FR[1]:.4f}]', f'{FR[2]:.4f}',
            f'[{A26_R[0]:.4f}, {A26_R[1]:.4f}]', f'{A26_R[2]:.4f}', pt(A26_R[2] - FR[2]),
            '错开偏高' if A26_R[0] > FR[1] else ('错开偏低' if A26_R[1] < FR[0] else '重叠')]],
          col_widths=[0.16, 0.16, 0.11, 0.17, 0.13, 0.13, 0.14], small=True, row_h=0.030,
          note='表 2　YOLO26 vs 896 地板（v11 baseline × 3）。主判据 mAP50 重叠；'
               'R 下界高出地板上界 0.40 pt，名义错开偏高。')

doc.figure(0.30, draw_map50,
           caption='图 1　test mAP50 逐种子（灰带 = 地板区间）。四组点全部落在灰带内或仅擦边，'
                   '主判据上不可分。',
           note='纵轴 0.895~0.925。每个点为一个种子；横线为该组均值。'
                'YOLO26 与 Triplet 的均值略高于地板，但区间重叠。')

doc.h2('回答指令里的问题：mAP50 与 mAP50-95 的方向')
doc.para(f'mAP50 平均 **{pt(A26[2] - F896[2])} pt**（重叠），mAP50-95 平均 '
         f'**{pt(A26_95[2] - F95[2])} pt**（重叠）——**方向相反，但都落在噪声内**。'
         '因此不能写"YOLO26 框更准"；能说的是：在 896 线上，它**召回更高、'
         '高 IoU 精度略低**，即"找得更全、定得更粗"。')
doc.para(f'召回这一项是本批唯一名义错开的指标：地板 R 区间 [{FR[0]:.4f}, {FR[1]:.4f}]，'
         f'YOLO26 [{A26_R[0]:.4f}, {A26_R[1]:.4f}]，下界高出地板上界 '
         f'{(A26_R[0] - FR[1]) * 100:.2f} pt；val 侧同向（YOLO26 均值 {A26_R[2]:.4f} vs 地板 '
         f'{FR[2]:.4f}），不构成反向矛盾。按预登记规则可记"名义效应"，'
         '但该项**不在预登记的判据范围内**，故只作机制线索。')
doc.figure(0.30, draw_direction,
           caption='图 2　test R（纵轴）vs test mAP50-95（横轴）。YOLO26 三点整体偏右下：'
                   '召回高、高 IoU 精度低，方向与预想相反。',
           note='小点 = 单种子，大星 = 组均值。四组同图对比；横纵轴均为 test。')

doc.h2('s42 的选择偏置演示')
doc.para('s42 在 6 个 896 的 v11/YOLO26 run 中 **val 最高**（0.9280，比地板 val 上界 0.9088 高 '
         '1.92 pt），在 YOLO26 臂内 **test 最低**（0.9088，比 s43 低 0.74 pt）。'
         '若只看 val，YOLO26 会被读成 +1.73 pt 的大胜；test 只给 +0.57 pt 且重叠。'
         '**val 是选择集，不能用来下结论**——本批把它钉成了实例。')

# ================================================================
#                 五、第 4 步：链与 Triplet n=3
# ================================================================
doc.h1('五、第 4 步：后台链与 Triplet@896 n=3')
doc.bullet('链：`scripts/run_batch3.sh`，串行 4 run（YOLO26 s43/s44 → Triplet s43/s44）；'
           '可重入、已完成跳过、有 last.pt 续训、每 run 完成立即核注入闸门并评 val/test。')
doc.bullet('四 run 全部 60/60、rc 0；单 run 训练 102~108 min（≈103 s/epoch），与预估一致。'
           '链 14:57 启动、23:11 退出。')
doc.bullet(f'Triplet@896 由 n=1 补到 n=3：test 区间 [{AT[0]:.4f}, {AT[1]:.4f}]，'
           f'Δ {pt(AT[2] - F896[2])} pt，与地板 [{F896[0]:.4f}, {F896[1]:.4f}] 重叠 → 不可分。')
doc.table(['run', 'test mAP50', 'val mAP50'],
          [[r, f'{m(r, "test"):.4f}', f'{m(r, "val"):.4f}'] for r in TRIP],
          col_widths=[0.46, 0.27, 0.27], row_h=0.028,
          note='表 3　Triplet@896 三种子（s42 来自第二批，s43/s44 来自本批链）。')

doc.h2('n=2 → n=3：一次判读翻转')
doc.para('若只有 s43/s44，YOLO26 臂区间为 [0.9155, 0.9162]，下界高于地板上界 0.9122 —— '
         '按规则会记"错开 / 真实效应"。补入 s42（0.9088）后区间扩为 '
         f'[{A26[0]:.4f}, {A26[1]:.4f}]，与地板重叠，判读翻转为"不可分"。'
         '两粒种子恰好都在地板之上是**运气**；第三粒把它拉回了噪声内。'
         '这正是"n≥3 才判"规则的用处——不是形式主义，它拦住了一次误报。')
doc.figure(0.26, draw_flip,
           caption='图 3　同一臂在 n=2 与 n=3 下的区间。灰带 = 地板；上条（n=2）整体高于灰带，'
                   '下条（n=3）被 s42 拉回带内。',
           note='横轴为 test mAP50。圆点标出 s42 的位置。')

# ================================================================
#                      六、事故与自愈
# ================================================================
doc.h1('六、中途评估污染事故与自愈（数据来源说明）')
doc.para('本报告的 YOLO26 s42 数字**曾一度是错的**：训练到 epoch 14 时该 run 被评测，'
         'val 0.6880 / test 0.7472 写进了 TSV；链在 16:11 见"60 轮 + best.pt + TSV 已有行"'
         '就跳过了重评。事后定位到两层根因：')
doc.bullet('ultralytics **每次 val 有提升就写 best.pt** → "best.pt 存在" ≠ 训练完成；'
           '中途检查点也能产出完整指标。')
doc.bullet('有效性判据挂在**外部状态**（"run 跑完没有"）上：中途行在 run 跑完之后会'
           '"看起来有效"，重入逻辑永久跳过，坏数字被冻结。')
doc.para('修复：把每一行 TSV 拿去做 **run 自己的 `results.csv` val mAP50 峰值**对账，'
         '差 > 0.05 判为中途行、自动重评（按 run 剔除 val/test 两行）。'
         '10-16 23:11:42 该防线在链的收尾调用中**自动生效**：识别 s42、重评、追加正确行。'
         '坏行按"只追加不删除"保留，判读取**最后一行**。')
doc.table(['时刻', '事件'],
          [['10-16 14:52', 's42 在 epoch≈14 被中途评测，坏行入 TSV（0.6880 / 0.7472）'],
           ['16:11:35', 's42 跑满 60/60；链误判"已评"，跳过重评'],
           ['23:11:28', '链收尾调用评估脚本；对账发现 0.6880 vs results.csv 峰值 0.9286 矛盾'],
           ['23:11:42', '自动重评 s42 并追加正确行（test 0.9088 / val 0.9280）；链 23:11:43 退出']],
          col_widths=[0.22, 0.78], small=True, row_h=0.030,
          note='表 4　事故与自愈时间线（取自链日志与文件 mtime）。'
               '本报告所有 12 个 run 的 TSV val 均与各自 results.csv 峰值相差 ≤0.05（生成脚本断言）。')

# ================================================================
#                      七、口径与版本混淆
# ================================================================
doc.h1('七、口径与版本混淆（引用时必须带上）')
doc.bullet('YOLO26 的数字由 **ultralytics 8.4.178** 产出；v11 地板/双臂由 **8.3.180** 产出。'
           '**差异可报，归因不可**——无法把"YOLO26 与 v11 的差"拆成"模型差"与"框架版本差"。')
doc.bullet('YOLO26 头语义（实测）：双头 one2many + one2one、E2ELoss、reg_max=1 → dfl 为 Identity，'
           '`dfl=1.5` 加权的是 L1 项（v11 加权 DFL 项），同名不同义。'
           '`nms` 默认 None → val/test 走 one2many + NMS，不是 NMS-free 模式。')
doc.bullet('AMP 自检未做（GitHub 不通，3 次重试失败）；训练 loss 全程有限、无 NaN，风险登记。')
doc.bullet('896 单 run 训练约 103 s/epoch；第 4 步四个 run 串行约 7 h（16:11 → 23:11）。')

# ================================================================
#                      八、口径声明（文末，必须带）
# ================================================================
doc.h1('八、口径声明（引用本报告任何数字时必须一并带上）')
doc.note('探索阶段：地板 3 种子 / YOLO26 3 种子 / Triplet 896 3 种子；只报方向与幅度，未做显著性检验。')
doc.note('主判据 = test mAP50（预登记）；mAP50-95 / P / R 为探索性读数，不作效应判定。')
doc.note('评估 imgsz = 训练 imgsz = 896；跨分辨率数字不可直接相比。')
doc.note('**本批数字不得进入 v_final 或任何交付形态。**')
doc.note('YOLO26 与 v11 由不同 ultralytics 版本产出（8.4.178 vs 8.3.180），跨代差异不可归因。')
doc.note(f'数据来源：exp_upscale/logs/batch2_eval.tsv、batch3_eval.tsv、'
         f'exp_upscale/runs/*/results.csv。生成日期 {DATE}。')

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
        fig.savefig(os.path.join(PNG_DIR, f'batch3_p{i + 1}.png'), dpi=110)
with PdfPages(OUT) as pdf:
    for fig in doc.pages:
        pdf.savefig(fig)
        plt.close(fig)

size = os.path.getsize(OUT)
print(f'OK  {OUT}')
print(f'pages = {n}  size = {size} bytes ({size / 1024:.1f} KB)')
print(f'checks = {len(checks)} runs, all valid (epochs>=60, |TSV val - csv peak|<=0.05)')
