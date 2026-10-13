# -*- coding: utf-8 -*-
"""生成《基于 YOLOv11+CBAM 的输电线路缺陷检测算法》实验报告 v_final
（四套对比 + 896 独立口径 + 泄漏核查 + 推荐模型综合对比）。

四套 = 320 SGD / 320 AdamW / 640 SGD / 640 AdamW
另加：896×896 独立一套（口径不与 640 混算，见第十一章）；数据泄漏核查（第十二章）
数据来源（全部来自 logs/，不做任何重新估算）：
  - 320 SGD   test : logs/eval_test_sgd.log        （7 组，4 位小数）
  - 320 AdamW test : logs/eval_test_adamw.log      （7 组，4 位小数）
  - 640 SGD   test : logs/train_640_all.log（6 组）+ logs/eval_test_640.log（baseline）
  - 640 AdamW test : logs/train_640_adamw_all.log  （5 组，本轮到 v12 两组未跑）
  - 推理 ms/FPS    : logs/fps_benchmark.txt        （端到端、FP16、batch=1、轮转 4 轮）
  - 参数量         : logs/params_all.txt + runs/wire_v11_v3/weights（v11 体系）
  - 训练曲线       : runs/wire_v11_v3/results.csv、runs/wire_v11_640/results.csv
  - 896 一套       : logs/eval_896.log（v11 与 v5 逐类块；896 无 fps 基准）
  - 泄漏/选择口径  : WORKLOG.md 钉死口径 + logs/leak_eval.txt、logs/p0f_permutation.txt

口径说明（写进报告正文/表注，勿删）：
  1. v12 的 640 一套取 wire_v12_640（在 640 下训练），不用 wire_v12s_v4 在 640 下推理的结果。
  2. 推理 ms/FPS 只与「架构 + 输入尺寸」有关，与优化器无关；640 AdamW 一套沿用同架构同尺寸的
     SGD 组实测值，320 AdamW 一套沿用 320 组实测值。
  3. 未单测的 CBAM 组在推理列记「—」；其开销与同架构基线之差小于轮间波动
     （v11+CBAM(neck)@320 实测 3.348 ms vs 基线 3.482 ms，而基线自身轮间波动 ±0.19 ms）。
  4. v5 的推理耗时由 CPU 侧主导，320 与 640 实测几乎相同（10.280 / 10.220 ms），
     其 FPS 不代表框架的 GPU 算力差距，跨框架比较须注明。
  5. YOLOv5 官方 val.py 只输出 3 位小数，故 v5 各组的指标保留 3 位。
  6. 参数量为 nc=3 产物口径；v5 用 attempt_load(fuse=False)（7,027,720），
     融合 BN 后为 7,018,216，与 v11 的读法不同口径。

输出：D:/2/report/基于YOLOv11+CBAM的缺陷检测算法_v_final.pdf
"""
import csv
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

OUT = 'D:/2/report/基于YOLOv11+CBAM的缺陷检测算法_v_final.pdf'
RUNS = 'D:/2/runs'
DATE = '2026-10-13'

# v5-vs-v11 的 lrf 不对称说明 —— 第二章与第十一章共用一个常量，改一处即两处同步（2026-10-13 用户指定）
LRF_NOTE = ('v5 与 v11 的 lrf 不同（v5 lrf=0.01、cos_lr=false，线性衰减；'
            'v11 lrf=0.005、cos_lr=true，余弦衰减），该不对称在 320 / 640 / 896 '
            '各套中逐字相同，在本实验范围内不影响分辨率相关结论')
LRF_TAIL = '；但 v5-vs-v11 的量化比较须注明此不对称。'

# ---------------- 字体（微软雅黑） ----------------
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
fp_ax = FontProperties(fname=F_REG, size=8)

PAGE_W, PAGE_H = 8.27, 11.69          # A4
TOP, BOTTOM = 0.945, 0.065
X0, WIDTH = 0.10, 0.80
LINE = 0.0185
ROW = 0.028
DEBUG = bool(os.environ.get('REPORT_DEBUG'))

C_SGD320, C_ADAMW320, C_SGD640, C_ADAMW640 = '#4472c4', '#ed7d31', '#70ad47', '#ffc000'


# ---------------- 版面容器 ----------------
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
        """图题居中；过长时自动折行（否则居中单行会被页边裁掉首尾字）。"""
        for ln in wrap_cjk(text, 74):
            self._ensure(LINE)
            self.ax.text(0.5, self.y, ln, ha='center', va='top',
                         fontproperties=fp_note, color='#333333',
                         transform=self.ax.transAxes)
            self.y -= LINE * 0.95
        self.y -= 0.006

    def figure(self, height, draw, caption, note=None, tick_pad=0.040, title_pad=0.026):
        """在当前页预留 height 高度画一张图，随后接图题/表注。

        tick_pad ：把 axes 底边上抬，留给 x 轴刻度标签；
        title_pad：把 axes 顶边下压，留给轴上方的图标题。
        两者都是必需的——刻度文字在图下方、轴标题在图上方，都会越出 axes 本身，
        不预留就会与相邻的正文互相压字（整块占位仍严格等于 height）。
        """
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


def read_results(name):
    """读 results.csv，返回 {列名: [值...]}；epoch 统一为 1-based。"""
    path = os.path.join(RUNS, name, 'results.csv')
    rows = [{k.strip(): v for k, v in r.items()}
            for r in csv.DictReader(open(path, encoding='utf-8-sig'))]
    cols = list(rows[0].keys())
    epochs, raw = [], {c: [] for c in cols}
    for r in rows:
        try:
            epochs.append(int(float(r['epoch'])))
        except (ValueError, KeyError):
            continue
        for c in cols:
            raw[c].append(r[c])
    off = 0 if (epochs and min(epochs) >= 1) else 1
    out = {}
    for c in cols:
        if c == 'epoch':
            out[c] = [e + off for e in epochs]
            continue
        vals = []
        for v in raw[c]:
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(float('nan'))
        out[c] = vals
    return out


# ================================================================
#                     数据（全部来自 logs/）
# ================================================================
# 配置短名 → 显示名
NAME = {
    'v11': 'v11 (baseline)',
    'v11_bb': 'v11+CBAM(backbone)',
    'v11_neck': 'v11+CBAM(neck_sppf)',
    'v11_both': 'v11+CBAM(both)',
    'v12': 'v12 (baseline)',
    'v12_cbam': 'v12+CBAM',
    'v5': 'v5 (YOLOv5s)†',
}
ORDER = ['v11', 'v11_bb', 'v11_neck', 'v11_both', 'v12', 'v12_cbam', 'v5']

# 推理性能（logs/fps_benchmark.txt；只与架构+尺寸有关，与优化器无关）
PERF = {
    'v11':      ('9,428,953', '3.482', '287.2', '6.482', '154.3'),
    'v11_bb':   ('9,480,545', '—', '—', '—', '—'),
    'v11_neck': ('9,461,819', '3.348', '298.7', '6.484', '154.2'),
    'v11_both': ('9,513,411', '—', '—', '—', '—'),
    'v12':      ('9,254,297', '2.689', '371.9', '6.512', '153.6'),
    'v12_cbam': ('9,314,179', '—', '—', '—', '—'),
    'v5':       ('7,027,720', '10.280', '97.3', '10.220', '97.8'),
}

# (key, P, R, mAP50, mAP50-95, break, thunderbolt, wear)
S320_SGD = [
    ('v11_neck', .8814, .7157, .7939, .3970, .558, .964, .860),
    ('v11',      .7931, .7056, .7787, .4035, .560, .954, .822),
    ('v12',      .8768, .6864, .7611, .3798, .524, .943, .816),
    ('v11_both', .7932, .6739, .7559, .3871, .522, .973, .773),
    ('v12_cbam', .8384, .6663, .7435, .3627, .482, .935, .814),
    ('v11_bb',   .8561, .6140, .7360, .3548, .552, .948, .707),
    ('v5',       .799,  .669,  .721,  .345,  .556, .937, .671),
]
S320_ADAMW = [
    ('v11_neck', .8270, .6955, .7908, .3964, .596, .973, .803),
    ('v12_cbam', .8653, .6794, .7824, .3920, .558, .941, .848),
    ('v11',      .8152, .7123, .7817, .3940, .577, .938, .830),
    ('v12',      .8707, .6924, .7785, .3906, .525, .950, .860),
    ('v11_both', .8603, .6651, .7731, .3839, .575, .949, .796),
    ('v5',       .779,  .729,  .755,  .322,  .533, .963, .769),
    ('v11_bb',   .7762, .6407, .7418, .3759, .562, .951, .712),
]
S640_SGD = [
    ('v5',       .917,  .857,  .907,  .500,  .831, .969, .921),
    ('v11_bb',   .9368, .8097, .9061, .5207, .853, .969, .896),
    ('v11_neck', .9080, .8170, .9016, .5192, .827, .973, .904),
    ('v11',      .8887, .8264, .8992, .5286, .811, .969, .918),
    ('v11_both', .8746, .8284, .8960, .5156, .801, .977, .910),
    ('v12_cbam', .8855, .8310, .8874, .5071, .796, .970, .896),
    ('v12',      .8648, .8107, .8682, .4909, .754, .961, .890),
]
S640_ADAMW = [
    ('v11_both', .8872, .8067, .8998, .5072, .835, .966, .899),
    ('v5',       .908,  .822,  .891,  .494,  .787, .972, .913),
    ('v11_bb',   .8973, .8154, .8898, .5073, .809, .968, .892),
    ('v11_neck', .9011, .8047, .8879, .5096, .834, .941, .889),
    ('v11',      .8940, .8076, .8873, .5140, .791, .968, .903),
]

# 896×896 独立一套（口径不与 640 混算）：数据源 logs/eval_896.log
# (key, P, R, mAP50, mAP50-95, break, thunderbolt, wear) —— 均按源精度取字符串
S896 = [
    ('v5',  '.902',  '.877',  '.931',  '.529',  '.862', '.973', '.957'),
    ('v11', '.9401', '.8212', '.9148', '.5362', '.870', '.962', '.912'),
]
# 896 无推理基准（logs/fps_benchmark.txt 只测 320/640）；参数量沿用四套同口径
S896_PARAMS = {'v11': '9,428,953', 'v5': '7,027,720'}

SETS = [
    ('320×320 + SGD',   S320_SGD,    '320', C_SGD320),
    ('320×320 + AdamW', S320_ADAMW,  '320', C_ADAMW320),
    ('640×640 + SGD',   S640_SGD,    '640', C_SGD640),
    ('640×640 + AdamW', S640_ADAMW,  '640', C_ADAMW640),
]


def lookup(rows, key):
    for r in rows:
        if r[0] == key:
            return r
    return None


def mean_set(rows, idx):
    return sum(r[idx] for r in rows) / len(rows)


def fmt_v(x, key=None):
    """v5 只给 3 位（源精度 3 位），其余一律 4 位。"""
    return f'{x:.3f}' if key == 'v5' else f'{x:.4f}'


# ================================================================
#                           绘图工具
# ================================================================
SHORT = {'v11': 'v11', 'v11_bb': 'v11+CBAM\n(backbone)', 'v11_neck': 'v11+CBAM\n(neck_sppf)',
         'v11_both': 'v11+CBAM\n(both)', 'v12': 'v12', 'v12_cbam': 'v12+CBAM',
         'v5': 'v5(YOLOv5s)'}


def vals_by_config(rows):
    """按 ORDER 取某套的 mAP50，缺的组返回 None。"""
    d = {r[0]: r for r in rows}
    return [d[k][3] if k in d else None for k in ORDER]


def _bars(ax, vals, xcenters, width, color, label, fs=6.3, dy=0.006, fmt='{:.3f}'):
    ax.bar(xcenters, [v if v is not None else 0 for v in vals], width,
           label=label, color=color, edgecolor='white', linewidth=0.5)
    for a, v in zip(xcenters, vals):
        if v is not None:
            ax.text(a, v + dy, fmt.format(v), ha='center', va='bottom', fontsize=fs)


def _style(ax, title, ylabel, ylim):
    ax.set_ylabel(ylabel, fontproperties=fp_ax)
    ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontproperties=fp_h2, pad=5)
    for lb in ax.get_xticklabels():
        lb.set_fontproperties(fp_ax)


def draw_fig1(ax):
    """图 1：四套 × 7 个模型配置的 test mAP50。"""
    x = np.arange(len(ORDER))
    m, w = len(SETS), 0.8 / len(SETS)
    for i, (setname, rows, _sz, col) in enumerate(SETS):
        xi = x - 0.4 + w * (i + 0.5)
        # 四个系列的数值标签逐级抬高，避免相邻柱的字互相压住
        _bars(ax, vals_by_config(rows), xi, w * 0.9, col, setname,
              fs=5.6, dy=0.006 + i * 0.021)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k] for k in ORDER], fontproperties=fp_ax)
    _style(ax, '四套设置的 test mAP50 对比（v12 在 640+AdamW 未跑，缺该柱）',
           'test mAP50', (0.60, 1.07))
    ax.legend(ncol=4, fontsize=7.0, framealpha=0.9, loc='upper center')


def draw_fig3(ax):
    """图 3：分辨率对比，每个配置两根柱（320 可得套均值 / 640 可得套均值）。"""
    x = np.arange(len(ORDER))
    v320, v640 = [], []
    for k in ORDER:
        a = [lookup(s, k)[3] for s in (S320_SGD, S320_ADAMW) if lookup(s, k)]
        b = [lookup(s, k)[3] for s in (S640_SGD, S640_ADAMW) if lookup(s, k)]
        v320.append(sum(a) / len(a) if a else None)
        v640.append(sum(b) / len(b) if b else None)
    _bars(ax, v320, x - 0.2, 0.36, C_SGD320, '320（两优化器均值）', fs=6.6)
    _bars(ax, v640, x + 0.2, 0.36, C_SGD640, '640（两优化器均值）', fs=6.6)
    for a, lo, hi in zip(x, v320, v640):
        if lo and hi:
            ax.annotate('', xy=(a + 0.2, hi - 0.004), xytext=(a - 0.2, lo + 0.004),
                        arrowprops=dict(arrowstyle='->', color='#c00000', lw=0.9))
            ax.text(a, max(lo, hi) + 0.030, f'+{hi - lo:.3f}', ha='center',
                    va='bottom', fontsize=6.6, color='#c00000')
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k] for k in ORDER], fontproperties=fp_ax)
    _style(ax, '分辨率对比：320 vs 640（红线为提升量）', 'test mAP50', (0.60, 1.12))
    ax.legend(ncol=2, fontsize=7.5, framealpha=0.9, loc='upper left')


def draw_fig4(ax):
    """图 5：四套在三类缺陷上的均值 mAP50 对比。"""
    cls = ['break 断股', 'thunderbolt 雷击', 'wear 磨损']
    x = np.arange(3)
    m, w = len(SETS), 0.8 / len(SETS)
    for i, (setname, rows, _sz, col) in enumerate(SETS):
        vals = [mean_set(rows, 5), mean_set(rows, 6), mean_set(rows, 7)]
        xi = x - 0.4 + w * (i + 0.5)
        _bars(ax, vals, xi, w * 0.9, col, setname)
    ax.set_xticks(x)
    ax.set_xticklabels(cls, fontproperties=fp_ax)
    _style(ax, '四套设置下各类别的 test mAP50（各套内均值）', 'test mAP50', (0.40, 1.14))
    ax.legend(ncol=4, fontsize=7.0, framealpha=0.9, loc='upper center')


def png_draw(path):
    """把已生成的 PNG（report/_cbam_attention/*.png）原样贴进预留的 axes。"""
    def _d(ax):
        ax.imshow(plt.imread(path))
        ax.axis('off')
    return _d


# ================================================================
#                              成稿
# ================================================================
doc = Doc()

doc.title('基于 YOLOv11+CBAM 的输电线路缺陷检测算法实验报告')
doc.subtitle('数据集 dataset_final_bbox（断股 break / 雷击 thunderbolt / 磨损 wear 3 类）')
doc.subtitle('四套对比：320 SGD / 320 AdamW / 640 SGD / 640 AdamW　·　'
             '附 896×896 独立对照与数据泄漏核查　·　' + DATE)

# ---------------- 一、数据集 ----------------
doc.h1('一、数据集')
doc.para('本实验面向架空输电线路缺陷检测任务，共检测 3 类缺陷：break（断股）、'
         'thunderbolt（雷击）、wear（磨损）。数据标注由 SAM（Segment Anything Model）'
         '自动生成分割掩膜，再将 polygon 标注转换为 bbox，形成检测数据集 '
         'dataset_final_bbox（task: detect），最后按 8:1:1 划分为训练集 1428 张、'
         '验证集 178 张、测试集 179 张（共 1785 张）。各划分的图片数与类别实例数见表 1。')
doc.table(['数据划分', '图片数', '断股 break', '雷击 thunderbolt', '磨损 wear', '实例总数'],
          [['训练集 train', '1428', '1014', '650', '788', '2452'],
           ['验证集 val', '178', '131', '80', '64', '275'],
           ['测试集 test', '179', '150', '69', '64', '283'],
           ['合计', '1785', '1295', '799', '916', '3010']],
          col_widths=[0.20, 0.14, 0.16, 0.19, 0.15, 0.16],
          note='表 1　数据集分布（图片数 / 各缺陷类别实例数 / 实例总数）')
doc.para('三类缺陷存在明显的尺度与形态差异：断股（break）为细长线状目标、'
         '与背景导线纹理高度相似，标注实例最多（1295 个）；雷击（thunderbolt）'
         '为局部熔痕、特征对比度高；磨损（wear）面积较大但边界模糊。'
         '这一差异在后续各类别指标中有直接反映。')
doc.para('四套实验均使用同一份数据、同一划分、同一 seed（42）、batch=8、训练 60 epoch，'
         '取 fitness 最优的 best.pt，在测试集（179 张图片、283 个标注框）上评估；'
         '两两之间只改变输入尺寸或优化器，其余对齐。')
doc.para('关于权重选择的口径：本报告一律报 fitness 最优的 best.pt（fitness = '
         '0.1×mAP50 + 0.9×mAP50-95），不报 last.pt。核对结果：best.pt 与 last.pt '
         '在 test 上的 mAP50 差 ≤0.55pt（wire_v5_640 为 1.2pt），即早停的净收益 ≈ 0；'
         'wire_v5_640 单独注明——该 run 的 last 优于 best 1.2pt，若按 last 报则 mAP50 为 0.919。'
         '之所以不按 test 结果回头改选 last，是因为「用测试集选权重」本身构成测试集泄漏，'
         '比压低 1.2pt 更严重；故全套统一报 best.pt。')
doc.para('早停判据的口径：以 val 早停为准时，best 与 last 的 test mAP50 差 |Δ| ≤0.55pt '
         '视为噪声，不触发对选权重流程的重估；仅当 |Δ| > 1pt 才触发。'
         '按此判据，本实验 9 个 run 中只有 wire_v5_640（1.2pt）越过阈值，其余全在噪声内。')
doc.note('注：模型权重按框架默认 fitness（0.1·mAP50 + 0.9·mAP50-95）选择。')

# ---------------- 二、四套对比总表 ----------------
doc.h1('二、四套对比总表（测试集）')
doc.para('四套 = 输入尺寸（320 / 640）× 优化器（SGD + lr0=0.01 / AdamW + lr0=0.001）'
         '的四种组合。每一套内含 YOLOv11s 基线及其 3 种 CBAM 插入位置、'
         'YOLOv12s 基线及 +CBAM、YOLOv5s 基线；其中 640 AdamW 一套只跑了 5 组'
         '（v11 四组 + v5），v12 两组本轮未跑，表中如实留空。')
doc.para('表 2~表 5 给出四套各自的完整结果，每套按 test mAP50 降序排列，'
         '同一行号只表示同一排名，不代表同一模型；同一模型的跨套对照见表 6。')

T2_COLS = ['模型配置', 'P', 'R', 'mAP50', 'mAP50-95', '推理 ms/张', 'FPS', '参数量']
T2_W = [0.255, 0.078, 0.078, 0.086, 0.098, 0.108, 0.078, 0.119]
T2_OFF = {'320': (1, 2), '640': (3, 4)}          # (ms, fps) 在 PERF 元组里的下标

for i, (setname, rows, sz, _c) in enumerate(SETS):
    body, mp, ms = [], [], []
    for r in rows:
        key = r[0]
        ms_i, fps_i = T2_OFF[sz]
        p = PERF[key]
        body.append([NAME[key], fmt_v(r[1], key), fmt_v(r[2], key),
                     fmt_v(r[3], key), fmt_v(r[4], key),
                     p[ms_i], p[fps_i], p[0]])
        mp.append(r[3])
        ms.append(r[4])
    body.append(['均值（%d 组）' % len(rows), '—', '—',
                 f'{sum(mp) / len(mp):.4f}', f'{sum(ms) / len(ms):.4f}', '—', '—', '—'])
    doc.table(T2_COLS, body, col_widths=T2_W, small=True, row_h=0.0235,
              left=0.055, width=0.905,
              note=f'表 {2 + i}　{setname}　测试集完整结果（best.pt；按 mAP50 降序）')

doc.note('表 2~表 5 通用表注：'
         '① 推理 ms/张 与 FPS 为端到端口径（读图 + letterbox + 归一化 + 前向 + NMS），'
         'RTX 4060 Laptop / FP16 / batch=1 / 同一张测试图，轮转 4 轮丢弃首轮取中位数，'
         '测量窗口内 SM 频率均为 2475~2490 MHz（详见 logs/fps_benchmark.txt）。'
         '② 推理只与「架构 + 输入尺寸」有关，与优化器无关，故两套 AdamW 沿用同架构同尺寸的实测值。'
         '③ 未单测的 CBAM 组记「—」：其推理开销与同架构基线之差低于轮间波动'
         '（v11+CBAM(neck)@320 实测 3.348 ms vs 基线 3.482 ms，而基线自身轮间波动 ±0.19 ms）。'
         '④ v12 的 640 一套取在 640 下训练的 wire_v12_640；其推理值由同架构产物 '
         'wire_v12s_v4@640 测得（参数量逐位一致，9,254,297）。'
         '⑤ 参数量为 nc=3 产物口径；v5 用 fuse=False 读法（7,027,720），'
         '融合 BN 后为 7,018,216，与 v11 的读法不同口径。'
         '⑥ † v5 的推理耗时由 CPU 侧主导（320 与 640 实测几乎相同，10.280 / 10.220 ms），'
         '其 FPS 不代表框架的 GPU 算力差距，跨框架比较时须注明。'
         '⑦ YOLOv5 官方 val.py 只输出 3 位小数，故 v5 各组保留 3 位。')

doc.para('四套的均值呈阶梯状：320 SGD 0.7557 < 320 AdamW 0.7719 << 640 SGD 0.8951，'
         '640 AdamW 0.8912（5 组口径）。640 两套的全部组都落在 0.868~0.907，'
         '而 320 两套的最好成绩只有 0.7939 与 0.7908。'
         '值得注意的是：640 下按 mAP50，YOLOv5s（0.907）略高于 YOLOv11s（0.8992）；'
         '但按 mAP50-95（定位精度），YOLOv11s（0.5286）高于 YOLOv5s（0.500）。'
         '综合定位精度、推理速度与架构代际，本报告推荐 YOLOv11s（见第十章综合对比表）。')
doc.para('注（v5-vs-v11 的口径）：640 下 v5 与 v11 的 mAP50 差 0.8pt，'
         '小于同一套内 v11 四组的组内极差 1.01pt（0.8960~0.9061），故只作方向性结论、不作量化。'
         '另：' + LRF_NOTE + LRF_TAIL)

# 表 6 同一模型四套对照（并入第二章）
doc.h2('2.5　同一模型在四套下的对照')
doc.para('为排除排名变化带来的干扰，表 6 按模型配置对齐四套结果，'
         '并给出 640（取同优化器的 640 一套）相对 320 的绝对提升。')
t6 = [['模型配置', '320 SGD', '320 AdamW', '640 SGD', '640 AdamW', 'Δ均值(640−320)']]
for key in ORDER:
    v320 = [lookup(S320_SGD, key), lookup(S320_ADAMW, key)]
    v640 = [lookup(S640_SGD, key), lookup(S640_ADAMW, key)]
    got320 = [x[3] for x in v320 if x]
    got640 = [x[3] for x in v640 if x]
    m320 = sum(got320) / len(got320)
    m640 = sum(got640) / len(got640) if got640 else None
    t6.append([NAME[key],
               fmt_v(v320[0][3], key) if v320[0] else '—',
               fmt_v(v320[1][3], key) if v320[1] else '—',
               fmt_v(v640[0][3], key) if v640[0] else '—',
               fmt_v(v640[1][3], key) if v640[1] else '—',
               (f'{m640 - m320:+.3f}' if key == 'v5' else f'{m640 - m320:+.4f}')
               if m640 else '—'])
doc.table(t6[0], t6[1:], col_widths=[0.30, 0.125, 0.135, 0.125, 0.135, 0.18],
          note='表 6　同一模型配置在四套训练设置下的 test mAP50（Δ 为两端可得值的均值之差，'
               '640 一侧含同优化器的 AdamW 值；v12 在 640 AdamW 未跑，故其 Δ 只由 SGD 一侧给出；'
               'v5 各值只有 3 位小数，其 Δ 末位有 ±0.001 不确定度）')
doc.para('全部 7 个配置在 640 下都取得提升：按表 6 的 Δ 列（两端可得套的均值之差）'
         '为 +0.098~+0.161；若两端都只取 SGD 一套，则幅度为 +0.107~+0.186。'
         '提升幅度与模型架构无单调关系，'
         '说明该收益主要来自输入分辨率对细长小目标的还原，而非某一架构的专属优势。'
         '注意 v12 的提升最小（+0.098）且其 640 一侧只有 SGD，'
         '该值不宜与其余配置的均值差直接比较。')
doc.figure(0.235, draw_fig1,
           '图 1　四套训练设置下各模型配置的 test mAP50 柱状图（横轴为模型配置，'
           '四色分别为四套设置；v12 在 640+AdamW 一套未跑；纵轴自 0.60 起，非零起点）')

# 表 7 优化器对比
doc.h1('三、优化器对比')
doc.para('在两种分辨率下分别比较 SGD（lr0=0.01，与 YOLOv5 官方配置一致）'
         '与 AdamW（lr0=0.001，AdamW 的常用量级）。学习率与优化器必须匹配：'
         '早期实验曾让 AdamW 沿用 lr0=0.01，属于错配，同一模型 test mAP50 仅 0.635，'
         '换 SGD+lr0=0.01 后升至 0.779（+0.144）；若在错配口径下比较优化器，'
         '会得出完全相反的结论。')
t7 = [['训练设置', '组数', '均值 mAP50', '均值 mAP50-95', '最优组 mAP50']]
for setname, rows, _sz, _c in SETS:
    best = max(rows, key=lambda r: r[3])
    t7.append([setname, str(len(rows)),
               f'{mean_set(rows, 3):.4f}', f'{mean_set(rows, 4):.4f}',
               f'{best[3]:.4f}（{NAME[best[0]].split(" (")[0]}）'])
doc.table(t7[0], t7[1:], col_widths=[0.26, 0.09, 0.16, 0.19, 0.30],
          note='表 7　四套的整体表现。注意 640 AdamW 只有 5 组（缺 v12 两组），'
               '其均值与 7 组口径不可直接比较，同构子集对照见表 8。')

t8 = [['可比子集', '320 SGD', '320 AdamW', '320 差', '640 SGD', '640 AdamW', '640 差']]
for label, keys in [('v11 四组（完全同构）', ['v11', 'v11_bb', 'v11_neck', 'v11_both']),
                    ('v11 四组 + v5（五组同构）', ['v11', 'v11_bb', 'v11_neck', 'v11_both', 'v5'])]:
    row = [label]
    for sz, a, b in (('320', S320_SGD, S320_ADAMW), ('640', S640_SGD, S640_ADAMW)):
        ma = sum(lookup(a, k)[3] for k in keys) / len(keys)
        mb = sum(lookup(b, k)[3] for k in keys) / len(keys)
        row += [f'{ma:.4f}', f'{mb:.4f}', f'{mb - ma:+.4f}']
    t8.append(row)
doc.table(t8[0], t8[1:], col_widths=[0.235, 0.115, 0.125, 0.11, 0.115, 0.125, 0.11],
          small=True, row_h=0.0245,
          note='表 8　同构子集下的优化器对照（消除 640 AdamW 缺 v12 造成的口径差）')
doc.para('优化器的优劣随分辨率反转：320 下 AdamW 更高（7 组均值 0.7719 vs 0.7557，+0.0162），'
         '而 640 下 AdamW 反而更低。由于 640 AdamW 缺 v12 两组，'
         '用完全同构的子集比对更可靠：v11 四组在 320 下 AdamW 高 +0.0057、'
         '在 640 下低 −0.0095；五组同构子集同样是 320 下 +0.0114、640 下 −0.0108。'
         '两种口径下方向一致地反转，说明 AdamW 的优势并不是稳定的规律，'
         '而是依赖分辨率（很可能与不同尺度下的有效学习率、梯度噪声水平有关）。'
         '无论哪个方向，优化器带来的差异都不足 0.02，远小于分辨率带来的 +0.14。')

# 图 2 优化器对比（两个面板：320 / 640）
if DEBUG:
    print(f'  [figure fig2] y={doc.y:.3f} need=0.26')
doc._ensure(0.26)
_y = doc.y - 0.26
for _j, (_tag, _a, _b) in enumerate((('320×320', S320_SGD, S320_ADAMW),
                                     ('640×640', S640_SGD, S640_ADAMW))):
    axo = doc.fig.add_axes([0.105 + _j * 0.465, _y + 0.055, 0.37, 0.185])
    _x = np.arange(len(ORDER))
    _va = [lookup(_a, k)[3] if lookup(_a, k) else None for k in ORDER]
    _vb = [lookup(_b, k)[3] if lookup(_b, k) else None for k in ORDER]
    _bars(axo, _va, _x - 0.19, 0.34, '#4472c4', 'SGD', fs=5.6)
    _bars(axo, _vb, _x + 0.19, 0.34, '#ed7d31', 'AdamW', fs=5.6)
    for _i, (_u, _v) in enumerate(zip(_va, _vb)):
        if _u and _v:
            axo.text(_i, min(_u, _v) + 0.052, f'{_v - _u:+.3f}', ha='center',
                     va='bottom', fontsize=6.0, color='#c00000')
    _style(axo, f'{_tag}：SGD vs AdamW', 'test mAP50', (0.60, 1.14))
    axo.set_xticks(_x)
    axo.set_xticklabels([SHORT[k].replace('\n', '') for k in ORDER],
                        fontproperties=fp_ax, rotation=32, ha='right', fontsize=7)
    axo.legend(ncol=2, fontsize=7.0, framealpha=0.9, loc='upper center')
doc.y = _y - 0.020
doc.caption('图 2　优化器对比（左 320×320，右 640×640；红色数字为 AdamW − SGD；纵轴自 0.60 起）')
doc.note('图 2 说明：320 下红字多为正（AdamW 领先），640 下红字转为负（SGD 领先），'
         '即优化器的相对优劣随分辨率反转；图中 AdamW 缺 v12 与 v12+CBAM 两组'
         '（每个面板各少两根柱）。')

# 表 9 分辨率对比
doc.h1('四、分辨率对比（320 vs 640）')
doc.para('这是本组实验最显著的结论。两两对照（同优化器下 320 → 640）：')
t9 = [['优化器', '320 均值 mAP50', '640 均值 mAP50', 'Δ mAP50',
       '320 均值 mAP50-95', '640 均值 mAP50-95', 'Δ mAP50-95']]
for opt, a, b in (('SGD', S320_SGD, S640_SGD), ('AdamW', S320_ADAMW, S640_ADAMW)):
    t9.append([opt, f'{mean_set(a, 3):.4f}', f'{mean_set(b, 3):.4f}',
               f'{mean_set(b, 3) - mean_set(a, 3):+.4f}',
               f'{mean_set(a, 4):.4f}', f'{mean_set(b, 4):.4f}',
               f'{mean_set(b, 4) - mean_set(a, 4):+.4f}'])
doc.table(t9[0], t9[1:], col_widths=[0.11, 0.155, 0.155, 0.12, 0.16, 0.16, 0.14],
          small=True, row_h=0.0245,
          note='表 9　输入分辨率对整体性能的影响（各套均值；'
               'SGD 两侧均 7 组可比，AdamW 一侧 320 为 7 组、640 为 5 组，'
               '为避免口径差，AdamW 行的 640 值按 320 同构的 5 组（v11 四组 + v5）计算）')
doc.para('输入尺寸由 320 提升到 640，测试集 mAP50 均值提升约 +0.14，'
         'mAP50-95 均值提升约 +0.13，最优组由 0.7939 升至 0.9070。'
         '该提升覆盖全部模型配置，无例外。代价是显存峰值由约 0.9 GB 升至约 3.7 GB、'
         '单组 60 epoch 训练由约 26 分钟升至约 72 分钟（约 2.8 倍）、'
         '推理耗时由 3.48 ms/张升至 6.48 ms/张（约 1.9 倍）。'
         '考虑到精度收益（+0.14）比优化器（±0.02）与注意力模块（±0.01）高一个数量级，'
         '提高输入分辨率是本任务性价比最高的优化手段。')
doc.figure(0.205, draw_fig3,
           '图 3　分辨率对比：320×320 vs 640×640（每个配置取该尺寸下可得套的均值；'
           '红色箭头与数字为 640 相对 320 的提升量）')

# 表 10 CBAM 消融
doc.h1('五、CBAM 消融实验')
doc.para('以 YOLOv11s 为基线，分别在骨干网络（backbone 第 2/4/6/8 层）、'
         '颈部与 SPPF（neck_sppf 第 9 层）以及两者同时插入 CBAM，'
         '在 320/640 × SGD/AdamW 四种设置下各做一组消融。'
         'CBAM 的注入采用子类化 DetectionTrainer 并重写 get_model() 的方式，'
         '在模型按 yaml 重建完成之后再注入；若在 model.train() 之前对模型对象注入，'
         '会被训练器的重建过程丢弃，属于无效注入（曾表现为 +CBAM 与 baseline 参数量完全相同、'
         '层数也完全相同）。参数量核验：baseline 9,428,953，backbone +51,592，'
         'neck_sppf +32,866，both +84,458，与注入层数成正比且逐位可复现。')
doc.figure(0.128,
           png_draw('D:/2/report/_cbam_attention/wire_v11_640_both_spatial.png'),
           '图 4　CBAM 空间注意力热力图（wire_v11_640_both；叠加在测试图上；'
           '5 层的框内/框外比均在 1.0 附近，0.957~1.110，即对缺陷无稳定偏好）',
           tick_pad=0.006, title_pad=0.006)
t10 = [['配置', '320 SGD', '320 AdamW', '640 SGD', '640 AdamW']]
base = 'v11'
for key, label in [('v11', 'v11 baseline'), ('v11_bb', '+ CBAM (backbone)'),
                   ('v11_neck', '+ CBAM (neck_sppf)'), ('v11_both', '+ CBAM (both)')]:
    row = [label]
    for _sn, rows, _sz, _c in SETS:
        r = lookup(rows, key)
        if key == base or r is None:
            row.append(f'{r[3]:.4f}' if r else '—')
        else:
            b = lookup(rows, base)
            row.append(f'{r[3]:.4f} ({r[3] - b[3]:+.4f})')
    t10.append(row)
doc.table(t10[0], t10[1:], col_widths=[0.26, 0.185, 0.185, 0.185, 0.185],
          note='表 10　CBAM 插入位置消融（test mAP50，括号内为相对同套 baseline 的差值；'
               '640 AdamW 一套未跑 v12，v11 四组齐全，不影响本表）')
doc.para('四种设置下 CBAM 都没有表现出稳定增益，差值方向随优化器与分辨率改变：')
doc.bullet('320 + SGD：仅 neck_sppf 为正（0.7939，+0.0152），backbone −0.0427、both −0.0228。')
doc.bullet('320 + AdamW：仅 neck_sppf 为正（0.7908，+0.0091），backbone −0.0399、both −0.0086。')
doc.bullet('640 + SGD：三处全部落在 ±0.007 以内（+0.0069 / +0.0024 / −0.0032），'
           '可视为无差异。')
doc.bullet('640 + AdamW：both 为正（0.8998，+0.0125），backbone +0.0025、neck_sppf +0.0006；'
           '而同一配置在 640 + SGD 下 both 是负值，方向相反。')
doc.para('此外，YOLOv12s 上 CBAM 的方向也不稳定：320 SGD 下由 0.7611 降至 0.7435（−0.0176）、'
         '320 AdamW 下由 0.7785 升至 0.7824（+0.0039）、640 SGD 下由 0.8682 升至 0.8874（+0.0192）。'
         '综合看，CBAM 的增益幅度随分辨率提高而被稀释（320 下最大 ±0.04，640 下最大 ±0.013），'
         '方向随优化器、分辨率与架构变化，不宜作为通用涨点手段。'
         '原因推测为：本任务目标尺度小、训练样本有限（仅 1428 张），'
         '骨干浅层的空间注意力容易放大背景导线纹理的响应，'
         '干扰细长线状目标的特征表达；而 640 输入下目标有效像素已充足，'
         '注意力机制可补充的信息本就不多。')

# ---------------- 七、各类别结果 ----------------
doc.h1('六、各类别结果')
doc.para('表 11 给出四套全部模型在三个类别上的 test mAP50（按套与优化器分组）。')
t11 = [['尺寸', '优化器', '模型配置', 'break 断股', 'thunderbolt 雷击', 'wear 磨损', '整体 mAP50']]
for setname, rows, sz, _c in SETS:
    opt = setname.split('+ ')[1]
    for r in rows:
        t11.append([sz, opt, NAME[r[0]].replace('†', ''), f'{r[5]:.3f}', f'{r[6]:.3f}',
                    f'{r[7]:.3f}', fmt_v(r[3], r[0])])
doc.table(t11[0], t11[1:], col_widths=[0.075, 0.105, 0.28, 0.13, 0.175, 0.12, 0.115],
          small=True, row_h=0.0215,
          note='表 11　四套测试集的分类别 mAP50（每套内按整体 mAP50 降序；'
               '640 AdamW 5 行，缺 v12 两组）')
doc.para('三类缺陷中，雷击（thunderbolt）检测效果最好，全部 26 行都在 0.935 以上——'
         '其局部熔痕特征对比度高、与背景区分度大；磨损（wear）次之，'
         '320 下 0.671~0.860、640 下 0.889~0.921，随分辨率明显改善且更集中。')
doc.para('最值得说明的是断股（break）：在 320 下它是唯一的性能瓶颈，'
         '14 行模型的 break mAP50 落在 0.482~0.596，其中 9 行低于 0.56，'
         '与同期雷击相差约 0.36~0.45；在 640 下各组升至 0.754~0.853，'
         '与雷击的差距收窄到约 0.11~0.21，且不再有个别组塌陷到 0.5 以下。'
         '断股是细长线状目标、像素占比低，断口处的单丝结构与正常导线纹理高度相似，'
         '320 输入下有效像素被进一步压缩；提高分辨率后其可用信息显著增加，'
         '瓶颈随之解除。即此前「断股是类别级瓶颈」的判断只在 320 下成立。')
doc.figure(0.200, draw_fig4,
           '图 5　各类别 test mAP50 对比（横轴为类别，四色为四套，各套内取均值）')

# ---------------- 七、训练曲线 ----------------
doc.h1('七、训练曲线')
doc.para('图 6、图 7 分别为 YOLOv11s 基线在 320×320（wire_v11_v3）与 '
         '640×640（wire_v11_640）下的 60 epoch 训练曲线，'
         '包含训练损失与验证集 mAP。两组仅输入尺寸不同，其余配置完全一致，可直接对比。')


def add_curves(name, cap):
    """一组的损失曲线 + mAP 曲线（上下两块）。

    整块高度 0.475 是这样分配的（全部以页面高度为单位的分数）：
      0.028  顶部留给第一张图的轴标题（不占就会压住上一段正文）
      0.150  第一张图（损失）
      0.065  第一张图的刻度 + xlabel + 第二张图的轴标题
      0.150  第二张图（mAP）
      0.038  第二张图的刻度 + xlabel
      0.044  图题（可能折成两行）
    """
    if DEBUG:
        print(f'  [curves {name}] y={doc.y:.3f} need=0.475')
    doc._ensure(0.478)
    H = 0.150
    y1 = doc.y - 0.028 - H
    ax1 = doc.fig.add_axes([0.135, y1, 0.78, H])
    d = read_results(name)
    for k, lab in zip(['train/box_loss', 'train/cls_loss', 'train/dfl_loss'],
                      ['box loss', 'cls loss', 'dfl loss']):
        ax1.plot(d['epoch'], d[k], linewidth=1.3, label=lab)
    ax1.set_title('训练损失曲线', fontproperties=fp_h2, pad=5)
    ax1.set_xlabel('epoch', fontproperties=fp_note)
    ax1.set_ylabel('loss', fontproperties=fp_note)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=8)
    ax1.legend(fontsize=8, framealpha=0.9, prop=fp_ax)

    y2 = y1 - 0.215
    ax2 = doc.fig.add_axes([0.135, y2, 0.78, H])
    for k, lab in zip(['metrics/mAP50(B)', 'metrics/mAP50-95(B)'], ['mAP50', 'mAP50-95']):
        ax2.plot(d['epoch'], d[k], linewidth=1.5, marker='', label=lab)
    ax2.set_title('验证集 mAP 曲线', fontproperties=fp_h2, pad=5)
    ax2.set_xlabel('epoch', fontproperties=fp_note)
    ax2.set_ylabel('mAP', fontproperties=fp_note)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=8, framealpha=0.9, prop=fp_ax)

    doc.y = y2 - 0.044
    doc.caption(cap)


add_curves('wire_v11_v3',
           '图 6　YOLOv11s 基线训练曲线 @ 320×320（数据源：runs/wire_v11_v3）')
add_curves('wire_v11_640',
           '图 7　YOLOv11s 基线训练曲线 @ 640×640（数据源：runs/wire_v11_640）')
doc.para('两条曲线均未见明显过拟合：60 epoch 内验证集 mAP 仍在缓慢上升、'
         '训练损失仍在下降，说明训练轮数尚未饱和，延长训练仍有空间（见第九章）。')
doc.note('数据完整性说明：wire_v11_640_neck 与 wire_v11_640_backbone 各训练了两段，'
         'neck/backbone 第一段因 WinError 1455 页面文件不足中断，取最后一段'
         '（本报告的「60 轮」叙事与 best/last 权重均取自最后一段）。')

# ---------------- 八、问题与原因 ----------------
doc.h1('八、问题与原因')
doc.bullet('问题一：YOLOv5s 的 mAP50 最高（640 下 0.907），YOLOv11s 未反超。'
           '原因：（a）YOLOv5 使用 anchor 机制，对细长线状目标（断股）的先验框匹配更友好，'
           '这正是本任务最难的类别；（b）YOLOv5s 在 60 epoch 内仍在涨（损失曲线未收敛），'
           '而 v11 在高分辨率下收敛更快，即 v5 这一侧尚未跑满，'
           '两者不处在同一训练充分度上。')
doc.bullet('问题二：CBAM 未带来稳定增益。增益方向对优化器与插入位置敏感'
           '（640 下四套设置的差值全部落在 ±0.013 以内且方向不一致），'
           '说明其效果接近噪声量级，而非稳定的结构改进。')
doc.para('对问题二做了专门的机理排查（code/tools/viz_cbam_attention.py 与 '
         'code/tools/analyze_cbam_degradation.py），结论是：'
         'CBAM 在本任务里近似恒等映射，而根因既不是任务难度，也不是注意力模块选型，'
         '而是空间注意力读出头（7×7 卷积 + sigmoid）的结构性缺陷。')
doc.bullet('现象层：6 个 CBAM 模型 × 21 个 CBAM 模块 × 8 张测试图的跨图聚合（口径为'
           '「同一模块在 8 张图上的波动范围」，并非 168 个独立样本）显示 —— 空间注意力标准差'
           '中位数 0.0308（最大 0.0960）、通道注意力标准差中位数 0.0259、'
           '对比度比（框内均值/框外均值）中位数 1.0152。即空间注意力接近常数场、'
           '通道门控接近均匀分布，两者相乘相当于给特征图乘一个常数。')
doc.bullet('排除「梯度传不到」：在一轮真实训练的每个反向传播上挂 backward hook，'
           '实测 CBAM 权重梯度范数为宿主层的 6.23e-02（最小 1.44e-02），'
           '远高于「梯度消失」的 1e-3 判据；且 1 个 epoch 的权重位移速率与宿主层同量级'
           '（比值 0.981）。')
doc.bullet('排除「权重根本没动」：CA 权重每 epoch 相对位移 0.957%，宿主层 0.975%。'
           '需说明的是，静态检查里「权重 std 停在初始化附近」不能推出「没学到」——'
           '各向同性的随机游走本就近似保住权重范数，只有方向一致的漂移才会显著改变它；'
           '该证据只说明位移没有方向。')
doc.bullet('定位到读出头：空间注意力的输入（通道均值图与通道最大值图的拼接）本身'
           '带着缺陷的区分信息 —— 通道最大值图的空间标准差 0.26~0.71（全部 > 0.05，'
           '且高于同尺寸随机噪声图的对照 0.27~0.52），框内/框外比 1.19~1.94；'
           '而经过 7×7 卷积 + sigmoid 之后，输出的框内/框外比只剩 0.98~1.09。'
           '即读出头把 1.4~1.9 倍的区分度压回了 ~1。')
doc.bullet('两种压缩机制（均由 sigmoid 造成）：（a）较浅层（层 2/4/6）卷积输出带有'
           '大的正直流分量（+3.1 ~ +4.9），sigmoid 单边饱和到 ≈1，输出是常数 1；'
           '值得注意的是噪声图的该分量反而更大（8.3 / 5.0 / 6.7），'
           '说明该直流分量与图像内容无关，是「正均值输入 × 无偏置卷积」的结构性偏置；'
           '（b）较深层（层 8/9）卷积输出 ≈ 0（−0.08 / −0.23），输出落在 sigmoid '
           '线性区、近似常数 0.5，且卷积输出的幅度先比输入小 2~5 倍。')
doc.bullet('为什么这解释了「无增益」：一个近似常数的注意力图，'
           '在数学上会被紧随其后的批归一化层严格吸收'
           '（BN(c·x) = γ(cx − cμ)/√(c²σ²) + β = BN(x)，常数缩放的梯度恒为 0），'
           '网络无法从这个自由度学到任何东西。')
doc.figure(0.128,
           png_draw('D:/2/report/_cbam_attention/wire_v11_640_both_channel.png'),
           '图 8　CBAM 通道注意力权重图（wire_v11_640_both；各层权重近似均匀分布，'
           '跨图标准差 0.006~0.033，无区分度）',
           tick_pad=0.006, title_pad=0.006)
doc.bullet('遗留的原因假设已作废：此前怀疑「骨干浅层的空间注意力放大背景导线纹理响应」，'
           '但实测空间注意力的输出对缺陷的框内/框外比是 0.98~1.09（近 1），'
           '它既不放大缺陷也不放大背景，而是被 sigmoid 压成了常数 —— 该假设不成立。')
doc.bullet('问题三：优化器的效果不稳定（320 与 640 下方向相反）。'
           '原因：（a）两种优化器使用各自匹配的学习率（0.01 / 0.001），'
           '在不同分辨率下的有效步长与梯度噪声水平不同，'
           'AdamW 的自适应步长在小尺度、高噪声梯度下优势更明显，'
           '而尺度变大后梯度更稳定，SGD 的隐式正则反而更有利；'
           '（b）差异量级（<0.02）本就在测试集波动范围内（test 仅 179 张 / 283 框），'
           '因此不构成普适规律。')
doc.bullet('旁证：训练中途日志里的 Speed: 行不可用于比较推理速度'
           '（曾出现 640 组 3.6 ms < 320 组 6.9 ms 的物理上不可能的结果），'
           '本地报告的推理数据一律取自专门的端到端基准（logs/fps_benchmark.txt）。')

# ---------------- 九、解决方案 ----------------
doc.h1('九、解决方案')
doc.bullet('提高输入分辨率到 768：本任务原图为 1294×654，640 下仍有下采样空间；'
           '实测 320→640 提升 +0.14，768 有望继续受益（需评估显存与实时性）。')
doc.bullet('增加训练轮数到 150：v5 与 v11 的曲线在 60 epoch 时均未收敛，'
           '延长训练有望同时提升两者，也可能改变 v5 与 v11 的相对次序。')
doc.bullet('离线数据增强：断股实例仅 1295 个且形态单一，'
           '用离线增强（旋转/拼接/局部裁剪+Copy-Paste）扩充细长目标样本，'
           '针对性强化最弱类别。')
doc.bullet('两阶段检测：先检出导线区域，再在局部高分辨率 ROI 上做缺陷分类/检测，'
           '用等效放大换取断口的有效像素，同时控制整体推理成本。')
doc.para('以下为待验证方向，尚未做实验，不作为本报告的解决方案：')
doc.bullet('待验证方向一：修改空间注意力的读出头，而非更换注意力模块。'
           '第八章的实测表明瓶颈是「7×7 卷积 + sigmoid」这一读出头把输入的区分度压成了常数，'
           '因此可试的方向是中心化空间注意力的输入（先减去空间均值再卷积，'
           '以消除造成饱和的直流分量）、或换用非饱和激活。'
           '该方向需实验验证后才可下结论。')
doc.bullet('待验证方向二（原「换 SE/ECA」）：SE / ECA 只做通道注意力、参数更少，'
           '在小目标少样本任务上通常更稳 —— 但本任务不适用这一通常经验：'
           'SE/ECA 同样依赖全局池化 + MLP，若本任务的根本原因是梯度无法驱动注意力分支，'
           '换模块未必能解决。需先验证梯度问题，再决定是否换模块。'
           '（实测已排除「梯度传不到」，瓶颈在空间注意力的读出头；'
           'SE/ECA 是纯通道模块，不走这条路，也等于没有回答第八章暴露的问题。）')
doc.bullet('TensorRT INT8 量化：面向 Jetson Orin Nano 部署，'
           '在 640 输入（6.48 ms/张 @RTX 4060）基础上做 FP16/INT8 量化与层融合，'
           '换取边缘端实时性。')

# ---------------- 十、结论与局限 ----------------
doc.h1('十、结论与局限')
doc.h2('10.1　推荐模型综合对比')
doc.para('下表在统一口径下对比两个候选模型。指标（mAP50 / mAP50-95 / P / R）取自 '
         '896×896——v11 与 v5 唯一共同跑过的高分辨率尺寸；FPS 取自 640×640 端到端'
         '基准（896 未做推理基准）。')
t14 = [['维度', 'v11', 'v5', '说明'],
       ['mAP50', '0.9148', '0.9310', 'v5 高 1.6pt（方向性，非量化）'],
       ['mAP50-95', '0.5362', '0.5290', 'v11 高 0.7pt'],
       ['FPS', '154', '98', 'v11 快 57%'],
       ['架构', '2024', '2020', ''],
       ['P', '0.940', '0.902', ''],
       ['R', '0.821', '0.877', '']]
doc.table(t14[0], t14[1:], col_widths=[0.16, 0.145, 0.145, 0.45],
          row_h=0.026,
          note='表 12　推荐模型综合对比（v11 = YOLOv11s 基线，v5 = YOLOv5s；'
               'mAP50/mAP50-95/P/R 取自 896×896，FPS 取自 640×640 端到端基准）')
doc.para('综合 mAP50-95、推理速度、精确率与架构代际，v11 为本数据集推荐模型；'
         'v5 在 mAP50 与召回率上略优，两者差异只作方向性表述。')
doc.para('结论：')
doc.bullet('分辨率是压倒性主变量：320 → 640 使 test mAP50 均值提升约 +0.14'
           '（0.7557/0.7719 → 0.8951/0.8912），mAP50-95 提升约 +0.13，'
           '且几乎覆盖全部模型配置。断股类从 0.48~0.60 提升到 0.75~0.85，'
           '此前的类别级瓶颈被解除。')
doc.bullet('优化器影响次之且方向不稳定：320 下 AdamW 略高（7 组均值 +0.0162），'
           '640 下 SGD 略高（同构子集 −0.0095 ~ −0.0108），方向反转。'
           '真正决定性的是学习率与优化器是否匹配（错配 lr0=0.01 时 mAP50 仅 0.635，'
           '纠正后 0.779）。')
doc.bullet('CBAM 无稳定增益：四种设置（320/640 × SGD/AdamW）下差值方向不一致，'
           '640 下全部落在 ±0.013 以内，不能作为通用涨点手段。')
doc.bullet('推荐模型：YOLOv11s 基线。按 mAP50，YOLOv5s（0.907）略高于 v11（0.8992）'
           '（差 0.8pt，在组内极差 1.01pt 内）；'
           '但按 mAP50-95（定位精度）v11（0.5286）高于 v5（0.500），推理速度快约 57%'
           '（154 vs 98 FPS），且架构代际更新。综合这三项，本数据集推荐 YOLOv11s，'
           '依据见上方表 12。')
doc.para('局限：')
doc.bullet('验证集（178 张 / 275 实例）与测试集（179 张 / 283 实例）规模都偏小，'
           '单次评估置信区间较宽，组间小于 0.01 的差异（如 CBAM 在 640 下的增益、'
           '优化器在同构子集下的差值）不宜作强结论。')
doc.bullet('640 + AdamW 一套只有 5 组（缺 v12 两组），四套并非完全对称，'
           '凡是跨 640 两套的均值比较都必须按同构子集做，本报告已如此处理。')
doc.bullet('未做分辨率的连续扫描（如 480、512、768）与训练时长扫描，'
           '640 是否接近任务上限、60 epoch 是否训练充分均未确定。')
doc.bullet('推理基准只测了 8 组（架构 × 尺寸），两个 CBAM 组与 v12+CBAM 未单测，'
           '表中以「—」标注；且本机为笔记本 GPU，SM 频率存在热漂移，'
           '锁频需管理员权限，故采用顶频 + 轮转 + 丢首轮的方式控制，'
           '组间小于 0.2 ms 的差异不应解读。')
doc.bullet('各框架仍保留各自推荐的其余超参（数据增广强度、损失权重、warmup 策略等），'
           '仅对齐了优化器、初始学习率、输入尺寸与随机种子；'
           '论文中应注明「各框架采用各自推荐配置，通用超参保持一致」。')

# ---------------- 十一、896 对照（独立口径） ----------------
doc.h1('十一、896×896 对照（独立口径，不与 640 混算）')
doc.para('本条与前面的四套分开陈述，原因有二：（1）896 只跑了 YOLOv11s 基线与 YOLOv5s '
         '两个配置（无 CBAM 组、无 v12），样本不足以构成完整的一套；'
         '（2）896 与 640 输入尺寸不同，若并入 640 的均值会把「尺寸差」混进「模型差」。'
         '故 896 的读数只作同尺寸内的两两对照，不与 640 的数值相加或求均值。')
doc.para('两组的训练超参已对齐：同为 SGD + lr0=0.01、batch=8、imgsz=896、60 epoch、seed=42；'
         '唯一未对齐的是末端学习率比例（v5 一侧 lrf=0.01，v11 一侧 lrf=0.005）。')
doc.para('注（v5-vs-v11 的口径，同第二章 lrf 注）：' + LRF_NOTE + LRF_TAIL)
t12 = [['模型配置', 'P', 'R', 'mAP50', 'mAP50-95', 'break 断股', 'thunderbolt 雷击',
        'wear 磨损', '参数量']]
for key, p, r, m50, m95, b, tb, w in S896:
    t12.append([NAME[key].replace('†', ''), p, r, m50, m95, b, tb, w, S896_PARAMS[key]])
doc.table(t12[0], t12[1:],
          col_widths=[0.19, 0.075, 0.075, 0.085, 0.10, 0.10, 0.145, 0.10, 0.13],
          small=True, row_h=0.026,
          note='表 13　896×896 下的 test 结果（best.pt；数据源 logs/eval_896.log；'
               'v5 按源精度 3 位；896 未做推理基准，表中不含 ms/FPS）。'
               '参数量口径：v11 训练参数量 9,428,953（logs/train_896.log），'
               'fused 后 9,413,961（BN 折叠，logs/eval_896.log），全表统一用训练参数量。')
doc.para('896 下的模型间方向与 640 同向：YOLOv5s 的 mAP50（0.931）高于 YOLOv11s（0.9148），'
         '与 640 下 v5 在 mAP50 上领先一致，说明「v5 的 mAP50 更高」不是 640 单点的偶然。'
         '但从 mAP50-95 看，v11（0.5362）略高于 v5（0.529），也与 640 下 v11 的定位优势一致。')
doc.para('断股（break）在 896 下仍是三类中最弱（v11 0.870 / v5 0.862），'
         '且其 mAP50-95 只有 0.449 / 0.426，远低于同尺寸雷击（0.620 / 0.603）：'
         '即 896 已能修正细长目标的「大致定位」（mAP50），却仍未改善其「精细定位」。')
doc.para('公平性边界：两组同为 SGD / lr0=0.01 / batch=8 / 60 epoch / seed=42，'
         '差异集中在框架自身的实现（anchor 机制、损失与增广默认值）与 lrf（0.01 vs 0.005）。'
         '故本对照足以支撑「同尺寸下 v5 的 mAP50 更高」这一方向性结论。'
         '量化边界：896 只有一个尺寸、每组仅一个模型，本身没有组内重复，'
         '无法直接估计噪声尺度；故借用 640 同架构四组的组内极差 1.01pt 作代理，'
         '本组 1.6pt 略超该代理 → 其差值不作量化比较，只作方向性结论。')

# ---------------- 十二、数据泄漏与 clean 子集核查 ----------------
doc.h1('十二、数据泄漏与 clean 子集核查（报告口径）')
doc.para('方法：dHash 汉明距离 ≤2 划 leak、其余 clean；5 个随机划分种子做置换检验。')
doc.para('本数据集由视频分帧得到，训练/测试之间可能存在同场景近复制品。'
         '若存在，报告中的 test 指标会被「记忆」抬高。为此做了一次专门的泄漏核查：'
         '用 dHash 汉明距离把 test（与 val）划成 leak（最近邻距离 ≤2，近复制品）'
         '与 clean（无近复制品）两个子集，分别评估，并以 5 个随机划分种子做置换检验。')
doc.para('核查结论（报告口径原文，不得改写）：')
doc.para('test 侧泄漏效应不可分辨（两格 p=0.60）；机理是样本量最小类的 AP '
         '在小子集上不可稳定估计——四格无一格三类同号，每格由单一类主导'
         '（56%~171%），其中 wear 跨格符号一致。跨格符号一致只说明「每次都是'
         '同一个薄类被小样本反复主导」，不含可识别机制。'
         'val 侧两格超出零分布，但同样由 wear 主导（896 独扛 110%、'
         '640 占 56.3%），且 val_leak 的 wear 仅 3 图/16 实例 → 该超出不可解读为泄漏效应。'
         '结论：报告口径不修正 headline；clean 子集读数备查；clean ≠ 未见过的场景。')
doc.note('字形已适配：上段引录自 WORKLOG.md 原文，仅将原文的重推号（双线右箭头）'
         '写作「→」，语义与数字一字未改（微软雅黑无该字形）。')
t13 = [['模型 / split', '总分差(点)', 'break', 'thunderbolt', 'wear', '主导类（占比）']]
for row in [
    ['896 test', '+0.91', '+1.15', '+2.89', '−1.30', 'tb（+106%）'],
    ['640 test', '+1.81', '+9.30', '−2.50', '−1.30', 'break（+171%）'],
    ['896 val', '+4.49', '−0.38', '−0.94', '+14.83', 'wear（+110%）'],
    ['640 val', '+9.29', '+9.57', '+2.62', '+15.70', 'wear（+56%）'],
]:
    t13.append(row)
doc.table(t13[0], t13[1:], col_widths=[0.17, 0.15, 0.115, 0.165, 0.11, 0.22],
          small=True, row_h=0.0245,
          note='表 14　leak − clean 的逐类拆解（单位：点；每类对 mAP50(all) 的贡献 = '
               '该类 AP 差 ÷ 3）。四个格子里没有一格三类同号，且每格都由单一类主导（|占比| 56%~171%）')
doc.para('由表 14 可见：leak−clean 的总分差由单一类别主导；wear 一类的符号在 val 两格'
         '一致为正（+15.70/+14.83）、在 test 两格一致为负（−1.30/−1.30）。'
         '这不是随机的正负抵消，而是「样本量最小的那一类（val_leak 中 wear 仅 3 图/16 实例）'
         '在小子集上的 AP 波动」主导了差值——即这些差值不可读作泄漏的度量。')
doc.para('部署与写作含义：报告中的 test 指标不因泄漏而需要修正；但「clean 子集」'
         '只表示「无近复制品」，不等于「未见过的场景」（clean 中仍有大量疑似同段视频的别帧），'
         '因此不得据此宣称模型可泛化到新场景。'
         '该限制不改变第十章的数据集内模型选择结论——推荐基于同一数据集内的'
         '相对性能，不含跨场景泛化背书。'
         '另需注意：val 的 best.pt 由完整 val 选出，val 侧还混入选择效应，'
         '故 val 侧的 leak−clean 双重不干净，只作备查。')

# ---------------- 页码 ----------------
n = len(doc.pages)
for i, fig in enumerate(doc.pages):
    fig.text(0.5, 0.035, f'第 {i + 1} 页 / 共 {n} 页', ha='center', va='bottom',
             fontproperties=fp_note, color='#666666')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
PNG_DIR = os.environ.get('REPORT_PNG')
if PNG_DIR:
    os.makedirs(PNG_DIR, exist_ok=True)
    for i, fig in enumerate(doc.pages):
        fig.savefig(os.path.join(PNG_DIR, f'v_final_p{i + 1}.png'), dpi=110)
with PdfPages(OUT) as pdf:
    for fig in doc.pages:
        pdf.savefig(fig)
        plt.close(fig)

size = os.path.getsize(OUT)
print(f'OK  {OUT}')
print(f'页数 = {n}  大小 = {size} bytes ({size / 1024:.1f} KB)')
