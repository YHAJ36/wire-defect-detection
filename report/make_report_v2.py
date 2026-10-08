# -*- coding: utf-8 -*-
"""生成《基于 YOLOv11+CBAM 的输电线路缺陷检测算法》实验报告 v2（SGD 版 7 组数据）。

数据来源：
  - test 集指标     : D:/2/eval_test_sgd.log（7 组 best.pt 在 test 集评估）
  - val 指标 / 曲线 : D:/2/runs/<name>/results.csv
输出：D:/2/report/基于YOLOv11+CBAM的缺陷检测算法_v2.pdf
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

OUT = 'D:/2/report/基于YOLOv11+CBAM的缺陷检测算法_v2.pdf'
RUNS = 'D:/2/runs'

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
fp_td = FontProperties(fname=F_REG, size=9)

PAGE_W, PAGE_H = 8.27, 11.69          # A4
TOP, BOTTOM = 0.945, 0.065
X0, WIDTH = 0.10, 0.80                # 正文/表格左右边界
LINE = 0.0185                         # 9.5pt 行高（axes 分数）
ROW = 0.028                           # 表格单行高
DEBUG = bool(os.environ.get('REPORT_DEBUG'))


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
        self.ax.text(0.5, self.y, text, ha='center', va='top',
                     fontproperties=fp_sub, color='#555555',
                     transform=self.ax.transAxes)
        self.y -= 0.036

    def h1(self, text):
        self._ensure(0.065)
        # 与上文留白
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
        lines = wrap_cjk(text, 88 if indent else 88)
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

    def table(self, col_labels, cell_text, col_widths=None, height=None, note=None):
        h = height or (len(cell_text) + 1) * ROW
        # 余量 = 表后间距 + 表注（最多 2 行）
        pad = 0.06
        if DEBUG:
            print(f'  [table {len(cell_text)}r] y={self.y:.3f} need={h + pad:.3f} '
                  f'{"→ new page" if self.y - (h + pad) < BOTTOM else "ok"}')
        self._ensure(h + pad)
        y0 = self.y - h
        tbl = self.ax.table(
            cellText=cell_text, colLabels=col_labels,
            bbox=(X0, y0, WIDTH, h), cellLoc='center', colWidths=col_widths,
        )
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

    def caption(self, text):
        self.ax.text(0.5, self.y, text, ha='center', va='top',
                     fontproperties=fp_note, color='#333333',
                     transform=self.ax.transAxes)
        self.y -= 0.024


# 行首禁则（不能出现在行首的收尾标点）与行尾禁则（不能出现在行尾的起始标点）
PUNCT_NO_START = '，。；：、）》」”％…！？,.;:)]}%'
PUNCT_NO_END = '（《「“‘([{'


def _tokenize(text):
    """切成 (片段, 显示宽度) ：CJK/全角单字算 2 单位，连续 ASCII 词算一个整体。"""
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
    """按显示宽度折行，ASCII 词/数字不内部断开，并做行首/行尾禁则处理。"""
    lines, cur, u = [], '', 0
    for tok, w in _tokenize(text):
        # 行首禁则：换行后若第一个字符是收尾标点，回贴到上一行末
        # （典型情况是「…（−0.0427 与 −0.0228）。」中 ）与 。 连用时）
        if not cur and tok and tok[0] in PUNCT_NO_START and lines:
            lines[-1] += tok
            continue
        if cur and u + w > max_units:
            # 行首禁则：收尾标点不另起一行，回贴到上一行末尾
            if tok and tok[0] in PUNCT_NO_START:
                lines.append(cur + tok)
                cur, u = '', 0
                continue
            # 行尾禁则：起始标点不留行尾，一并带到下一行
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
    """读 results.csv，返回 {列名: [值...]}（列名已 strip）。

    epoch 统一为 1-based：ultralytics 的 results.csv 本身就是 1-based，
    YOLOv5 的从 0 开始，故按最小值判断是否需要 +1，避免坐标轴整体偏移。
    """
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
#                              数据
# ================================================================
# 表1 数据集分布
T1_COLS = ['数据划分', '图片数', '断股 break', '雷击 thunderbolt', '磨损 wear', '实例总数']
T1_ROWS = [
    ['训练集 train', '1428', '1014', '650', '788', '2452'],
    ['验证集 val', '178', '131', '80', '64', '275'],
    ['测试集 test', '179', '150', '69', '64', '283'],
    ['合计', '1785', '1295', '799', '916', '3010'],
]

# 表2 模型对比（test 集，按 mAP50 降序）
T2_COLS = ['模型', 'P', 'R', 'mAP50', 'mAP50-95']
T2_ROWS = [
    ['YOLOv11s + CBAM (neck_sppf)', '0.8814', '0.7157', '0.7939', '0.3970'],
    ['YOLOv11s (baseline)', '0.7931', '0.7056', '0.7787', '0.4035'],
    ['YOLOv12s (baseline)', '0.8768', '0.6864', '0.7611', '0.3798'],
    ['YOLOv11s + CBAM (both)', '0.7932', '0.6739', '0.7559', '0.3871'],
    ['YOLOv12s + CBAM', '0.8384', '0.6663', '0.7435', '0.3627'],
    ['YOLOv11s + CBAM (backbone)', '0.8561', '0.6140', '0.7360', '0.3548'],
    ['YOLOv5s (baseline)', '0.7990', '0.6690', '0.7210', '0.3450'],
]

# 表4 CBAM 消融（4 种插入位置）
T4_COLS = ['配置', 'test mAP50', 'test mAP50-95', 'val mAP50', 'Δ test mAP50']
T4_ROWS = [
    ['YOLOv11s baseline', '0.7787', '0.4035', '0.7470', '—'],
    ['+ CBAM (backbone)', '0.7360', '0.3548', '0.6767', '−0.0427'],
    ['+ CBAM (neck_sppf)', '0.7939', '0.3970', '0.7223', '+0.0152'],
    ['+ CBAM (both)', '0.7559', '0.3871', '0.7276', '−0.0228'],
]

# 表3 各类别（YOLOv11s + CBAM neck_sppf / YOLOv11s baseline）
T3_COLS = ['模型', '类别', 'P', 'R', 'mAP50', 'mAP50-95']
T3_ROWS = [
    ['YOLOv11s + CBAM (neck)', 'all', '0.881', '0.716', '0.794', '0.397'],
    ['', 'break 断股', '0.754', '0.500', '0.558', '0.230'],
    ['', 'thunderbolt 雷击', '0.955', '0.913', '0.964', '0.528'],
    ['', 'wear 磨损', '0.936', '0.734', '0.860', '0.432'],
    ['YOLOv11s (baseline)', 'all', '0.793', '0.706', '0.779', '0.404'],
    ['', 'break 断股', '0.646', '0.520', '0.560', '0.245'],
    ['', 'thunderbolt 雷击', '0.956', '0.884', '0.954', '0.571'],
    ['', 'wear 磨损', '0.778', '0.713', '0.822', '0.395'],
]

# 表5 优化器混淆变量
T5_COLS = ['优化器配置', 'YOLOv5s mAP50', 'YOLOv11s mAP50', '差值 (v11 − v5)', '判定']
T5_ROWS = [
    ['AdamW + lr0=0.01（超参错配）', '0.734', '0.635', '−0.099', 'v5 领先'],
    ['SGD + lr0=0.01（正确匹配）', '0.721', '0.779', '+0.058', 'v11 领先'],
]

# ================================================================
#                              成稿
# ================================================================
doc = Doc()

doc.title('基于 YOLOv11+CBAM 的输电线路缺陷检测算法实验报告')
doc.subtitle('数据集 dataset_final_bbox（断股 / 雷击 / 磨损 3 类）　·　统一 SGD + seed 42 配置　·　2026-10-07')

# ---------------- 一、数据集说明 ----------------
doc.h1('一、数据集说明')
doc.para('本实验面向架空输电线路缺陷检测任务，共检测 3 类缺陷：break（断股）、'
         'thunderbolt（雷击）、wear（磨损）。数据标注采用 SAM（Segment Anything Model）'
         '自动生成分割掩膜，再将 polygon 标注转换为 bbox，形成检测任务数据集 '
         'dataset_final_bbox（task: detect），最后按 8:1:1 的比例划分为训练集、'
         '验证集与测试集。各划分的图片数与类别实例数见表 1。')
doc.table(T1_COLS, T1_ROWS, col_widths=[0.20, 0.14, 0.16, 0.19, 0.15, 0.16],
          note='表 1　数据集分布（图片数 / 各缺陷类别实例数 / 实例总数）')
doc.para('三类缺陷存在明显的样本不均衡与尺度差异：断股（break）为细长线状目标、'
         '背景纹理相近，标注实例最多（1295 个）；雷击（thunderbolt）为局部熔痕、'
         '特征对比度高；磨损（wear）面积较大但边界模糊。这一差异在后续各类别'
         '指标中有直接反映。')

# ---------------- 二、模型对比 ----------------
doc.h1('二、模型对比（测试集）')
doc.para('为保证横向可比性，7 组模型统一采用 SGD 优化器（lr0=0.01、momentum=0.937、'
         'weight_decay=0.0005）与相同随机种子（seed=42），输入尺寸 320×320，'
         '训练 60 个 epoch，batch size 为 8，DataLoader 的 workers 设为 0'
         '（Windows spawn 模式下多进程易触发页面文件耗尽）。各组均取训练过程中'
         'fitness 最优的权重 best.pt，在测试集（179 张图片、283 个标注框）上评估，'
         '结果按 mAP50 降序排列于表 2。')
doc.table(T2_COLS, T2_ROWS, col_widths=[0.40, 0.15, 0.15, 0.15, 0.15],
          note='表 2　7 组模型在测试集上的检测性能对比（best.pt，fitness = 0.1×mAP50 + 0.9×mAP50-95）')
doc.para('在统一下来的 SGD 配置下，YOLOv11s+CBAM(neck_sppf) 取得最高 mAP50（0.7939），'
         'YOLOv11s 基线次之（0.7787）；YOLOv12s 基线为 0.7611；YOLOv5s 基线最低（0.7210）。'
         '若以 mAP50-95 衡量定位精度，则是 YOLOv11s 基线最高（0.4035），'
         'YOLOv11s+CBAM(neck_sppf) 为 0.3970，二者接近。')

# ---------------- 三、CBAM 消融 ----------------
doc.h1('三、CBAM 消融实验')
doc.para('以 YOLOv11s 为基线，分别在骨干网络（backbone 第 2/4/6/8 层）、'
         '颈部与 SPPF 模块（neck_sppf 第 9 层）以及两者同时插入 CBAM 注意力模块，'
         '考察通道注意力与空间注意力的增益。CBAM 的注入采用子类化 '
         'DetectionTrainer 并重写 get_model() 的方式，在模型按 yaml 重建完成之后再注入——'
         '此前的做法是在 model.train() 之前对模型对象注入，会被训练器重建过程丢弃，'
         '属于无效注入。四种配置的对比见表 4。')
doc.table(T4_COLS, T4_ROWS, col_widths=[0.30, 0.18, 0.20, 0.16, 0.16],
          note='表 4　CBAM 插入位置消融（Δ 为相对 baseline 的 test mAP50 变化）')
doc.para('从结果看，CBAM 并未带来稳定增益：仅在 neck_sppf 位置插入时测试集 mAP50 '
         '提升 0.0152（0.7787 → 0.7939），而插入 backbone 或同时插入两处时反而下降'
         '（分别为 −0.0427 与 −0.0228）。原因推测为：本任务目标尺度偏小、样本量有限，'
         '骨干网络浅层的空间注意力容易放大背景纹理响应，干扰细长线状目标的特征表达；'
         '而在颈部（多尺度融合之后）插入、且只在 SPPF 一处引入，则能抑制背景干扰。'
         '该结论在 YOLOv12s 上得到反向印证——在 YOLOv12s 的 A2C2f 模块全局插入 CBAM 后，'
         '测试集 mAP50 由 0.7611 降至 0.7435（−0.0176）。')

# ---------------- 四、各类别检测结果 ----------------
doc.h1('四、各类别检测结果')
doc.para('表 3 给出性能最优的 YOLOv11s+CBAM(neck_sppf) 与 YOLOv11s 基线在测试集上'
         '按类别拆分的指标。')
doc.table(T3_COLS, T3_ROWS, col_widths=[0.30, 0.22, 0.12, 0.12, 0.12, 0.12],
          note='表 3　YOLOv11s 两种配置在测试集上的分类别指标（按 best.pt 评估）')
doc.para('三类缺陷中，雷击（thunderbolt）检测效果最好，两组模型的 mAP50 均达 0.95 以上，'
         '得益于其局部熔痕特征对比度高、类间区分度大。磨损（wear）次之，'
         '在 CBAM(neck_sppf) 配置下 mAP50 达到 0.860（基线为 0.822），说明'
         '注意力机制对该类的边界判别有实际帮助。')
doc.para('断股（break）是当前唯一的性能瓶颈：两组模型的 break mAP50 仅为 0.558 与 0.560，'
         '与雷击相差约 0.40。结合全部 7 组模型来看，break 类别的 mAP50 普遍落在 '
         '0.482~0.560 区间，是拉低整体指标的主要因素。其困难在于：断股为细长线状目标、'
         '像素占比低，且断口处的单丝结构与正常导线纹理高度相似，在 320×320 输入下'
         '有效像素进一步被压缩。后续工作应针对该类做专项优化（如提高输入分辨率、'
         '引入针对细长目标的损失或数据增广）。')

# ---------------- 五、训练曲线 ----------------
doc.h1('五、训练曲线')
doc.para('图 1 为 YOLOv11s 基线（wire_v11_v3）的 60 epoch 训练曲线，'
         '包含训练损失（box/cls/dfl）与验证集 mAP 的变化；相应地，'
         '图 2 给出 YOLOv5s 基线（wire_v5_v2）的训练损失（box/obj/cls）与 mAP 曲线。'
         '两者在训练后期均已收敛，验证集 mAP 曲线在 50 个 epoch 之后趋于平缓。')


def add_curves(doc, name, cap, loss_keys, loss_labels, map_keys, map_labels):
    if DEBUG:
        print(f'  [figure {name}] y={doc.y:.3f} need=0.55 '
              f'{"→ 换页" if doc.y - 0.55 < BOTTOM else ""}')
    doc._ensure(0.55)
    y1 = doc.y - 0.235
    ax1 = doc.fig.add_axes([0.135, y1, 0.78, 0.205])
    d = read_results(name)
    for k, lab in zip(loss_keys, loss_labels):
        ax1.plot(d['epoch'], d[k], linewidth=1.3, label=lab)
    ax1.set_title('训练损失曲线', fontproperties=fp_h2, pad=5)
    ax1.set_xlabel('epoch', fontproperties=fp_note)
    ax1.set_ylabel('loss', fontproperties=fp_note)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=8)
    ax1.legend(fontsize=8, framealpha=0.9)

    y2 = y1 - 0.275
    ax2 = doc.fig.add_axes([0.135, y2, 0.78, 0.205])
    for k, lab in zip(map_keys, map_labels):
        ax2.plot(d['epoch'], d[k], linewidth=1.5, marker='', label=lab)
    ax2.set_title('验证集 mAP 曲线', fontproperties=fp_h2, pad=5)
    ax2.set_xlabel('epoch', fontproperties=fp_note)
    ax2.set_ylabel('mAP', fontproperties=fp_note)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=8, framealpha=0.9)

    doc.y = y2 - 0.035
    doc.caption(cap)


add_curves(
    doc, 'wire_v11_v3', '图 1　YOLOv11s 基线训练曲线（数据源：runs/wire_v11_v3）',
    ['train/box_loss', 'train/cls_loss', 'train/dfl_loss'],
    ['box loss', 'cls loss', 'dfl loss'],
    ['metrics/mAP50(B)', 'metrics/mAP50-95(B)'],
    ['mAP50', 'mAP50-95'],
)
add_curves(
    doc, 'wire_v5_v2', '图 2　YOLOv5s 基线训练曲线（数据源：runs/wire_v5_v2）',
    ['train/box_loss', 'train/obj_loss', 'train/cls_loss'],
    ['box loss', 'obj loss', 'cls loss'],
    ['metrics/mAP_0.5', 'metrics/mAP_0.5:0.95'],
    ['mAP50', 'mAP50-95'],
)

# ---------------- 六、原因分析 ----------------
doc.h1('六、原因分析：优化器混淆变量')
doc.para('本实验早期曾得到“YOLOv5 全面领先 YOLOv11”的结论，后续排查发现该结论由'
         '优化器配置不匹配造成的混淆变量导致，是本轮实验最重要的一个方法学发现。')
doc.para('问题在于：三套框架的历史训练配置并不一致——YOLOv5 使用 SGD（lr0=0.01），'
         '而 YOLOv11 / YOLOv12 使用 AdamW，且沿用了 lr0=0.01。AdamW 的适用学习率'
         '通常在 1e-3 量级，0.01 属于偏大，会导致训练震荡、难以收敛到较优解；'
         '而 SGD 恰恰需要 0.01 这一量级。也就是说，先前的对比是在“一套框架用了'
         '合适学习率、另一套框架用了错配学习率”的前提下进行的，结果不具可比性。')
doc.table(T5_COLS, T5_ROWS, col_widths=[0.34, 0.17, 0.18, 0.18, 0.13],
          note='表 5　优化器配置对结论的影响（同一测试集、同一随机种子）')
doc.para('统一到 SGD（lr0=0.01，与 YOLOv5 一致）后，结论发生反转：YOLOv5s 由 0.734 '
         '降至 0.721，而 YOLOv11s 由 0.635 升至 0.779——同一个 YOLOv11s 模型，'
         '仅更换优化器配置，测试集 mAP50 提升了 0.144（0.635 → 0.779）；'
         '差距方向也从“v5 领先 v11 约 0.099”翻转为“v11 领先 v5 约 0.058”。')
doc.para('因此，先前“YOLOv5 全面领先”的判断是优化器不匹配造成的假象，'
         '并非模型架构本身的差异。这一发现也提示：在多框架横向对比时，'
         '优化器与学习率必须作为受控变量显式对齐，否则实验结论可能完全反向。'
         '作为进一步验证，本报告成稿时已另行启动一组 AdamW（lr0=0.001，'
         '即 AdamW 的合适学习率）对照训练，用于确认“学习率错配”而非“AdamW 本身”'
         '才是先前 v11 表现偏低的真正原因。')

# ---------------- 七、结论 ----------------
doc.h1('七、结论')
doc.bullet('在统一 SGD + seed 42 的公平配置下，YOLOv11s+CBAM(neck_sppf) 取得最优测试集 '
           'mAP50 = 0.7939，YOLOv11s 基线为 0.7787，两者均优于 YOLOv12s（0.7611）'
           '与 YOLOv5s（0.7210）。若以 mAP50-95 衡量，YOLOv11s 基线最高（0.4035）。')
doc.bullet('CBAM 未表现出稳定增益：仅 neck_sppf 位置插入时测试集 mAP50 提升 0.0152，'
           '在 backbone 或 both 位置插入则分别下降 0.0427 与 0.0228；'
           '在 YOLOv12s 上全局插入亦下降 0.0176。注意力机制的有效性高度依赖插入位置'
           '与任务特性，不宜作为通用涨点手段。')
doc.bullet('断股（break）是唯一的性能瓶颈，7 组模型的 break mAP50 仅 0.482~0.560，'
           '而雷击已达 0.93 以上、磨损最高 0.860。后续优化的重点应放在断股类。')
doc.bullet('方法学结论：多框架横向对比必须对齐优化器与学习率。优化器配置错配曾使'
           '同一 YOLOv11s 的测试集 mAP50 相差 0.144，并导致完全相反的排名结论。')

# ---------------- 八、局限性 ----------------
doc.h1('八、局限性')
doc.bullet('验证集规模偏小（仅 178 张图片），单轮验证结果波动较大，'
           '且验证集与测试集给出的模型排序并不完全一致——例如验证集上 YOLOv5s 的 '
           'mAP50 最高（0.7577），而测试集上最高的是 YOLOv11s 系列。'
           '本报告统一以测试集结论为准。')
doc.bullet('三套框架的其余超参数仍保留各自的推荐默认值（数据增广强度、损失权重、'
           'warmup 策略等），仅对齐了优化器、初始学习率与随机种子。'
           '论文中应注明“各框架采用各自推荐配置，通用超参保持一致”，'
           '以避免读者误认为所有超参完全对齐。')
doc.bullet('本报告的结论基于 320×320 输入与 60 epoch 的训练预算，'
           '未做分辨率与训练时长的充分扫描；断股类的专项提升空间（如提高输入分辨率）'
           '尚未验证。')

# ---------------- 页码 ----------------
n = len(doc.pages)
for i, fig in enumerate(doc.pages):
    fig.text(0.5, 0.035, f'第 {i + 1} 页 / 共 {n} 页', ha='center', va='bottom',
             fontproperties=fp_note, color='#666666')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with PdfPages(OUT) as pdf:
    for fig in doc.pages:
        pdf.savefig(fig)
        plt.close(fig)

size = os.path.getsize(OUT)
print(f'OK  {OUT}')
print(f'页数 = {n}  大小 = {size} bytes ({size / 1024:.1f} KB)')
