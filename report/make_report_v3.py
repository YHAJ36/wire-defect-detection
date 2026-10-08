# -*- coding: utf-8 -*-
"""生成《基于 YOLOv11+CBAM 的输电线路缺陷检测算法》实验报告 v3（三套对比：320 SGD / 320 AdamW / 640）。

数据来源：
  - 320 SGD  test 指标 : D:/2/eval_test_sgd.log
  - 320 AdamW test 指标 : D:/2/eval_test_adamw.log
  - 640      test 指标 : D:/2/train_640_all.log（5 组）+ D:/2/eval_test_640.log（baseline）
  - 训练曲线            : D:/2/runs/<name>/results.csv
输出：D:/2/report/基于YOLOv11+CBAM的缺陷检测算法_v3.pdf

精度说明：YOLOv5 官方 val.py 输出为 3 位小数，故 v5 各组的 mAP50-95 保留 3 位。
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

OUT = 'D:/2/report/基于YOLOv11+CBAM的缺陷检测算法_v3.pdf'
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
fp_th_s = FontProperties(fname=F_BOLD, size=7.5)
fp_td = FontProperties(fname=F_REG, size=9)
fp_td_s = FontProperties(fname=F_REG, size=7.5)

PAGE_W, PAGE_H = 8.27, 11.69          # A4
TOP, BOTTOM = 0.945, 0.065
X0, WIDTH = 0.10, 0.80
LINE = 0.0185
ROW = 0.028
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
        self.ax.text(0.5, self.y, text, ha='center', va='top',
                     fontproperties=fp_note, color='#333333',
                     transform=self.ax.transAxes)
        self.y -= 0.024


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

# 表2 三套并排（各自按 test mAP50 降序；同序号=同排名，非同模型）
T2_COLS = ['#', '320 SGD 组', 'mAP50', 'mAP50-95',
           '320 AdamW 组', 'mAP50', 'mAP50-95',
           '640 组', 'mAP50', 'mAP50-95']
T2_ROWS = [
    ['1', 'v11+CBAM(neck)', '0.7939', '0.3970', 'v11+CBAM(neck)', '0.7908', '0.3964', 'v5', '0.9070', '0.500'],
    ['2', 'v11 (baseline)', '0.7787', '0.4035', 'v12+CBAM', '0.7824', '0.3920', 'v11+CBAM(backbone)', '0.9061', '0.5207'],
    ['3', 'v12', '0.7611', '0.3798', 'v11 (baseline)', '0.7817', '0.3940', 'v11+CBAM(neck)', '0.9016', '0.5192'],
    ['4', 'v11+CBAM(both)', '0.7559', '0.3871', 'v12', '0.7785', '0.3906', 'v11 (baseline)', '0.8992', '0.5286'],
    ['5', 'v12+CBAM', '0.7435', '0.3627', 'v11+CBAM(both)', '0.7731', '0.3839', 'v11+CBAM(both)', '0.8960', '0.5156'],
    ['6', 'v11+CBAM(backbone)', '0.7360', '0.3548', 'v5', '0.7550', '0.322', 'v12+CBAM', '0.8874', '0.5071'],
    ['7', 'v5', '0.7210', '0.345', 'v11+CBAM(backbone)', '0.7418', '0.3759', 'v12', '0.8682', '0.4909'],
    ['—', '均值', '0.7557', '0.3757', '均值', '0.7719', '0.3793', '均值', '0.8951', '0.5117'],
]

# 表3 同一模型三套对照（行=模型配置）
T3_COLS = ['模型配置', '320 SGD', '320 AdamW', '640', 'Δ (640 − 320 SGD)']
T3_ROWS = [
    ['YOLOv11s (baseline)', '0.7787', '0.7817', '0.8992', '+0.1205'],
    ['YOLOv11s + CBAM (backbone)', '0.7360', '0.7418', '0.9061', '+0.1701'],
    ['YOLOv11s + CBAM (neck_sppf)', '0.7939', '0.7908', '0.9016', '+0.1077'],
    ['YOLOv11s + CBAM (both)', '0.7559', '0.7731', '0.8960', '+0.1401'],
    ['YOLOv12s (baseline)', '0.7611', '0.7785', '0.8682', '+0.1071'],
    ['YOLOv12s + CBAM', '0.7435', '0.7824', '0.8874', '+0.1439'],
    ['YOLOv5s (baseline)', '0.7210', '0.7550', '0.9070', '+0.1860'],
    ['均值', '0.7557', '0.7719', '0.8951', '+0.1394'],
]

# 表4 优化器对比
T4_COLS = ['优化器（320×320）', '均值 mAP50', '均值 mAP50-95', '最优组 mAP50', 'val 均值 mAP50']
T4_ROWS = [
    ['SGD + lr0=0.01', '0.7557', '0.3757', '0.7939（v11+CBAM neck）', '0.7291'],
    ['AdamW + lr0=0.001', '0.7719', '0.3793', '0.7908（v11+CBAM neck）', '0.7270'],
]

# 表5 分辨率对比
T5_COLS = ['输入尺寸', '均值 mAP50', '均值 mAP50-95', '最优组 mAP50', '相对 320 SGD']
T5_ROWS = [
    ['320×320（SGD）', '0.7557', '0.3757', '0.7939', '—'],
    ['320×320（AdamW）', '0.7719', '0.3793', '0.7908', '+0.0162'],
    ['640×640（SGD）', '0.8951', '0.5117', '0.9070', '+0.1394'],
]

# 表6 CBAM 消融（YOLOv11s 四组，320 与 640 各一）
T6_COLS = ['配置', '320 mAP50', '320 mAP50-95', '640 mAP50', '640 mAP50-95']
T6_ROWS = [
    ['YOLOv11s baseline', '0.7787', '0.4035', '0.8992', '0.5286'],
    ['+ CBAM (backbone)', '0.7360', '0.3548', '0.9061', '0.5207'],
    ['+ CBAM (neck_sppf)', '0.7939', '0.3970', '0.9016', '0.5192'],
    ['+ CBAM (both)', '0.7559', '0.3871', '0.8960', '0.5156'],
]

# 表7 640 各类别（test 集，按 mAP50 降序）
T7_COLS = ['模型', 'break 断股', 'thunderbolt 雷击', 'wear 磨损', '整体 mAP50']
T7_ROWS = [
    ['YOLOv5s (v5_640)', '0.831', '0.969', '0.921', '0.9070'],
    ['YOLOv11s+CBAM (backbone)', '0.853', '0.969', '0.896', '0.9061'],
    ['YOLOv11s+CBAM (neck_sppf)', '0.827', '0.973', '0.904', '0.9016'],
    ['YOLOv11s (baseline)', '0.811', '0.969', '0.918', '0.8992'],
    ['YOLOv11s+CBAM (both)', '0.801', '0.977', '0.910', '0.8960'],
    ['YOLOv12s+CBAM', '0.796', '0.970', '0.896', '0.8874'],
    ['YOLOv12s (baseline)', '0.754', '0.961', '0.890', '0.8682'],
]

# ================================================================
#                              成稿
# ================================================================
doc = Doc()

doc.title('基于 YOLOv11+CBAM 的输电线路缺陷检测算法实验报告')
doc.subtitle('数据集 dataset_final_bbox（断股 / 雷击 / 磨损 3 类）　·　'
             '三套对比：320 SGD / 320 AdamW / 640　·　2026-10-08')

# ---------------- 一、数据集 ----------------
doc.h1('一、数据集')
doc.para('本实验面向架空输电线路缺陷检测任务，共检测 3 类缺陷：break（断股）、'
         'thunderbolt（雷击）、wear（磨损）。数据标注由 SAM（Segment Anything Model）'
         '自动生成分割掩膜，再将 polygon 标注转换为 bbox，形成检测数据集 '
         'dataset_final_bbox（task: detect），最后按 8:1:1 划分为训练集 1428 张、'
         '验证集 178 张、测试集 179 张（共 1785 张）。各划分的图片数与类别实例数见表 1。')
doc.table(T1_COLS, T1_ROWS, col_widths=[0.20, 0.14, 0.16, 0.19, 0.15, 0.16],
          note='表 1　数据集分布（图片数 / 各缺陷类别实例数 / 实例总数）')
doc.para('三类缺陷存在明显的尺度与形态差异：断股（break）为细长线状目标、'
         '与背景导线纹理高度相似，标注实例最多（1295 个）；雷击（thunderbolt）'
         '为局部熔痕、特征对比度高；磨损（wear）面积较大但边界模糊。'
         '这一差异在后续各类别指标中有直接反映。')

# ---------------- 二、三套对比总表 ----------------
doc.h1('二、三套对比总表（测试集）')
doc.para('本章给出三套实验的测试集结果并排列出：320×320 + SGD、320×320 + AdamW、'
         '640×640 + SGD。三套均为 7 组模型（YOLOv11s 基线及其 3 种 CBAM 插入位置、'
         'YOLOv12s 基线及 +CBAM、YOLOv5s 基线），统一 seed=42、batch=8、训练 60 epoch，'
         '取 fitness 最优的 best.pt，在测试集（179 张图片、283 个标注框）上评估。'
         '每套各自按 test mAP50 降序排列，表中同一行号仅表示同一排名，'
         '不代表同一模型；同一模型的跨套对照见表 3。')
doc.table(T2_COLS, T2_ROWS,
          col_widths=[0.040, 0.148, 0.070, 0.082, 0.148, 0.070, 0.082, 0.145, 0.070, 0.082],
          left=0.032, width=0.936, small=True, row_h=0.0245,
          note='表 2　三套实验测试集结果并排（best.pt；每套按 mAP50 降序；'
               'YOLOv5 官方 val.py 输出精度为 3 位小数，故 v5 各组的 mAP50-95 保留 3 位）')
doc.para('三套的均值差异显著：320 SGD 为 0.7557，320 AdamW 为 0.7719，'
         '640 为 0.8951。640 一套中全部 7 组均落入 0.868~0.907 区间，'
         '而 320 两套的最好成绩分别只有 0.7939 与 0.7908。')

# ---------------- 三、同一模型三套对照 ----------------
doc.h1('三、同一模型三套对照')
doc.para('为排除排名变化带来的干扰，表 3 按模型配置对齐三套结果，'
         '并给出 640 相对 320 SGD 的绝对提升。')
doc.table(T3_COLS, T3_ROWS, col_widths=[0.34, 0.15, 0.16, 0.14, 0.21],
          note='表 3　同一模型配置在三种训练设置下的 test mAP50 对照'
               '（Δ 为 640 相对 320 SGD 的差值）')
doc.para('全部 7 个模型配置在 640 下均取得提升，提升幅度为 +0.1071 ~ +0.1860，'
         '均值 +0.1394。其中 YOLOv5s 提升最大（+0.1860），'
         'YOLOv12s 基线提升最小（+0.1071）。提升幅度与模型架构无单调关系，'
         '说明该收益主要来自输入分辨率对细长小目标的还原，而非某一架构的专属优势。')

# ---------------- 四、优化器对比 ----------------
doc.h1('四、优化器对比（320 SGD vs 320 AdamW）')
doc.para('在相同输入尺寸（320×320）下比较两种优化器。两者均使用各自匹配的学习率：'
         'SGD 用 lr0=0.01（与 YOLOv5 官方配置一致），AdamW 用 lr0=0.001'
         '（AdamW 的常用量级）——这一点是关键前提，早期实验曾让 AdamW 沿用 lr0=0.01，'
         '属于学习率与优化器错配，会得出完全相反的结论。')
doc.table(T4_COLS, T4_ROWS, col_widths=[0.24, 0.16, 0.18, 0.26, 0.16],
          note='表 4　两种优化器在 320×320 下的测试集表现（各 7 组）')
doc.para('两者差距很小且方向并不一致：以测试集 7 组均值衡量，AdamW 略高'
         '（0.7719 vs 0.7557，差 +0.0162），7 组配对中有 6 组为正；'
         '但以各套的最优组衡量，SGD 略高（0.7939 vs 0.7908，差 +0.0031）；'
         '在验证集均值上同样是 SGD 略高（0.7291 vs 0.7270）。'
         '综合来看，优化器（在学习率正确匹配的前提下）带来的差异不足 0.02，'
         '远小于分辨率带来的 +0.1394，属于非决定性因素；'
         '而学习率与优化器是否匹配则是决定性的——同一 YOLOv11s 在 AdamW+lr0=0.01 下'
         '测试集 mAP50 仅 0.635，换用 SGD+lr0=0.01 后升至 0.779（+0.144）。')

# ---------------- 五、分辨率对比 ----------------
doc.h1('五、分辨率对比（320 vs 640）')
doc.para('这是本组实验最显著的结论。表 5 对比三种训练设置的整体表现。')
doc.table(T5_COLS, T5_ROWS, col_widths=[0.26, 0.17, 0.19, 0.18, 0.20],
          note='表 5　输入分辨率对整体性能的影响（各 7 组均值 / 最优）')
doc.para('输入尺寸由 320 提升到 640 后，测试集 mAP50 均值由 0.7557 升至 0.8951，'
         '提升 +0.1394；mAP50-95 均值由 0.3757 升至 0.5117，提升 +0.1360。'
         '最优组成绩由 0.7939 升至 0.9070。对比表 3 可见该提升覆盖全部 7 个模型配置，'
         '无例外。')
doc.para('从代价看，640 的显存占用峰值约 3.7 GB（320 约 0.9 GB），'
         '单组 60 epoch 训练平均耗时约 72 分钟（320 约 26 分钟），'
         '时间成本约为 2.8 倍，但在本项目 8 GB 显存的 RTX 4060 上完全可承受。'
         '考虑到精度提升幅度（+0.14）远大于优化器（+0.02）与注意力模块（±0.007）'
         '带来的差异，提高输入分辨率是本任务性价比最高的优化手段。')

# ---------------- 六、CBAM 消融 ----------------
doc.h1('六、CBAM 消融实验')
doc.para('以 YOLOv11s 为基线，分别在骨干网络（backbone 第 2/4/6/8 层）、'
         '颈部与 SPPF 模块（neck_sppf 第 9 层）以及两者同时插入 CBAM 注意力模块，'
         '在 320 与 640 两种分辨率下各做一组消融。CBAM 的注入采用子类化 '
         'DetectionTrainer 并重写 get_model() 的方式，在模型按 yaml 重建完成之后再注入；'
         '若按早期做法在 model.train() 之前对模型对象注入，会被训练器的重建过程丢弃，'
         '属于无效注入（曾表现为 +CBAM 与 baseline 参数量完全相同）。')
doc.table(T6_COLS, T6_ROWS, col_widths=[0.28, 0.16, 0.20, 0.16, 0.20],
          note='表 6　CBAM 插入位置消融（YOLOv11s，320 与 640 对比）')
doc.para('两种分辨率下 CBAM 均未表现出稳定增益，且增益幅度随分辨率提高而进一步缩小：')
doc.bullet('320×320：相对 baseline 0.7787，仅 neck_sppf 位置提升 +0.0152（0.7939），'
           'backbone 下降 −0.0427（0.7360）、both 下降 −0.0228（0.7559）。')
doc.bullet('640×640：相对 baseline 0.8992，backbone 提升 +0.0069（0.9061）、'
           'neck_sppf 提升 +0.0024（0.9016）、both 下降 −0.0032（0.8960），'
           '三处变化全部落在 ±0.007 以内。')
doc.bullet('640×640 下，+CBAM(backbone) 相对 baseline 略升 +0.0069（0.9061 vs 0.8992），'
           '是 v11 在 640 下的最高组；但 320 下同一位置为 −0.0427，方向不稳定，'
           '仍不构成稳定增益。')
doc.bullet('YOLOv12s 上的表现方向相反：320 下 +CBAM 由 0.7611 降至 0.7435（−0.0176），'
           '640 下则由 0.8682 升至 0.8874（+0.0192）。')
doc.para('原因推测为：本任务目标尺度小、样本量有限（训练集仅 1428 张），'
         '骨干浅层的空间注意力容易放大背景纹理响应，干扰细长线状目标的特征表达；'
         '而 640 输入下目标有效像素已充足，注意力机制可补充的信息本就不多，'
         '因此增益被进一步稀释。结论是：CBAM 的有效性高度依赖插入位置、'
         '任务特性与输入尺度，不宜作为通用涨点手段。')

# ---------------- 七、各类别结果 ----------------
doc.h1('七、各类别结果（640×640，测试集）')
doc.para('表 7 给出 640 一套全部 7 组模型在三个类别上的 mAP50。')
doc.table(T7_COLS, T7_ROWS, col_widths=[0.30, 0.17, 0.21, 0.16, 0.16],
          note='表 7　640×640 各组的分类别 test mAP50（按整体 mAP50 降序）')
doc.para('三类缺陷中，雷击（thunderbolt）检测效果最好，各组均达 0.961 以上，'
         '得益于其局部熔痕特征对比度高、类间区分度大；磨损（wear）次之，'
         '各组在 0.890~0.921 之间。')
doc.para('值得重点说明的是断股（break）：在 320×320 下它曾是唯一的性能瓶颈，'
         '7 组模型的 break mAP50 普遍落在 0.482~0.560 区间，与雷击相差约 0.40；'
         '而在 640×640 下，各组 break mAP50 升至 0.754~0.853，与雷击的差距收窄到'
         '约 0.12。断股为细长线状目标、像素占比低，断口处的单丝结构与正常导线纹理'
         '高度相似，320 输入下有效像素被进一步压缩；提高输入分辨率后其可用信息'
         '显著增加，瓶颈随之解除。此前「断股是类别级瓶颈」的判断只在 320 下成立。')

# ---------------- 八、训练曲线 ----------------
doc.h1('八、训练曲线')
doc.para('图 1、图 2 分别为 YOLOv11s 基线在 320×320（wire_v11_v3）与 '
         '640×640（wire_v11_640）下的 60 epoch 训练曲线，包含训练损失与验证集 mAP。'
         '两组仅输入尺寸不同，其余配置完全一致，可直接对比。')


def add_curves(doc, name, cap):
    """紧凑版：整块约 0.41 页高，两张曲线图可同页排布。"""
    if DEBUG:
        print(f'  [figure {name}] y={doc.y:.3f} need=0.42')
    doc._ensure(0.42)
    y1 = doc.y - 0.175
    ax1 = doc.fig.add_axes([0.135, y1, 0.78, 0.155])
    d = read_results(name)
    for k, lab in zip(['train/box_loss', 'train/cls_loss', 'train/dfl_loss'],
                      ['box loss', 'cls loss', 'dfl loss']):
        ax1.plot(d['epoch'], d[k], linewidth=1.3, label=lab)
    ax1.set_title('训练损失曲线', fontproperties=fp_h2, pad=5)
    ax1.set_xlabel('epoch', fontproperties=fp_note)
    ax1.set_ylabel('loss', fontproperties=fp_note)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=8)
    ax1.legend(fontsize=8, framealpha=0.9)

    y2 = y1 - 0.205
    ax2 = doc.fig.add_axes([0.135, y2, 0.78, 0.155])
    for k, lab in zip(['metrics/mAP50(B)', 'metrics/mAP50-95(B)'], ['mAP50', 'mAP50-95']):
        ax2.plot(d['epoch'], d[k], linewidth=1.5, marker='', label=lab)
    ax2.set_title('验证集 mAP 曲线', fontproperties=fp_h2, pad=5)
    ax2.set_xlabel('epoch', fontproperties=fp_note)
    ax2.set_ylabel('mAP', fontproperties=fp_note)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=8, framealpha=0.9)

    doc.y = y2 - 0.030
    doc.caption(cap)


add_curves(doc, 'wire_v11_v3',
           '图 1　YOLOv11s 基线训练曲线 @ 320×320（数据源：runs/wire_v11_v3）')
add_curves(doc, 'wire_v11_640',
           '图 2　YOLOv11s 基线训练曲线 @ 640×640（数据源：runs/wire_v11_640）')

# ---------------- 九、结论 ----------------
doc.h1('九、结论')
doc.bullet('分辨率是压倒性主变量：输入尺寸由 320 提升到 640，测试集 mAP50 均值由 '
           '0.7557 升至 0.8951（+0.1394，约 +0.14），且 7 个模型配置无一例外全部提升'
           '（+0.1071~+0.1860）。断股类从 0.48~0.56 升至 0.75~0.85，'
           '此前的类别级瓶颈被解除。')
doc.bullet('优化器影响次之且方向不一致：320 下 SGD 与 AdamW 的 7 组均值分别为 '
           '0.7557 与 0.7719（差 0.0162）；以最优组衡量则 SGD 略优'
           '（0.7939 vs 0.7908），val 均值同样是 SGD 略优（0.7291 vs 0.7270）。'
           '两者差距不足 0.02，不构成决定性因素；真正决定性的是学习率与优化器'
           '是否匹配——AdamW 错配 lr0=0.01 时同一模型的 mAP50 仅 0.635，'
           '换用 SGD+lr0=0.01 后升至 0.779（+0.144）。')
doc.bullet('CBAM 无稳定增益：320 下仅 neck_sppf 位置 +0.0152，backbone/both 为负；'
           '640 下三处变化全部在 ±0.007 以内；YOLOv12s 上 320 为 −0.0176、'
           '640 为 +0.0192。增益幅度随分辨率提高而稀释，方向随架构与位置变化，'
           '不能作为通用涨点手段。')
doc.bullet('双冠军，取决于指标口径：按 mAP50，640 的冠军是 YOLOv5s（0.9070），'
           'YOLOv11s 基线为 0.8992；按 mAP50-95（定位精度），冠军是 YOLOv11s 基线'
           '（0.5286），YOLOv5s 仅 0.500。即在 640 下一套中，'
           'mAP50 与 mAP50-95 给出的最优模型并不相同。')

# ---------------- 十、局限性 ----------------
doc.h1('十、局限性')
doc.bullet('验证集规模偏小（仅 178 张图片、275 个实例），单轮验证结果波动较大，'
           '且验证集与测试集给出的模型排序并不完全一致。本报告统一以测试集结论为准。')
doc.bullet('测试集仅 179 张图片、283 个实例，其中 break 150 个、thunderbolt 69 个、'
           'wear 64 个，样本量有限，单次评估的置信区间较宽，组间小于 0.01 的差异'
           '（如 CBAM 在 640 下的增益）不宜作强结论。')
doc.bullet('YOLOv5 官方的 val.py 输出精度为 3 位小数，故 v5 各组的 mAP50-95 只有 '
           '3 位有效数字，与 ultralytics 各组的 4 位不完全可比。')
doc.bullet('三套框架的其他超参数仍保留各自推荐默认值（数据增广强度、损失权重、'
           'warmup 策略等），仅对齐了优化器、初始学习率、输入尺寸与随机种子。'
           '论文中应注明「各框架采用各自推荐配置，通用超参保持一致」。')
doc.bullet('本报告未做分辨率的连续扫描（如 480、512、768），也未做训练时长扫描，'
           '640 是否已接近该任务的性能上限尚不确定；'
           '同时 640 下的推理耗时（约 6.5 ms/张，RTX 4060）约为 320 的 2.7 倍，'
           '在 Jetson Orin Nano 等边缘设备上部署时需重新评估实时性，'
           '必要时可考虑 640 训练 + 320/416 推理的知识蒸馏或输入降采样策略。')

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
        fig.savefig(os.path.join(PNG_DIR, f'v3_p{i + 1}.png'), dpi=110)
with PdfPages(OUT) as pdf:
    for fig in doc.pages:
        pdf.savefig(fig)
        plt.close(fig)

size = os.path.getsize(OUT)
print(f'OK  {OUT}')
print(f'页数 = {n}  大小 = {size} bytes ({size / 1024:.1f} KB)')
