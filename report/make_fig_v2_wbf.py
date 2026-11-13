# -*- coding: utf-8 -*-
"""干净 100 图卷上 8 组推理配置的 mAP50 对比（v2 口径，2026-10-26）。

数据来自 D:/2/exp_upscale/logs/eval_wbf_tta_clean.tsv，由 eval_wbf_tta.py 生成。
输出 fig_v2_wbf.png（报告图 6-6 / 汇报用）。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# (标签两行, mAP50, 族, 是否含TTA)
DATA = [
    ('基线', '单模均值', 89.98, 'base', False),
    ('基线', '+ TTA', 90.05, 'base', True),
    ('基线', 'WBF', 90.63, 'base', False),
    ('基线', 'WBF + TTA', 90.97, 'base', True),
    ('Triplet', '单模均值', 90.79, 'trip', False),
    ('Triplet', '+ TTA', 89.64, 'trip', True),
    ('Triplet', 'WBF', 92.04, 'trip', False),
    ('Triplet', 'WBF + TTA', 90.39, 'trip', True),
]

BASE = 89.98
FLOOR = 1.60
BAND_TOP = BASE + FLOOR
BEST = 92.04

C_BASE = '#B7C8D9'
C_BASE_D = '#7C9AB8'
C_TRIP = '#A6C4AE'
C_TRIP_D = '#6E9C7C'
C_HL = '#C0392B'
C_TTA_FACE = '#E9EDF1'

fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=200)
x = np.arange(len(DATA))
w = 0.62

# 噪声带
ax.axhspan(BASE, BAND_TOP, color='#8C8C8C', alpha=0.16, zorder=0, lw=0)
ax.axhline(BASE, color='#333333', ls='--', lw=1.4, zorder=2)
ax.text(-0.42, BAND_TOP + 0.05, f'噪声地板 +{FLOOR:.2f} pt（越过此线才算真涨）',
        ha='left', va='bottom', fontsize=9.5, color='#4A4A4A')

vals = []
for i, (fam, sub, v, g, tta) in enumerate(DATA):
    is_best = abs(v - BEST) < 1e-9
    if is_best:
        face, edge, hatch = C_HL, '#7B1F14', None
    elif tta:
        face = C_TTA_FACE
        edge = C_BASE_D if g == 'base' else C_TRIP_D
        hatch = '///'
    else:
        face = C_BASE if g == 'base' else C_TRIP
        edge = C_BASE_D if g == 'base' else C_TRIP_D
        hatch = None
    b = ax.bar(x[i], v - (BASE - 0.6), bottom=BASE - 0.6, width=w, color=face,
               edgecolor=edge, linewidth=1.4, hatch=hatch, zorder=3)
    vals.append(v)
    if is_best:
        ax.text(x[i], v - 0.10, f'{v:.2f}', ha='center', va='top', fontsize=12.5,
                fontweight='bold', color='white', zorder=5)
    else:
        ax.text(x[i], v + 0.045, f'{v:.2f}', ha='center', va='bottom', fontsize=10.5,
                fontweight='bold', color='#1A1A1A', zorder=4)

# 最优柱的标注
ib = [i for i, d in enumerate(DATA) if abs(d[2] - BEST) < 1e-9][0]
ax.annotate('★ 唯一越过地板的配置',
            xy=(ib - 0.32, BEST - 0.02), xytext=(ib - 2.15, 92.52),
            ha='center', va='bottom', fontsize=10, color=C_HL, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_HL, lw=1.2,
                            connectionstyle='arc3,rad=-0.18'))

ax.set_xticks(x)
ax.set_xticklabels([f'{a}\n{b}' for a, b, *_ in DATA], fontsize=9.5)
ax.set_ylim(BASE - 0.6, 93.15)
ax.set_ylabel('mAP50（%，干净 100 图闭卷集；纵轴自 89.4 起）', fontsize=10.5)
import sys as _sys
TITLE = '--notitle' not in _sys.argv
if TITLE:
    ax.set_title('推理端后处理在干净考卷上的效果：基线 / TTA / WBF / WBF+TTA',
                 fontsize=12.5, fontweight='bold', pad=13)
ax.grid(axis='y', ls=':', color='#CCCCCC', lw=0.8, zorder=1)
ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

# 族分隔
ax.axvline(3.5, color='#999999', lw=0.9, ls=':')
ax.text(1.5, 92.96, '基线族（nf_v11_896）', ha='center', fontsize=10, color='#4A5E70', fontweight='bold')
ax.text(5.5, 92.96, 'TripletAttention 族', ha='center', fontsize=10, color='#3F6B4F', fontweight='bold')

legend = [Patch(facecolor=C_BASE, edgecolor=C_BASE_D, label='单模 / WBF 融合'),
          Patch(facecolor=C_TTA_FACE, edgecolor='#8A8A8A', hatch='///', label='含 TTA（已弃用）'),
          Patch(facecolor=C_HL, edgecolor='#7B1F14', label='本项目最高 92.04%'),
          plt.Line2D([], [], color='#333333', ls='--', lw=1.4, label=f'干净集基线 {BASE:.2f}%')]
ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.135), fontsize=9,
          frameon=False, ncol=4, handlelength=1.6, columnspacing=1.6)

NOTE = ('注：TTA 在本任务中弃用——1×/0.83×/0.67× 多尺度缩图导致小目标特征丢失，Triplet 三颗种子\n'
        '全部为负增益（−1.44 / −1.30 / −0.71 pt），3/3 同号；故 WBF 单打（92.04）反而优于 WBF+TTA（90.39）。')
fig.text(0.012, 0.030, NOTE, ha='left', va='bottom', fontsize=8.6, color='#444444', linespacing=1.55)

fig.tight_layout(rect=(0, 0.165, 1, 1))
out = (r'D:/2/report/_summary_figs/fig_v2_wbf.png' if TITLE
       else r'D:/2/report/_summary_figs/fig_v2_wbf_notitle.png')
fig.savefig(out, bbox_inches='tight', facecolor='white')
print('saved', out)
