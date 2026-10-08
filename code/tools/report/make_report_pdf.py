# -*- coding: utf-8 -*-
"""生成《基于YOLOv11+CBAM的缺陷检测算法》实验报告 PDF（含收敛曲线图）。"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

F_REG = r'C:\Windows\Fonts\msyh.ttc'
F_BOLD = r'C:\Windows\Fonts\msyhbd.ttc'

# 全局注册中文字体，供绘图坐标轴/图例使用
font_manager.fontManager.addfont(F_REG)
font_manager.fontManager.addfont(F_BOLD)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fp_title = FontProperties(fname=F_BOLD, size=17)
fp_head = FontProperties(fname=F_BOLD, size=12)
fp_body = FontProperties(fname=F_REG, size=9.5)
fp_tbl_head = FontProperties(fname=F_BOLD, size=9.5)
fp_tbl_body = FontProperties(fname=F_REG, size=9.5)
fp_sub = FontProperties(fname=F_REG, size=10.5)

PAGE_W, PAGE_H = 8.27, 11.69  # A4


def add_table(ax, col_labels, cell_text, bbox, col_widths=None):
    tbl = ax.table(
        cellText=cell_text, colLabels=col_labels,
        bbox=bbox, cellLoc='center', colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#333333')
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor('#dce6f1')
            cell.get_text().set_fontproperties(fp_tbl_head)
        else:
            cell.get_text().set_fontproperties(fp_tbl_body)
        cell.set_height(cell.get_height() * 1.15)
    return tbl


# ---------------- 数据 ----------------
# 表1 数据集分布
t1_cols = ['数据划分', '图片数', '断股\n(break)', '雷击\n(thunderbolt)', '磨损\n(wear)', '实例总数']
t1_rows = [
    ['训练集 train', '1428', '1014', '650', '788', '2452'],
    ['验证集 val', '178', '131', '80', '64', '275'],
    ['测试集 test', '179', '150', '69', '64', '283'],
    ['合计', '1785', '1295', '799', '916', '3010'],
]

# 表2 模型性能对比（test 集）
t2_cols = ['模型', '准确率\nP/%', '召回率\nR/%', 'mAP@0.5/%', 'mAP@0.5:0.95/%']
t2_rows = [
    ['YOLOv5s', '82.2', '66.4', '73.4', '33.5'],
    ['YOLOv11', '73.1', '54.6', '63.5', '28.6'],
    ['YOLOv11+CBAM\n(backbone)', '71.9', '54.6', '62.8', '26.9'],
    ['YOLOv11+CBAM\n(neck_sppf)', '66.0', '63.8', '66.2', '29.5'],
    ['YOLOv11+CBAM\n(both)', '74.6', '56.3', '63.8', '29.0'],
]

# 表3 YOLOv11+CBAM(both) 各类别监测结果（test 集）
t3_cols = ['类别', '准确率\nP/%', '召回率\nR/%', 'mAP@0.5/%', 'mAP@0.5:0.95/%']
t3_rows = [
    ['断股 break', '59.2', '30.0', '35.2', '13.2'],
    ['雷击 thunderbolt', '96.2', '88.4', '91.2', '47.5'],
    ['磨损 wear', '68.3', '50.5', '65.2', '26.3'],
    ['平均 all', '74.6', '56.3', '63.8', '29.0'],
]

# 表4 CBAM 插入位置消融实验对比
t4_cols = ['CBAM 插入位置', 'mAP@0.5\n验证/%', 'mAP@0.5:0.95\n验证/%', 'mAP@0.5\n测试/%', 'mAP@0.5:0.95\n测试/%']
t4_rows = [
    ['无 CBAM（baseline）', '56.0', '25.8', '63.5', '28.6'],
    ['backbone', '56.4', '24.8', '62.8', '26.9'],
    ['neck_sppf', '55.6', '25.8', '66.2', '29.5'],
    ['both（backbone+neck）', '57.5', '25.4', '63.8', '29.0'],
]


# ---------------- 正文文本 ----------------
sec1_title = '一、对比与原因'
sec1_body = (
    '对比结论：\n'
    '（1）测试集上 YOLOv5s 的 mAP@0.5 达 73.4%，全面领先 YOLOv11（63.5%）及各 CBAM 变体（62.8%~66.2%）。\n'
    '（2）核心差距集中在“断股”类别：YOLOv5s 为 54.9%，而 YOLOv11 系列仅 31.9%~36.7%，断股检出能力几乎翻倍。\n'
    '　　断股是架空输电线缺陷中最危险、最需可靠检出的一类，该差距具有决定性意义。\n'
    '（3）雷击（91.2%~97.1%）与磨损（63.0%~69.5%）两类各模型表现接近且整体较高，非性能瓶颈。\n'
    '原因分析：\n'
    '（1）数据层面：数据集由 SAM 自动标注并经 polygon→bbox 转换，框定位存在噪声。YOLOv5 的 anchor 机制对噪声框更鲁棒，\n'
    '　　YOLOv11 的 anchor-free 解码对目标中心与尺度估计更敏感，在“断股”这类细小、长条状目标上误差被放大。\n'
    '（2）训练层面：YOLOv11 参数量更大，60 epoch 可能欠拟合；YOLOv5 在同等 epoch 下收敛更充分。\n'
    '（3）CBAM 层面：CBAM 注入后仅带来 ±2% 的波动，且验证集与测试集上的最优变体不一致（验证集 both 最优、测试集 neck_sppf 最优），\n'
    '　　说明在当前数据量下注意力机制未学到稳定增益；同时 CBAM 使单图推理从约 2.3ms 增至 8.7~11ms（约 4~5 倍），\n'
    '　　对 Jetson Orin Nano 边缘部署是显著负收益。'
)

sec2_title = '二、改进方案'
sec2_body = (
    '（1）数据增强：断股目标偏小，提高输入分辨率（320→640）、引入 copy-paste 复制粘贴增强，并对断股类别过采样；\n'
    '　　人工复核 SAM 标注框，修正 polygon→bbox 转换引入的定位噪声。\n'
    '（2）断股专项优化：采用两阶段检测——先定位导线区域，再在区域内高分辨率检测断股，提升小目标检出率。\n'
    '（3）训练策略：YOLOv11 增加至 100~150 epoch、换更强预训练权重（yolo11m/l），优化学习率调度与 mosaic 关闭时机。\n'
    '（4）模型选型：以断股检出为最高优先级时，推荐 YOLOv5s 并辅以断股增强数据微调；\n'
    '　　CBAM 可在 neck 单独注入并配合更长训练，观察是否随数据量增加出现稳定增益，否则弃用。\n'
    '（5）部署优化：最终模型进行 TensorRT INT8 量化以适配 Jetson Orin Nano，实测推理延迟与量化精度损失，保障实时性。'
)


def new_page():
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig, ax


def load_v11_metrics(path):
    """返回 box,cls,dfl,map50,map5095 列表（YOLOv11 results.csv）"""
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r]
    data = rows[1:]
    box = [float(r[2]) for r in data]
    cls = [float(r[3]) for r in data]
    dfl = [float(r[4]) for r in data]
    m50 = [float(r[7]) for r in data]
    m5095 = [float(r[8]) for r in data]
    return box, cls, dfl, m50, m5095


pdf_path = r'D:\2\基于YOLOv11+CBAM的缺陷检测算法.pdf'
pdf = PdfPages(pdf_path)

# ================= 第 1 页 =================
fig, ax = new_page()
ax.text(0.5, 0.955, '基于 YOLOv11 + CBAM 的缺陷检测算法', ha='center', va='center', fontproperties=fp_title)
ax.text(0.5, 0.925, '—— 架空输电线缺陷检测实验报告（断股 / 雷击 / 磨损）', ha='center', va='center', fontproperties=fp_sub)

ax.text(0.08, 0.885, '表 1  导线缺陷数据集分布（实例数）', ha='left', va='center', fontproperties=fp_head)
add_table(ax, t1_cols, t1_rows, bbox=[0.08, 0.70, 0.84, 0.16])

ax.text(0.08, 0.66, '表 2  模型性能对比分析（测试集，179 张 / 283 框）', ha='left', va='center', fontproperties=fp_head)
add_table(ax, t2_cols, t2_rows, bbox=[0.08, 0.36, 0.84, 0.24])

fig.savefig(r'D:\2\_report_page1.png', dpi=130)
pdf.savefig(fig)
plt.close(fig)

# ================= 第 2 页 =================
fig, ax = new_page()
ax.text(0.08, 0.955, '表 3  YOLOv11+CBAM 模型各类别监测结果（测试集，both 变体）', ha='left', va='center', fontproperties=fp_head)
add_table(ax, t3_cols, t3_rows, bbox=[0.08, 0.76, 0.84, 0.15])

ax.text(0.08, 0.70, '表 4  CBAM 插入位置消融实验对比', ha='left', va='center', fontproperties=fp_head)
add_table(ax, t4_cols, t4_rows, bbox=[0.08, 0.48, 0.84, 0.16])

fig.savefig(r'D:\2\_report_page2.png', dpi=130)
pdf.savefig(fig)
plt.close(fig)

# ================= 第 3 页：收敛曲线 =================
fig2 = plt.figure(figsize=(8.2, 6.8))
box, cls, dfl, m50, m5095 = load_v11_metrics(r'D:\2\runs\wire_v11_v2\results.csv')
_, _, _, m50_both, _ = load_v11_metrics(r'D:\2\runs\wire_v11_both_v2\results.csv')
epochs = np.arange(1, len(box) + 1)

ax1 = fig2.add_subplot(2, 1, 1)
ax1.plot(epochs, box, label='box_loss', color='#1f77b4')
ax1.plot(epochs, cls, label='cls_loss', color='#ff7f0e')
ax1.plot(epochs, dfl, label='dfl_loss', color='#2ca02c')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('(a) YOLOv11 训练损失收敛曲线')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(alpha=0.3)

ax2 = fig2.add_subplot(2, 1, 2)
ax2.plot(epochs, m50, label='mAP@0.5（baseline）', color='#1f77b4')
ax2.plot(epochs, m5095, label='mAP@0.5:0.95（baseline）', color='#ff7f0e')
ax2.plot(epochs, m50_both, '--', label='mAP@0.5（CBAM-both）', color='#d62728')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('mAP')
ax2.set_title('(b) YOLOv11 验证精度收敛曲线')
ax2.legend(loc='lower right', fontsize=8)
ax2.grid(alpha=0.3)

fig2.suptitle('图 1  YOLOv11 训练过程损失与精度收敛曲线（真实训练数据）', y=0.98, fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig(r'D:\2\_report_page3.png', dpi=130)
pdf.savefig(fig2)
plt.close(fig2)

# ================= 第 4 页：对比与改进 =================
fig, ax = new_page()
y = 0.94
ax.text(0.08, y, sec1_title, ha='left', va='center', fontproperties=fp_head)
y -= 0.045
for line in sec1_body.split('\n'):
    ax.text(0.08, y, line, ha='left', va='center', fontproperties=fp_body)
    y -= 0.027

y -= 0.02
ax.text(0.08, y, sec2_title, ha='left', va='center', fontproperties=fp_head)
y -= 0.045
for line in sec2_body.split('\n'):
    ax.text(0.08, y, line, ha='left', va='center', fontproperties=fp_body)
    y -= 0.027

fig.savefig(r'D:\2\_report_page4.png', dpi=130)
pdf.savefig(fig)
plt.close(fig)

pdf.close()
print('PDF saved:', pdf_path)
