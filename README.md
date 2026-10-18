# 架空输电线缺陷检测

基于 YOLOv11s 的输电线路三类缺陷检测（断股 / 雷击 / 磨损），输入 896×896。
项目包含分辨率对比、CBAM 与 TripletAttention 消融、WIoU v3 与磨损过采样单因素实验，
以及完整的误检/漏检归因。最终 test mAP50 **92.12%**、mAP50-95 **54.23%**
（NMS IoU 0.5，3 个随机种子均值）。

## 关键结果

test 集，3 个种子（42/43/44）均值，NMS IoU 0.7：

| 模型 | 640 | 896 |
|---|---:|---:|
| YOLOv11s 基线 | 88.42 | 90.79 |
| + CBAM（BN 重标定，neck/sppf） | 90.05 | 90.12 |
| + TripletAttention（neck/sppf） | 89.00 | **91.19** |
| YOLO26s 基线 | — | 91.35 |

最终对外口径采用 NMS IoU 0.5（在 val 上独立选定，非 test 调参）：
**92.12% [91.97, 92.21]**，mAP50-95 54.23%，P 91.68%，R 85.17%。

## 关键发现

- **分辨率是主变量**：896 相比 640，基线 mAP50 提升 2.36 个点。
- **注意力没有稳定增益**：CBAM 在 896 反而低于基线；Triplet 在 896 高 0.40 个点，
  但种子区间重叠，640 与 896 方向也不一致，不能宣称有效。
- **mAP50 与 mAP50-95 分化**：mAP50 到 92%，mAP50-95 只有 54%，说明“找到”不难、
  “框准”仍有明显差距；误差主要来自 break 的低置信度漏检与重复框。
- **数据存在近似重复**：test 179 张中 48 张在 train 有近重复，指标应理解为同场景泛化，
  而不是全新场景泛化。
- **负结果同样记录**：WIoU v3（+0.19）、磨损 2× 过采样（+0.75）均只有单种子，
  按预注册规则未追加种子、不写入主结果；P2 小目标头因部署设备算力不足而未执行。

## 网络结构

![YOLOv11s + TripletAttention](report/figures/yolo11_triplet_arch.png)

TripletAttention（WACV 2021）插在 backbone 的 SPPF（第 9 层）之后、C2PSA（第 10 层）之前，
分 C×H、C×W、H×W 三支做注意力，三支输出取平均。每支为
`ZPool → 7×7 Conv(bias=False) → BN → Sigmoid`，全模块仅 300 参数
（baseline 9,428,953 → 9,429,253，nc=3）。

结构图的 SVG / PDF / draw.io 源文件见 [report/figures/](report/figures/)，
可继续在 draw.io 或 Illustrator 中编辑。

## 数据集

- 3 类：`break`（断股）、`thunderbolt`（雷击）、`wear`（磨损）
- 1,785 张：train 1,428 / val 178 / test 179（8:1:1，seed 42）
- 来源：教师项目提供的 1,800 张现场图像，模型预标注 + MobileSAM 辅助，人工复核；
  15 张无目标图像剔除
- 因版权与体积，数据集、权重、训练产物未上传

## 复现

环境：Python 3.12，ultralytics 8.3.180，PyTorch 2.11（CUDA 12.8），RTX 4060。
训练统一为 60 epochs、batch 8、SGD（lr0=0.01、lrf=0.005、cos_lr）、AMP、`workers=0`（Windows）。

```bash
# 896 基线（单种子）
python code/yolov11/train_att_640.py --att baseline --imgsz 896 --seed 42

# TripletAttention 臂
python code/yolov11/train_att_640.py --att triplet --mode neck_sppf --imgsz 896 --seed 42

# CBAM + BN 重标定臂
python code/yolov11/train_att_640.py --att cbam_bn --mode neck_sppf --imgsz 896 --seed 42
```

数据路径在各训练脚本顶部的 `BASE` / `DATA` 常量中，按本机布局修改。
评估脚本在 [code/tools/eval/](code/tools/eval/)，各 run 的完整命令记录在
[WORKLOG.md](WORKLOG.md) 与 [DIRECTORY.md](DIRECTORY.md) 中。

## 目录结构

完整索引见 **[DIRECTORY.md](DIRECTORY.md)**（含各目录状态与“勿动”标记）。简表：

- [code/](code/)：训练、评估、数据处理脚本
  - [code/yolov11/](code/yolov11/)：主实验；`triplet_att.py`、`cbam_bn.py`、`wiou_v3.py` 为模块实现
  - [code/yolov12/](code/yolov12/)：YOLOv12 对照
  - [code/upscale/](code/upscale/)：Triplet/CBAM 探针、CPM 磨损数据生成
  - [code/tools/](code/tools/)：参数/注入/切分核查、评估、数据处理
- [exp_upscale/](exp_upscale/)：第二批起的新实验区（本仓库只保留配置与脚本）
- [report/](report/)：实验报告 PDF、结构图、数据汇总与生成脚本
- [scripts/](scripts/)：串行实验链入口
- WORKLOG.md：项目全量工作记录（含失败实验与决策过程）
- DIRECTORY.md：目录与产物地图
- PROTOCOL.md：三方协作协议

## 说明

- 所有对比固定数据切分、训练轮数与评估口径；val 用于选择，test 只做最终评估。
- `runs/` 为封版阶段产物，`exp_upscale/runs/` 为后续实验产物；训练产物均未入库。
- 各框架采用各自推荐配置，通用超参（优化器、初始学习率、epochs、batch、imgsz）保持一致。
