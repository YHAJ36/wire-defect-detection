# code/ —— 训练、评估与数据处理

按目录分四块。所有路径常量的基址写在脚本顶部的 `BASE` / `DATA`，搬到别的机器先改这两行。

## yolov11/ —— 主实验

| 文件 | 作用 |
|---|---|
| `train_v11_640.py` | 基线训练入口（早期封版用） |
| `train_att_640.py` | **统一三臂入口**：`--att baseline\|cbam_bn\|triplet`、`--mode neck_sppf`、`--imgsz`、`--seed` |
| `run_batch_noleak.sh` | 段级无泄漏重训链，按 `results.csv` 行数判进度，可断点续训 |
| `check_noleak_chain.sh` | 上述链的闸门（检查 6 个 run 是否齐全、有无半截产物） |
| `cbam.py` / `cbam_bn.py` / `triplet_att.py` / `wiou_v3.py` | 模块实现（插入点均为 SPPF 之后、C2PSA 之前） |
| `train_yolo26_896.py`、`p2_transfer.py`、`train_perturb.py` | YOLO26 对照、P2 小目标头迁移、扰动实验 |

## yolov12/ —— 对照

`train_v12_640.py`（基线）、`train_v12_640_cbam.py`（+CBAM）。

## baselines/ —— YOLOv5 对照组

- `yolov5_wire/`：本项目自写的 v5 训练脚本（`train_k210.py` 等的来源）
- `yolov5/`：官方仓库副本，**不入库**（体积大），需要时自行 `git clone ultralytics/yolov5`
- K210 用的结构改造（`SPPFK` / `Up2`）追加在 `yolov5/models/common.py`，改法见 `k210/K210部署总结.md`

## tools/ —— 闸门、评估、数据处理

| 子目录 | 内容 |
|---|---|
| `tools/` | `check_*.py` 参数/注入/切分/泄漏核查、注意力可视化 |
| `tools/eval/` | 各口径评估。**`eval_noleak.py` 是 v3 口径唯一来源**（段级留出集）；其余为历史口径与消融 |
| `tools/data/` | **切分三件套**：`build_noleak_dataset.py`（dHash 双闸聚类 → 1080 视觉段）、`split_by_scene.py`（段级 8:1:1）、`split_by_scene_strat.py`（六目标模拟退火配平，待重训） |
| `tools/diag/`、`tools/fuse/` | 诊断与 WBF 融合 |

## 跑之前注意

- 评估一律 `encoding='utf-8-sig'` + `delimiter='\t'` 读 TSV（文件带 BOM）
- YOLO26 用 ultralytics 8.4.178，v11/v12 用 8.3.180，**混用不报错但差 9.85 pt**
- 各 run 的完整命令与判读标准记在 `WORKLOG.md`，目录状态见 `DIRECTORY.md`
