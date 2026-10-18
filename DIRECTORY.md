# D:\2 目录总览

> 本文件是 D:\2 的目录索引，2026-10-17 建立。新增顶层目录/批次时先在此登记。
>
> **整理原则**：只加索引，不搬历史产物。原因见 §七——大量路径已被 WORKLOG、报告与
> 脚本逐字引用，移动或改名会让审计链断裂。本目录树中的"整理"指**可读、可查、可判状态**，
> 不是把文件重新摆放。

## 一、先读哪几份

| 顺序 | 文件 | 作用 |
|---|---|---|
| 1 | `PROTOCOL.md` | 三方协作协议、角色与纪律（用户 = 转述，DeepSeek = 规划，Claude = 本机执行） |
| 2 | `WORKLOG.md` | 项目全量工作记录（原文形态，最新为 A.9.24） |
| 3 | `DIRECTORY.md` | 本文件：目录与产物地图 |
| 4 | `README.md` | 对外简介（主结果、关键发现、复现入口、目录） |

## 二、根目录

| 路径 | 体积 | 状态 | 说明 |
|---|---:|---|---|
| `code/` | 53.5 MB | 活跃 | 训练、评估、数据处理、报告生成脚本（见 §三） |
| `scripts/` | 0.1 MB | 活跃 | 串行实验链入口（见 §五） |
| `exp_upscale/` | 3.8 GB | **活跃实验区** | 第二批/第三批全部实验产物（见 §四） |
| `runs/` | 2.2 GB | **封版，勿动** | v_final 所依据的基线/消融训练结果 |
| `logs/` | 197 MB | 只读历史 | 封版阶段的训练、评估、探针日志（83 项） |
| `report/` | 31 MB | 交付区 | PDF、生成脚本、逐页预览、混淆矩阵（见 §六） |
| `archive/` | 0.3 MB | 只读历史 | 早期逐日工作记录（2026-09-27 → 10-12）与 AdamW 旧报告 |
| `dataset_final_bbox/` | 129 MB | **主数据集，勿动** | 3 类 1785 张：train 1428 / val 178 / test 179；`data.yaml` 在此 |
| `dataset_archive/` | 446 MB | 只读历史 | 数据集合并/切分前的各版本（dataset、merged、new、148_auto…） |
| `dataset_leak_subsets/` | 168 MB | 可重建 | P0 泄漏量化子集；由 `code/tools/make_leak_subsets.py` 从主数据集重建 |
| `dataset_cpm_wear/` | 124 MB | 派生数据 | CPM 磨损增强集；由 `code/upscale/make_cp_wear.py` 生成 |
| `models/` | 58 MB | 权重 | `yolo26s.pt`（YOLO26 环境预置）、`mobile_sam.pt` |
| `weights/` | 0 | 空目录 | 历史遗留；未使用 |
| `yolo11n.pt` | 5.4 MB | **证据文件，勿动** | 根目录残留权重；DSH 文件系统事件的取证对象，路径本身有意义 |
| `envs/` | 4.9 GB | 本地环境 | `yolov5-env`（YOLOv5 对照组环境） |
| `ultralytics_wire_main/` | 312 MB | **只读** | 老师原项目（含 .rar），未修改；`.gitignore` 已排除 |
| `pyqt5_demos/` | 0.8 MB | 无关 | PyQt5 相机/检测界面演示，不属于论文实验 |
| `.git/` | 8.5 MB | 版本库 | 本地提交历史 |
| `.claude/` | — | 会话配置 | 本机 Claude 会话与设置；不手改 |
| `.gitignore` | — | — | 数据、runs、logs、envs、权重、预览图不入库 |

## 三、`code/`

| 子目录 | 内容 |
|---|---|
| `yolov11/` | 主实验训练脚本。基线 `train_v11_640.py`；统一三臂入口 `train_att_640.py`（baseline / cbam_bn / triplet）；`train_yolo26_896.py`；注意力实现 `cbam.py`、`cbam_bn.py`、`cbam_fixed.py`、`triplet_att.py`；第一批 `train_v11_batch1.py`；旧消融/优化器脚本 |
| `yolov12/` | YOLOv12 对照：`train_v12_640.py`、`train_v12_640_cbam.py`、消融脚本 |
| `baselines/yolov5/` | YOLOv5 对照组（嵌套 git 仓库，仅本地保留） |
| `baselines/yolov5_wire/` | 本项目自写的 YOLOv5 训练脚本 |
| `tools/` | 闸门与工具：`check_*.py`（参数/注入/切分/泄漏/连续性）、`viz_cbam_attention.py`、`make_leak_subsets.py`、`make_probe0_ckpts.py` |
| `tools/eval/` | 各口径评估：`eval_test_640.py`、`eval_test_896.py`、`eval_att_640.py`、`eval_att_batch2.py`、`eval_yolo26_896.py`、泄漏子集评估、per-class 解析 |
| `tools/data/` | 数据构建：JSON→YOLO、polygon→bbox、切分/合并、预览 |
| `tools/report/` | 旧版报告生成 `make_report_pdf.py` |
| `upscale/` | 第二批探针：Triplet vs CBAM 的 DC/尺度对比、VRAM、GT 框统计、CPM 数据生成、batch1 评估 |
| `app/`、`modules/`、`scripts/`、`src/` | **空目录**（早期脚手架，未使用） |

## 四、`exp_upscale/`（第二批起的新实验区）

| 路径 | 内容 |
|---|---|
| `runs/<run>/` | 训练产物（weights、results.csv、曲线）。**不要移动**：TSV/报告按 run 名回查 |
| `eval/<run>_<split>/` | 评估产物，按 run+split 独立成目录，不混进训练目录 |
| `logs/` | 判读与账本：`batch2_eval.tsv`、`batch3_eval.tsv`（主判据）、`batch*_state.tsv`（机器账本）、`batch*_ledger.md`（人读账本）、`chain_batch*_*.log`（原始训练日志）、`A9xx_*.md`（WORKLOG 片段）、探针输出 |
| `smoke/` | 链路/注入/续训冒烟测试（6 组，713 MB），非正式实验 |

**判读唯一入口**：`batch2_eval.tsv` / `batch3_eval.tsv`。同一 `(run, split)` 多行时取**最后一行**
（事故坏行保留在前，正确行后追加）。每行先用 `results.csv` 的 val mAP50 峰值自检，
差值 > 0.05 视为陈旧行，必须重评。

命名约定：
- 训练 run：`nf_<model>_<imgsz>_s<seed>`（地板）、`att_v11_<imgsz>_<att>_s<seed>`（注意力臂）
- 批次账本：`batchN_eval.tsv` / `batchN_state.tsv` / `batchN_ledger.md`

## 五、`scripts/`

| 脚本 | 状态 | 说明 |
|---|---|---|
| `run_batch3.sh` | **最新正式链** | 第三批：YOLO26@896 补种子 + 896 Triplet 补种子；可重入、断点续训、每 run 落账 |
| `_chain3_entry.sh` | 入口包装 | 调用 `run_batch3.sh` 的会话入口 |
| `run_att_chain_batch2.sh` | 已完成 | 第二批链（640/896 的 CBAM+BN / Triplet 补种子） |
| `run_att_chain.sh`、`run_batch1.sh` | 已完成 | 第一批链与单批入口 |
| `run_batch3.sh.bak_20261017_before_logfix` | **备份，勿删** | 修复日志双写/GBK 混入前的原件 |
| `run_all_*.sh`、`run_e20_*.sh`、`run_e60_fixed.sh`、`run_probe5_control.sh`、`watch_ep40_bias.sh` | 历史 | 封版阶段的批量/消融/监控脚本 |
| `fix_settings.py` | 工具 | 修复 ultralytics `settings.json` |

纪律：链脚本一旦启动即冻结，运行中不改；改前先留 `.bak_<时间>_<原因>`。

## 六、`report/`

### 交付 PDF

| 文件 | 生成日期 | 状态 |
|---|---|---|
| `架空输电线YOLOv11+CBAM缺陷检测算法_v_final.pdf` | 2026-10-13 | **已封版，勿改** |
| `第二批CBAM修复实验_896_探索报告_v1.pdf` | 2026-10-16 | 交付 v1；§七 判据读法有误，见下条勘误 |
| `第二批CBAM修复实验_896_探索报告_v1_勘误_v1.pdf` | 2026-10-17 | **勘误**：只更正 §七 的判据读法，数值与结论不变 |
| `第三批YOLO26_896_与Triplet补种子_探索报告_v1.pdf` | 2026-10-17 | 第三批探索报告（6 页） |
| `模型改进与数据对照_v1.pdf` | 2026-10-17 | 对外汇总：模型/改进/数据对照 |
| `参考文献结构图合集_v1.pdf` | 2026-10-19 | 论文结构图合集（14 页）；**版权原因不入库，仅本地** |
| `面试项目资料_人工智能研发工程师助理_v1.pdf` | 2026-10-19 | 个人面试材料（21 页）；**仅本地，不入库** |

### 生成脚本

| 脚本 | 输出 | 注意 |
|---|---|---|
| `make_report_v5.py` | v_final | **无 `__main__` 保护：import 即重生成封版 PDF** |
| `make_report_batch2.py` | batch2 v1 | 同上；会重跑混淆矩阵 |
| `make_report_batch3.py` | batch3 v1 | 同上；读两份 TSV 并做陈旧行断言 |
| `make_erratum_batch2.py` | 勘误 v1 | 只读 `batch2_eval.tsv`，不碰批 2 原 PDF |
| `make_report_v2/v3/v4.py` | 历史版本 | 产物在 `archive/` |
| `make_model_pdf.py` | 模型改进与数据对照 v1 | 读 `model_data_v0.json` |
| `make_arch_diagram.py` | `figures/yolo11_triplet_arch.*` | 生成 PNG/SVG/PDF/drawio/HTML 五份 |
| `make_refs_pdf.py` | 参考文献结构图合集 v1 | 依赖 `References/`；**产物不入库** |
| `make_refs_gallery.py` | `References/_gallery/` 候选页 | 依赖 `References/`；本地工具 |
| `make_interview_pdf.py` | 面试资料 v1 | 个人材料；**脚本与产物均不入库** |
| `make_interview_figs.py` | `figures/interview_*.png` | 个人材料配图；**不入库** |

### `report/figures/`（入库）

| 文件 | 说明 |
|---|---|
| `yolo11_triplet_arch.png/.svg/.pdf/.drawio/.html` | YOLOv11s + TripletAttention 结构图，`drawio`/`html` 可继续编辑 |

逐页 PNG 必须显式设 `REPORT_PNG=<目录>` 才导出（AI 规则 8）；不设变量不得声称"已验证"。

### 其他

- `_preview*/`、`_pdf_preview/`：各版报告的逐页 PNG（可再生成，已 gitignore）
- `_batch2_confusion_896/`：第二批 3 个 run 的混淆矩阵与 JSON
- `_cbam_attention/`：CBAM 注意力可视化
- `_phase0/`、`_phase1/`：Phase 0 诊断与 Phase 1 复评的 JSON/中间产物
- `_refs_pdf/`：参考文献合集渲染中间页
- `archive/`：v2–v5.2 历史 PDF

## 七、为什么不做物理搬移

以下路径被 WORKLOG、报告正文或脚本逐字引用，移动/改名会破坏复现与审计：

- `report/_preview_batch2/`、`report/_batch2_confusion_896/`（batch2 报告与 WORKLOG 引用）
- `scripts/run_batch3.sh.bak_20261017_before_logfix`（WORKLOG 记录原件）
- `D:\2\yolo11n.pt`（DSH 文件系统事件的取证对象）
- `exp_upscale/runs/<run>/`、`eval/<run>_<split>/`（TSV 与报告按名字回查）
- `logs/`、`runs/`、`dataset_final_bbox/`（封版/只读）

且项目纪律为「仅追加不删除」「不可逆改名先问」。因此本次整理**只新增索引文件，未移动、
未改名、未删除任何既有文件**。若将来确需物理归并，应单独立项，同步改引用并留备份。

## 八、命名约定（向后适用）

- 新实验只落 `exp_upscale/`，不写入封版 `runs/`
- 报告：`<主题>_v<N>.pdf`；勘误：`<原报告>_勘误_v<N>.pdf`
- 备份：`<原文件名>.bak_<YYYYMMDD>_<原因>`
- WORKLOG 片段：`exp_upscale/logs/A9xx_<用途>.md`，追加后保留片段原件
