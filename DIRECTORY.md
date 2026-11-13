# D:\2 目录总览

> 本文件是 D:\2 的目录索引，2026-10-17 建立，2026-10-27 修订（补 noleak 三目录、报告 v3 交付物、口径纪律），2026-11-13 修订（补 K230/K210/安全帽三套目录、根目录散文件归位，见 §九）。新增顶层目录/批次时先在此登记。
>
> **整理原则**：只加索引，不搬历史产物。原因见 §七——大量路径已被 WORKLOG、报告与
> 脚本逐字引用，移动或改名会让审计链断裂。本目录树中的"整理"指**可读、可查、可判状态**，
> 不是把文件重新摆放。

## 一、先读哪几份

| 顺序 | 文件 | 作用 |
|---|---|---|
| 1 | `PROTOCOL.md` | 三方协作协议、角色与纪律（用户 = 转述，DeepSeek = 规划，Claude = 本机执行） |
| 2 | `WORKLOG.md` | 项目全量工作记录（原文形态，最新为 A.9.90） |
| 3 | `DIRECTORY.md` | 本文件：目录与产物地图 |
| 4 | `README.md` | 对外简介（主结果、关键发现、复现入口、目录） |

**读数字前必须先看口径（2026-10-27 定）**

| 说法 | 能不能用 |
|---|---|
| 跨段泛化 **74.7% ~ 89.2% 区间**（test 203 图 / val 181 图，三种子） | **唯一现行口径**。写结论、写 PPT、写论文都用这个 |
| 92.12%（179 图，含 44.1% 同段泄漏）、89.98%（帧级隔离 100 图）、94.38%（探针） | **只作历史对照**，不得单列为主结果 |
| 13 个探索方向的 Δ 值（同一张 179 图卷上互比） | 方向性结论有效；绝对值受泄漏影响，偏乐观 |

原因：段级重划分后，只换「留出哪批视觉段」就差 13.2 pt，而三种子之间的跨度只有
1.60 pt（噪声地板）。**同一颗权重在不同卷上的差值，不是方法差异。** 详见 WORKLOG A.9.62、
A.9.63 与报告 6.13.6。

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
| `dataset_noleak/` | 136 MB | **现行主数据集** | 按 1080 个视觉段整体重划：train 1401 / val 181 / test 203，跨段泄漏 0；`data.yaml` 在此，`assign.json` 是分配表（含 `seg_of`）；已重训 6 个 run（v3 口径） |
| `dataset_noleak_clean/` | 21 MB | 只读，**禁止训练** | 帧级隔离版（val 92 / test 100），仅用于纯推理对照；⚠ `data.yaml` 的 `train` 占位指向 `dataset_final_bbox/images/train`，拿它训练 = 把段级 test 的邻居图喂回训练集 |
| `dataset_noleak_strat/` | 260 KB | **待物化** | 六目标模拟退火配平划分（train 1409 / val 173 / test 203，0 项越界）；当前只有 `assign_strat.json`，图/标签需 `split_by_scene_strat.py --apply` 才落地；尚未重训 |
| `models/` | 58 MB | 权重 | `yolo26s.pt`（YOLO26 环境预置）、`mobile_sam.pt` |
| `weights/` | — | **已删除** | 历史遗留空目录，2026-11-13 删除 |
| `yolo11n.pt` | 5.4 MB | **证据文件，勿动** | 根目录残留权重；DSH 文件系统事件的取证对象，路径本身有意义 |
| `envs/` | 4.9 GB | 本地环境 | `yolov5-env`（YOLOv5 对照组环境） |
| `ultralytics_wire_main/` | 312 MB | **只读** | 老师原项目（含 .rar），未修改；`.gitignore` 已排除 |
| `deploy/` | 262 MB | 已收尾 | **K230** 部署目录（立创庐山派 K230，CanMV v1.8，nncase 2.11.0，COM8）。生产版 `cam_demo3.py`（5.0 fps）、`yolo11n_640_u8r.kmodel`（3.44 MB）；过程见 WORKLOG A.9.75~A.9.84 |
| `manual/` | 17 MB | 教程 | **K230** 手动转换四步（01_export_onnx → 02_check → 03_compile → 04_verify），逐步骤「看什么数字才算过」；含 `DOTNET_ROLL_FORWARD` 环境坑 |
| `k210/` | 258 MB | **已收尾，板已归还** | **K210** 全链路（训练→导出→编译→上传→板上推理）。总览见 `k210/K210部署总结.md`；最终模型 `v5/strike.kmodel`（雷击 61.0%）、`v5/best_k210.kmodel`（断股 25.2%）；失败中间产物已于 2026-11-13 删除（约 172 MB，见 §九） |
| `dataset_helmet/` | 2 MB | 素材库 | 安全帽数据。现行训练集在 `k210/hat/dataset/`（1,011/203 裁片）；本目录 `_raw/labels.zip` 是 SHWD YOLO 标签（7,571 条）；两个误下的摩托车头盔集与中断的原图下载已于 2026-11-13 删除（约 155 MB，见 §九）。说明见 `dataset_helmet/README.md` |
| `_archive/` | — | **已删除** | 2026-11-13 建立（收 nncase 编译 dump），同日按「没用的不留下」删除 |
| `pyqt5_demos/` | 0.8 MB | 无关 | PyQt5 相机/检测界面演示，不属于论文实验；**2026-11-13 起**另有 `界面改造说明_零基础.md` 与 `界面效果图_老师原版.jpg`（老师给的效果图，原文件名是 MD5 哈希） |
| `.git/` | 8.5 MB | 版本库 | 本地提交历史 |
| `.claude/` | — | 会话配置 | 本机 Claude 会话与设置；不手改 |
| `.gitignore` | — | — | 数据、runs、logs、envs、权重、预览图不入库 |

## 三、`code/`

| 子目录 | 内容 |
|---|---|
| `yolov11/` | 主实验训练脚本。基线 `train_v11_640.py`；统一三臂入口 `train_att_640.py`（baseline / cbam_bn / triplet）；`train_yolo26_896.py`；注意力实现 `cbam.py`、`cbam_bn.py`、`cbam_fixed.py`、`triplet_att.py`；第一批 `train_v11_batch1.py`；**段级无泄漏链** `run_batch_noleak.sh`（按 `results.csv` 行数判进度、可续训）+ 闸门 `check_noleak_chain.sh`；旧消融/优化器脚本 |
| `yolov12/` | YOLOv12 对照：`train_v12_640.py`、`train_v12_640_cbam.py`、消融脚本 |
| `baselines/yolov5/` | YOLOv5 对照组（嵌套 git 仓库，仅本地保留） |
| `baselines/yolov5_wire/` | 本项目自写的 YOLOv5 训练脚本 |
| `tools/` | 闸门与工具：`check_*.py`（参数/注入/切分/泄漏/连续性）、`viz_cbam_attention.py`、`make_leak_subsets.py`、`make_probe0_ckpts.py` |
| `tools/eval/` | 各口径评估：`eval_test_640.py`、`eval_test_896.py`、`eval_att_640.py`、`eval_att_batch2.py`、`eval_yolo26_896.py`、泄漏子集评估、per-class 解析；**`eval_noleak.py`**（段级留出集，v3 口径唯一来源） |
| `tools/data/` | 数据构建：JSON→YOLO、polygon→bbox、切分/合并、预览。**切分三件套**：`build_noleak_dataset.py`（dHash 双闸聚类 → 1080 视觉段）、`split_by_scene.py`（段级 8:1:1，逐段随机，已训）、`split_by_scene_strat.py`（六目标模拟退火配平，待训） |
| `tools/report/` | 旧版报告生成 `make_report_pdf.py` |
| `upscale/` | 第二批探针：Triplet vs CBAM 的 DC/尺度对比、VRAM、GT 框统计、CPM 数据生成、batch1 评估 |
| `app/`、`modules/`、`scripts/`、`src/` | **空目录**（早期脚手架，未使用） |

## 四、`exp_upscale/`（第二批起的新实验区）

| 路径 | 内容 |
|---|---|
| `runs/<run>/` | 训练产物（weights、results.csv、曲线）。**不要移动**：TSV/报告按 run 名回查 |
| `eval/<run>_<split>/` | 评估产物，按 run+split 独立成目录，不混进训练目录 |
| `logs/` | 判读与账本：`batch2_eval.tsv`、`batch3_eval.tsv`（主判据）、`batch*_state.tsv`（机器账本）、`batch*_ledger.md`（人读账本）、`chain_batch*_*.log`（原始训练日志）、`A9xx_*.md`（WORKLOG 片段）、探针输出 |
| `logs/eval_noleak*.{tsv,json}` | **v3 口径唯一来源**：`eval_noleak_final.tsv` 是 6 个段级 run 的最终评估表（含 WBF / TTA 行）；`eval_noleak.tsv` 是历史过程表 |
| `smoke/` | ~~链路/注入/续训冒烟测试（6 组，713 MB）~~ **已删除（2026-11-13）**；另有 `runs/smoke_noleak_base`、`runs/smoke_noleak_triplet`（118 MB）同日删除。WORKLOG A.9.7x 原文即标注「可清理垃圾（未删，等确认）」 |

**判读唯一入口**：`batch2_eval.tsv` / `batch3_eval.tsv`。同一 `(run, split)` 多行时取**最后一行**
（事故坏行保留在前，正确行后追加）。每行先用 `results.csv` 的 val mAP50 峰值自检，
差值 > 0.05 视为陈旧行，必须重评。段级无泄漏实验（`*_noleak_*`）不看这两张表，读
`logs/eval_noleak_final.tsv`。

⚠ 读 TSV 一律 `encoding='utf-8-sig'`（有 BOM）+ `delimiter='\t'`；漏掉任一条会报
`KeyError` 或把整行读成一列。

命名约定：
- 训练 run：`nf_<model>_<imgsz>_s<seed>`（地板）、`att_v11_<imgsz>_<att>_s<seed>`（注意力臂）
- 段级无泄漏 run：`nf_v11_noleak_s<seed>`、`att_v11_noleak_triplet_s<seed>`（无 imgsz 段，固定 896）
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
| `report/实验报告_全文.docx` = `report/架空输电线路缺陷检测_实验报告.docx` | 2026-10-27 | **主交付物**（107 页 / 6.0 万字 / 120 表）；两个文件名同内容（md5 一致），桌面另有同名副本。改报告 = 就地改 docx，**不要重建** |
| `report/架空输电线路缺陷检测_实验报告_v3.pdf` | 2026-10-27 | 上述 docx 的 PDF 版（107 页）；口径 v3 |
| `report/架空输电线路缺陷检测_实验报告_v2.pdf` | 2026-10-26 | 历史。v2 口径（89.98 / 90.79），**已被 v3 取代** |
| `report/汇报版_输电线缺陷检测_总结.pdf` | 2026-10-27 | 汇报版（11 页 / 0.88 MB）；由 `make_summary_doc.py` → docx → Word 导出。桌面有副本 |
| `report/汇报版_输电线缺陷检测_2026总结.pdf` | 2026-10-26 | **已作废**：v2 口径（92.12%，10 页），仅留档，不要对外发送。同名 `.docx` 是旧脚本产物，与正式 docx 文本完全相同（151 段 / 8390 字，已比对），可随时删 |
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
| `make_summary_doc.py` | `汇报版_..._总结.docx` | **现行入口**；直接写规范名，无需再 `cp`（2026-10-27 改）。导出 PDF 用 Word → `ExportAsFixedFormat(17)` |
| `make_summary_doc2.py` | —— | **已作废**（v2 口径），开头已加 `sys.exit` 拦截：误跑会覆盖 v3 正式 docx |
| `patch_v3.py` | 就地改 `实验报告_全文.docx` | 把 v2 → v3 口径的 14 处数字/结论同步过去；数字全部从 `eval_noleak_final.tsv` 读，不手打。改前自动留 `.bak` |
| `make_report_v5.py` | v_final | **无 `__main__` 保护：import 即重生成封版 PDF** |
| `make_report_batch2.py` | batch2 v1 | 同上；会重跑混淆矩阵 |
| `make_report_batch3.py` | batch3 v1 | 同上；读两份 TSV 并做陈旧行断言 |
| `make_erratum_batch2.py` | 勘误 v1 | 只读 `batch2_eval.tsv`，不碰批 2 原 PDF |
| `make_report_v2/v3/v4.py` | 历史版本 | 产物在 `archive/` |
| `make_model_pdf.py` | 模型改进与数据对照 v1 | 读 `model_data_v0.json` |
| `make_arch_diagram.py` | `figures/yolo11_triplet_arch.*` | 生成 PNG/SVG/PDF/drawio/HTML 五份（v1） |
| `make_arch_diagram_v2.py` | `figures/yolo11_triplet_arch_v2.*` | 同上五份（v2，沿用 v1 扁平风格重写；v1 脚本未改）。⚠ **未入库**，见 A.9.30 |
| `make_refs_pdf.py` | 参考文献结构图合集 v1 | 依赖 `References/`；**产物不入库** |
| `make_refs_gallery.py` | `References/_gallery/` 候选页 | 依赖 `References/`；本地工具 |
| `make_interview_pdf.py` | 面试资料 v1 | 个人材料；**脚本与产物均不入库** |
| `make_interview_figs.py` | `figures/interview_*.png` | 个人材料配图；**不入库** |

### `report/figures/`（入库）

| 文件 | 说明 |
|---|---|
| `yolo11_triplet_arch.png/.svg/.pdf/.drawio/.html` | YOLOv11s + TripletAttention 结构图（v1），`drawio`/`html` 可继续编辑 |
| `yolo11_triplet_arch_v2.png/.svg/.pdf/.drawio/.html` | 同上（v2，A.9.30）。**已存在本地但未提交**；是否替换 v1 尚未决定 |
| `interview_*.png` | 个人材料配图；**不入库**（本地存在） |

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

## 九、2026-11-13 整理记录（物理归位）

§七 的纪律不变：**被 WORKLOG / 报告 / 脚本逐字引用的路径一律不搬**。本次只处理"无引用、
无审计价值、纯堆在根目录"的散件，全部为可逆移动。同日经用户确认「没用的不留下」后，
对下方清单中的无引用垃圾执行了删除，见本节末尾的**删除记录**。

| 原路径 | 新路径 | 依据 |
|---|---|---|
| `gmodel_dump_dir/` | `_archive/nncase_dumps/gmodel_dump_dir/` | nncase 编译 dump（arg.sc/arg.txt），无引用 |
| `tmp/` | `_archive/nncase_dumps/tmp/` | 同上（`7_TargetDependentAfterQuantPass/`） |
| `_regen.log` | `logs/_regen.log` | 2026-10-29 一次重评日志，四份文档均无引用 |
| `_pdf_yolo.txt` | `deploy/_pdf_yolo.txt` | K230 教程 PDF 的抽取文本（"K230 YOLO 大作战"） |
| `界面改造说明_零基础.md` | `pyqt5_demos/界面改造说明_零基础.md` | 讲的是 `ultralytics_wire_main/main.py` 的 PyQt 界面改造 |
| `95f77b90….jpg`（MD5 文件名） | `pyqt5_demos/界面效果图_老师原版.jpg` | 即上述说明书配图，老师给的效果图；原名为哈希，无语义 |
| `dataset_helmet/_raw/helmet_v10.zip`、`helmet_cls_train.parquet`、`preview_cls.jpg` | `dataset_helmet/_raw/_rejected/` | 两次误下的**摩托车头盔**数据集，已否决 |
| `dataset_helmet/_raw/train_images.zip` | `dataset_helmet/_raw/_aborted/` | SHWD 原图下载中断残片（52 MB / 全量 1.1 GB） |

`k210/` 内部同步整理：`v5/_archive/` 收纳 35 个失败中间模型（b/c/d 系列 nncase 尝试、
v5n 全量模型的 OOM 产物）、21 个编译/训练日志、29 个调试脚本、16 个板端原始输出；
根目录 `ncc` 安装包与 YOLO11 探针件移入 `k210/_archive/`。可复现链路（训练/评估/导出/
上传/运行五个脚本 + 两个最终模型 + 数据集 + `runs/` 曲线）原地保留。

**未动**：`WORKLOG.md`（只追加不删除）、`report/`（就地改 docx，不重建）、`runs/`、
`logs/` 既有 83 项、`dataset_*` 各目录、`exp_upscale/`、`code/`、`yolo11n.pt`（取证对象）。

### 删除记录（2026-11-13，经用户确认「没用的不留下」）

| 删除项 | 体积 | 依据 |
|---|---:|---|
| `k210/_archive/`（ncc 安装包×2、YOLO11 探针件、tiny 探针模型、`_tree.json`、`__pycache__`） | 68 MB | ncc.exe 已解出可用，探针路线已否决（YOLO11 的 SiLU 走不通，结论在 WORKLOG A.9.88） |
| `k210/v5/_archive/`（35 个失败中间模型、21 个日志、29 个调试脚本、16 个板端输出） | 104 MB | 过程结论已沉淀进 WORKLOG A.9.88/89 与《K210部署总结.md》，无复跑价值 |
| `_archive/nncase_dumps/`（`gmodel_dump_dir/`、`tmp/`） | 10 MB | nncase 编译 dump，无引用 |
| `deploy/gmodel_dump_dir/`、`manual/gmodel_dump_dir/`、`manual/tmp/` | 6.3 MB | 同上 |
| `dataset_helmet/_raw/_rejected/`（两个摩托车头盔数据集 + 否决拼图） | 104 MB | 误下数据，已否决两次 |
| `dataset_helmet/_raw/_aborted/train_images.zip` | 51 MB | SHWD 原图中断下载残片（52 MB / 1.1 GB） |
| `exp_upscale/smoke/`（6 组冒烟测试） | 714 MB | **WORKLOG 第 6242 行原文即标注「可清理垃圾（未删，等确认）」**，本次为该确认 |
| `runs/smoke_noleak_base`、`runs/smoke_noleak_triplet` | 118 MB | 同上；`runs/` 内仅删这两项冒烟产物，其余封版证据未动 |
| `logs/_regen.log` | 30 KB | 无引用的重评日志 |
| `weights/` | 0 | 历史遗留空目录 |
| 项目代码内 39 个 `__pycache__/` | ≈1 MB | 自动再生缓存；`envs/` 虚拟环境内的 1,197 个保留（加速 import，非垃圾） |

**合计释放约 1.17 GB**。

**未删**（属证据/可重建素材/未授权范围）：`runs/` 其余全部、`logs/` 其余 82 项、
`exp_upscale/` 其余、`report/` 全部（含可再生预览图，未单独确认不碰）、各 `dataset_*`、
`envs/`、`models/`、`_backup/`、`archive/`、`yolo11n.pt`（取证对象）、`k210/` 可复现链路
（五脚本 + 两最终模型 + 数据集 + `runs/` 曲线）、`deploy/` `manual/` 主链路。
