# 项目工作记录

## AI 规则
1. 每次任务前先读本文件
2. 每完成动作追加到执行记录
3. 只追加不删除

## 项目信息
- 架空输电线缺陷监测系统（老师发的 ultralytics_wire_main.rar，部署 Jetson Orin Nano + 摄像头 + 天线 + 显示器，PyQt5 界面），检测 3 类：break（断股）/ thunderbolt（雷击）/ wear（磨损）。
- 环境：D:\2；yolo-env（训练/推理，ultralytics 8.3.180 / torch 2.11.0+cu128 / RTX4060）+ qt-env（PyQt5 隔离）+ D:\2\envs\yolov5-env（YOLOv5 隔离）。
- 起点权重：D:/1/models/（yolo11s.pt / yolo12s.pt / yolo11n.pt）；D:/1/yolo11s.pt 是下坏的 16.82MB 残缺版，勿用。
- 数据：dataset_final_bbox（task: detect，bbox 版），train 1428 / val 178 / test 179。

## 当前进度
- [x] 数据：SAM 自动标注 → polygon 转 bbox → 8:1:1 划分
- [x] CBAM 注入 bug 定位 + 子类化修复（DetectionTrainer.get_model）
- [x] YOLOv5s：wire_v5（初版）/ wire_v5_v2（SGD + seed42 重跑）
- [x] YOLOv11 SGD 版 4 组消融：wire_v11_v3 / _backbone_v3 / _neck_v3 / _both_v3
- [x] YOLOv12 SGD 版 2 组：wire_v12s_v4 / wire_v12s_cbam_v4
- [x] 7 组 test 集公平对比（统一 SGD + seed42）：v11+CBAM(neck) mAP50 0.794 最优
- [x] PyQt5 界面跑通 + 图片/摄像头检测联调
- [x] 实验报告 PDF 初版（旧 AdamW 数据，需重做）
- [x] 按 SGD 新数据重做实验报告 PDF（report/基于YOLOv11+CBAM的缺陷检测算法_v2.pdf，5 页）
- [x] AdamW(lr0=0.001) 7 组对照训练跑完（runs/wire_*_adamw，各 60ep）——val 均值 0.7270 vs SGD 0.7291，整体打平；test 集评估待做
- [x] AdamW 7 组 test 集评估（已完成，结论：v11_neck 0.7908 居首，AdamW 均值 0.7719 vs SGD 0.7557，见 archive/2026-10-07.md）
- [x] break 断股专项优化 —— **已由 640 分辨率解决**（test mAP50：320 时 0.48~0.56 唯一瓶颈 → 640 时 0.75~0.85）
- [x] **640 全套 6 组训练跑完**（runs/wire_*_640，各 60ep，2026-10-08 02:22 收工）
  - test mAP50：v5 0.9070 | v11+CBAM(backbone) 0.9061 | v11+CBAM(neck) 0.9016 | v11 0.8992 | v11+CBAM(both) 0.8960 | v12+CBAM 0.8874 | v12 0.8682
  - 分辨率是压倒性主变量（320 最好 0.794 → 640 最好 0.907，+0.113）；CBAM 在 640 下依然无稳定增益（v11 上 −0.003~+0.007）
  - 两个口径两个冠军：mAP50 冠军 v5_640，mAP50-95 冠军 v11_640 baseline（0.529 vs v5 0.500）
- [ ] 视频文件检测（本地无素材）
- [ ] 确定最终部署模型 + Jetson Orin Nano TensorRT 量化

## 关键决策
- CBAM 不能原地写在 model.train() 前——会被重建丢弃；正式用子类化 train_v11_ablation.py / train_v12_ablation.py
- 数据用 bbox 版 dataset_final_bbox（task: detect），非 polygon 分割版
- 统一优化器 SGD + seed 42 做横向对比（v5 原 SGD/seed0，v11/v12 原 AdamW）；其余超参各框架用推荐默认，论文注明「各框架推荐配置，通用超参一致」
- 以 test 集结论为准（val 仅 178 张，选模型不稳：val 上 v5 最高、test 上 v11 最高）
- CBAM 无稳定增益（v11 测试 neck +0.015 但 backbone/both 为负；v12 上 −0.018）

## 注意事项
- workers=0（Windows spawn 下 workers=8 报 WinError 1455）
- 先 import torch 再 import PyQt5（DLL 加载顺序冲突）
- label_smoothing 在 ultralytics 8.3.180 已 deprecated；YOLOv5 用 --batch-size 非 --batch
- GitHub 直连不通：settings.json weights_dir 已指向 D:\1\models，避免 ultralytics 联网自动下载
- 勿改写正在运行的 bash 脚本（bash 按字节偏移边读边执行，会错乱/重复执行）
- Windows spawn 下，640 分辨率 + workers=4 会因内存/页面文件不足触发子进程死锁；640 训练用 workers=0（详见 archive/2026-10-07.md 难点记录）
- code/baselines/yolov5/ 为嵌套 git 仓库（含 .git），未上传 GitHub，本地保留。v5 对照组脚本（train_v5*.py、eval_test_v5*.py）在本地，答辩时用本地展示

## 历史归档
详细逐日记录见 archive/（按日期分文件）；本文件只保留长期有效信息。

## 答辩素材（3个亮点）

### 1. CBAM 注入 bug（最亮）
- 现象：四组消融实验，+CBAM 版本和 baseline 参数量完全相同（9,459,018）
- 排查：读 ultralytics model.py 源码，发现 train() 内部按 yaml 重建模型
- 根因：注入写在 train() 之前，被重建丢弃
- 修法：子类化 DetectionTrainer.get_model()，模型重建后立即注入
- 验证：baseline 100层 vs +CBAM 140层，参数量 +84,458
- 启示：实验异常必须查到底层源码，不能只看表面指标

### 2. 优化器混淆变量（最严谨）
- 现象：初版实验 YOLOv5 全面领先
- 排查：发现 v5 用 SGD，v11 用 AdamW，优化器不一致
- 修正后：v5 SGD=0.721, v11 SGD=0.779（v11反超）
- 对比：v11 AdamW+lr0=0.01 只有 0.635，换 SGD+lr0=0.01 涨到 0.779（+0.144）
- 启示：多框架对比，通用超参必须对齐
- ✅ 口径已核（2026-10-08）：0.721/0.779 = **SGD test 口径**（v5_v2 0.7210 / v11 0.7787），原数字正确、无需修改；0.635（v11 AdamW+lr0=0.01）同为 test 口径，+0.144 成立。
  （此前「疑为 val」的提醒作废——那是误与 AdamW(lr0=0.001) 表 0.7550/0.7817 对比所致，SGD 表与 AdamW 表本就不同。）

### 3. Windows spawn 死锁（最工程）
- 现象：640训练跑到 epoch 55，日志停住，主进程存活但 CPU 0%
- 排查：worker 子进程崩在 import torch，主进程收不到通知
- 根因：Windows spawn 每个子进程重新加载 CUDA DLL，内存不够 → worker 崩 → 主进程死等
- 修法：workers=0（不 spawn）
- 启示：跨平台部署要考虑进程模型差异（Windows spawn vs Linux fork）

每个亮点答辩时按「现象-排查-根因-修法-启示」五步讲。
