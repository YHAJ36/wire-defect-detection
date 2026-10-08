# 架空输电线缺陷检测

基于 YOLOv5/v11/v12 的输电线缺陷检测，含 CBAM 注意力机制消融、分辨率对比、优化器对比。

## 关键发现
- 分辨率是压倒性主变量：320→640，mAP50 均值 +0.14
- CBAM 无稳定增益（±0.007）：目标尺度小、样本量有限，浅层空间注意力易放大背景纹理
- 双冠军：mAP50 冠军 v5（0.907），mAP50-95 冠军 v11（0.529）

## 数据集
- 3类：break（断股）/ thunderbolt（雷击）/ wear（磨损）
- 1785 张（train 1428 / val 178 / test 179）
- 因版权，数据集与权重未上传

## 目录结构
- code/：训练脚本
  - yolov11/、yolov12/：主实验
  - baselines/yolov5/：对照组（本地保留，未上传）
  - baselines/yolov5_wire/：本项目自写的 YOLOv5 脚本（官方代码需自行 clone）
  - tools/：数据处理、评估、报告脚本
- scripts/：一键运行脚本
- logs/：训练日志
- report/：实验报告 PDF
- archive/：工作记录
- WORKLOG.md：项目状态

## 使用方法
（略，看具体脚本）

## 说明
各框架采用各自推荐配置，通用超参（优化器、初始学习率、epochs、batch、imgsz）一致。
