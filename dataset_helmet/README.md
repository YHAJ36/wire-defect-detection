# dataset_helmet —— 安全帽数据素材库

> 2026-11-13 整理。本目录只放**原始素材**，训练用的裁片在 `D:/2/k210/hat/dataset/`。

## 现在能用的东西

| 路径 | 是什么 |
|---|---|
| `_raw/labels.zip` | **SHWD 的 YOLO 格式标注**，7,571 个 txt。class 1 = 安全帽（9,024 框）、class 0 = 未戴帽人头（111,503 框），与 SHWD 官方数字吻合 |
| `_raw/msrepo/` | ModelScope 仓库元数据（`safetyhelmet_detection_for_train.json` 索引） |
| `fetch_ms.py` | 从 ModelScope 拉取脚本（走阿里云 1.43 MB/s，比 hf-mirror 快 46 倍） |

实际拿到的图像是 ModelScope 上的 **SHWD 子集：600 张原图（500 train / 100 val）**，COCO 格式，
存于 `C:/Users/LR/.cache/modelscope/hub/datasets/modelscope/safetyhelmet_detection_for_train/master/data_files/extracted/`。
已由 `k210/hat/step1_build_dataset.py` 裁成 320×224 大头像：**训练 1,011 / 验证 203**。

## 归档的东西（别再用）

| 路径 | 为什么在这 |
|---|---|
| `_raw/_rejected/helmet_v10.zip` | HuggingFace `harijawahar/Helmet_Detection`，解压后发现是**摩托车头盔**（Helmet/NOHelmet），不是施工安全帽 |
| `_raw/_rejected/helmet_cls_train.parquet` | `pzalavad/HelmetDataset` 分类版，2,193 张 224×224，看了预览图仍是摩托车头盔，且重复图严重 |
| `_raw/_rejected/preview_cls.jpg` | 上面这个 parquet 的抽样拼图（否决依据） |
| `_raw/_aborted/train_images.zip` | `xmu-whx/helmet-mac` 的 SHWD 原图下载残片，52 MB / 全量 1.1 GB，服务器限速 31 KB/s 放弃 |

## 已知短板

SHWD 的「未戴帽」负样本大量来自 SCUT-HEAD 教室监控，裁出来多为**侧脸、后脑勺、低头**，
缺正面光头大头像。目标场景是 1~2 m 正面刷脸，实拍时精确率会打折。
补齐办法：自拍 50~100 段（2~3 人，1~2 m，戴/不戴，多角度）做微调。

板子已归还，安全帽模型**未训练**。拿到新板后从 `k210/hat/step1_build_dataset.py` 重新起链即可。
