"""YOLOv5 基线训练脚本（AdamW 版）。

在 yolov5 目录内运行：
    D:/2/envs/yolov5-env/Scripts/python.exe train_v5_adamw.py

与 train_v5.py（SGD 版）的差别：
  1. --optimizer AdamW
  2. lr0=0.001 —— 走自定义 hyp 文件，**不是** --lr0
     （YOLOv5 的 parse_opt() 没有 --lr0，学习率只能从 hyp yaml 读取；
      故新建 data/hyps/hyp.wire_adamw.yaml = hyp.scratch-low 仅改 lr0: 0.001）
  3. --name 默认 wire_v5_adamw（避免与 SGD 版 wire_v5 混淆）
  其余超参完全一致（lrf / momentum / weight_decay / 增强 / 损失增益均走 hyp 默认）。

继承自 train_v5.py 的改动：
  - --workers 0：Windows spawn 下 workers=8 报 WinError 1455 页面文件耗尽。
  - --seed 42：与 v11/v12 统一随机种子。
  - BASE_ARGV + sys.argv[1:]：支持命令行覆盖（argparse 同名选项后出现者胜出）。
"""

import os
import sys

# 确保本目录（YOLOv5 根目录）在 sys.path 里，使 train.py 内部的 `import utils/...` 可解析。
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 默认参数（不带命令行参数时即按此跑）。
BASE_ARGV = [
    "train_v5_adamw.py",
    "--data", "D:/2/dataset_final_bbox/data.yaml",
    "--hyp", "D:/2/code/baselines/yolov5/data/hyps/hyp.wire_adamw.yaml",
    "--epochs", "60",
    "--img", "320",
    "--batch-size", "8",
    "--weights", "yolov5s.pt",
    "--optimizer", "AdamW",
    "--project", "D:/2/runs",
    "--name", "wire_v5_adamw",
    "--workers", "0",
    "--seed", "42",
]

# 追加命令行参数，交给 YOLOv5 原生 parse_opt() 解析。
# 同名选项后出现者覆盖前者（argparse 语义），因此命令行可覆盖上面的任何默认项。
sys.argv = BASE_ARGV + sys.argv[1:]

from train import parse_opt, main  # noqa: E402  (必须在 sys.argv 就绪后导入)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
