"""YOLOv5 基线训练脚本（检测）。

在 yolov5 目录内运行：
    D:/2/envs/yolov5-env/Scripts/python.exe train_v5.py

配置等价于（YOLOv5 原生 train.py 参数）：
    python train.py --data D:/2/dataset_final_bbox/data.yaml --epochs 60 --img 320
                    --batch-size 8 --weights yolov5s.pt --project D:/2/runs --name wire_v5
                    --seed 42

改动 vs 用户命令：
  1. 加 --workers 0。原因（见记录 2026-09-30）：Windows spawn 下 workers=8 会开 8 个
     DataLoader 子进程，每个重复 import torch/CUDA DLL 撑爆页面文件，报 WinError 1455。
  2. 加 --seed 42，与 v11/v12 统一随机种子（2026-10-06）。
  3. 支持命令行覆盖：下面 BASE_ARGV 作为默认值，再追加 sys.argv[1:]。
     argparse 对同名选项取最后出现者，故 `train_v5.py --name wire_v5_v2` 可覆盖默认 --name。
"""

import os
import sys

# 确保本目录（YOLOv5 根目录）在 sys.path 里，使 train.py 内部的 `import utils/...` 可解析。
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 默认参数（不带命令行参数时即按此跑，等价于历史上的 wire_v5 配置 + seed 42）。
BASE_ARGV = [
    "train_v5.py",
    "--data", "D:/2/dataset_final_bbox/data.yaml",
    "--epochs", "60",
    "--img", "640",
    "--batch-size", "8",
    "--weights", "yolov5s.pt",
    "--project", "D:/2/runs",
    "--name", "wire_v5_640",
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
