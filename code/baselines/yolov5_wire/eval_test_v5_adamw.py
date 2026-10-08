"""在 test 集上评估 YOLOv5 的 wire_v5_adamw best.pt（AdamW + lr0=0.001 版）。

与 eval_test_v5_v2.py 同口径，仅 --weights / --name 换成 adamw 版。

在 yolov5 目录内运行：
    D:/2/envs/yolov5-env/Scripts/python.exe eval_test_v5_adamw.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

sys.argv = [
    "eval_test_v5_adamw.py",
    "--data", "D:/2/dataset_final_bbox/data.yaml",
    "--weights", "D:/2/runs/wire_v5_adamw/weights/best.pt",
    "--img", "320",
    "--batch-size", "8",
    "--task", "test",
    "--workers", "0",
    "--project", "D:/2/runs",
    "--name", "wire_v5_adamw_test",
    "--exist-ok",
]

from val import parse_opt, main  # noqa: E402

if __name__ == "__main__":
    main(parse_opt())
