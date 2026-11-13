# -*- coding: utf-8 -*-
"""在 test 集上评估 train_k210.py 训出的模型（PyTorch 推理，320 输入）。

注意：best.pt 里 SPPFK / Up2 的类名记录为 __main__.SPPFK，所以这两个类
必须在本文件顶层（即 __main__）定义，否则反序列化失败。
"""
import os
import sys

import torch
import torch.nn as nn

Y5 = r'D:/2/code/baselines/yolov5'
if Y5 not in sys.path:
    sys.path.insert(0, Y5)
os.chdir(Y5)

from models.common import SPPF  # noqa: E402


class SPPFK(SPPF):
    def forward(self, x):
        y = [self.cv1(x)]
        y.extend([y[-1]] * 3)
        return self.cv2(torch.cat(y, 1))


class Up2(nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(
            x, scale_factor=2, mode='nearest', recompute_scale_factor=False)


BASE_ARGV = [
    'val.py',
    '--data', 'D:/2/k210/v5/dataset_break/data.yaml',
    '--weights', 'D:/2/k210/v5/runs/v5n125_break_320/weights/best.pt',
    '--imgsz', '320',
    '--batch-size', '8',
    '--task', 'test',
    '--conf-thres', '0.001',
    '--iou-thres', '0.6',
    '--project', 'D:/2/k210/v5/runs',
    '--name', 'eval_test_320',
    '--exist-ok',
]

sys.argv = BASE_ARGV + sys.argv[1:]

from val import parse_opt, main  # noqa: E402

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
