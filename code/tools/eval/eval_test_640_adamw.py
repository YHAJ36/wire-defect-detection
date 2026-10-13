# -*- coding: utf-8 -*-
"""在 test 集上评估 640 + AdamW 的 4 组 YOLOv11 模型（baseline + CBAM 三组）的 best.pt。

口径同 eval_test_640.py（split=test / imgsz=640 / batch=8 / workers=0 / device=0），
仅 run 名换成 *_640_*_adamw，便于与 640 SGD 那批直接对比。

Windows spawn 下必须 workers=0 + __main__ 守卫。
CBAM 模型的 best.pt 反序列化需要 `cbam` 可导入，故把 yolov11 目录加入 sys.path。
"""
import os
import sys

for d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if d not in sys.path:
        sys.path.insert(0, d)

from ultralytics import YOLO

NAMES = [
    'wire_v11_640_adamw',
    'wire_v11_640_backbone_adamw',
    'wire_v11_640_neck_adamw',
    'wire_v11_640_both_adamw',
]


def main():
    print('run                              mAP50    mAP50-95   P        R', flush=True)
    for n in NAMES:
        m = YOLO(f'D:/2/runs/{n}/weights/best.pt')
        res = m.val(
            data='D:/2/dataset_final_bbox/data.yaml',
            split='test', imgsz=640, batch=8, workers=0, device=0,
            project='D:/2/runs', name=f'{n}_test', exist_ok=True, verbose=True,
        )
        b = res.box
        print(f'{n:<32} {b.map50:.4f}   {b.map:.4f}   {b.mp:.4f}  {b.mr:.4f}', flush=True)


if __name__ == '__main__':
    main()
