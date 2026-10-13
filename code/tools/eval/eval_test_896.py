# -*- coding: utf-8 -*-
"""在 test 集上评估 896 组的 YOLOv11 基线（wire_v11_896）best.pt。

口径同 eval_test_640.py / eval_test_640_adamw.py（split=test / batch=8 /
workers=0 / device=0），仅 imgsz 换 896、run 名换 896 组。

**896 组口径声明**（必须与 640 组区分）：
    896 组只有 v11 baseline + v5 两组，仅作同构子集对比（v11 vs v11、v5 vs v5），
    不与 640 的 7 组均值混算。

无需 `sys.path` 前置 —— 896 组是纯基线，权重里不含 `cbam.CBAM` 的 pickle 引用。
Windows spawn 下必须 workers=0 + __main__ 守卫。
"""
from ultralytics import YOLO

NAMES = [
    'wire_v11_896',
]


def main():
    print('run                              mAP50    mAP50-95   P        R', flush=True)
    for n in NAMES:
        m = YOLO(f'D:/2/runs/{n}/weights/best.pt')
        res = m.val(
            data='D:/2/dataset_final_bbox/data.yaml',
            split='test', imgsz=896, batch=8, workers=0, device=0,
            project='D:/2/runs', name=f'{n}_test', exist_ok=True, verbose=True,
        )
        b = res.box
        print(f'{n:<32} {b.map50:.4f}   {b.map:.4f}   {b.mp:.4f}  {b.mr:.4f}', flush=True)


if __name__ == '__main__':
    main()
