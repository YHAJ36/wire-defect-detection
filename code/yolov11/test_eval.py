"""在 test 集（179 张）上评估 YOLOv11 4 组消融的 best.pt，输出 mAP 对比。

注意：Windows spawn 下必须 workers=0 + if __name__ == '__main__' 守卫，
否则 DataLoader 子进程会重导本模块触发 RuntimeError。
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # 让 cbam 可被反序列化时 import

from ultralytics import YOLO

NAMES = [
    'wire_v11_v2',
    'wire_v11_backbone_v2',
    'wire_v11_neck_v2',
    'wire_v11_both_v2',
]


def main():
    print('run                              mAP50   mAP50-95   P      R', flush=True)
    for n in NAMES:
        m = YOLO(f'D:/2/runs/{n}/weights/best.pt')
        res = m.val(
            data='D:/2/dataset_final_bbox/data.yaml',
            split='test', imgsz=320, batch=8, workers=0, device=0,
            project='D:/2/runs', name=f'{n}_test', exist_ok=True, verbose=True,
        )
        b = res.box
        print(f'{n:<32} {b.map50:.4f}   {b.map:.4f}   {b.mp:.4f}  {b.mr:.4f}', flush=True)


if __name__ == '__main__':
    main()
