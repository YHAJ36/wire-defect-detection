# -*- coding: utf-8 -*-
"""YOLOv11s baseline，896 分辨率（对应 640 组的 wire_v11_640 基线）。

与 train_v11_640.py 的基线路径逐项一致，只改两点：imgsz 640→896、name。
不注入 CBAM —— 896 这轮只跑 baseline，用于验证"分辨率是否仍有增益"。

**批次口径**：先按 batch 8 跑，若 OOM 降 batch 4。两者差别须在记录里注明：
    "896 组是 batch N，与 640 组 batch 8 是不同口径"。
    （分辨率提升会同时抬高显存占用，batch 4 时每步梯度样本数减半，
      与 640/batch8 不是严格可比的对照。）

运行（**绝对路径解释器**，bare python 无 ultralytics/torch）：
    D:/yolo-env/Scripts/python.exe D:/2/code/yolov11/train_v11_896.py
冒烟测试：
    D:/yolo-env/Scripts/python.exe D:/2/code/yolov11/train_v11_896.py \
        --epochs 1 --name wire_v11_896_smoke
"""
import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='wire_v11_896')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=896)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO('D:/1/models/yolo11s.pt')

    model.train(
        data='D:/2/dataset_final_bbox/data.yaml',
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        optimizer='SGD', lr0=0.01, lrf=0.005,
        momentum=0.937, weight_decay=0.0005,
        cos_lr=True, warmup_epochs=3.0,
        box=7.5, cls=0.5, dfl=1.5,
        mosaic=0.8, mixup=0.1, copy_paste=0.1,
        hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
        degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5,
        close_mosaic=5, label_smoothing=0.1, dropout=0.0,
        amp=True, seed=42, workers=0, device=0,
        project='D:/2/runs', name=args.name, exist_ok=True,
    )


if __name__ == '__main__':
    main()
