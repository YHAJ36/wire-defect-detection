# -*- coding: utf-8 -*-
"""YOLOv12s 基线 / +CBAM 训练脚本（导线 3 类，AdamW 版）。

与 train_v12_ablation.py（SGD 版）的唯一差别：
    optimizer='AdamW', lr0=0.001
其余超参、CBAM 注入方式完全一致。

CBAM 注入目标：A2C2f（YOLOv12 自适应注意力块，backbone 第 6/8 层 + neck 第 11/14/17 层）。
注入方式：子类化 DetectionTrainer.get_model（模型重建后再注入），
         避免旧版"model.train() 之前注入被重建丢弃"的 bug。
"""
import argparse
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from cbam import inject_cbam


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cbam', action='store_true', help='是否注入 CBAM')
    p.add_argument('--name', required=True)
    p.add_argument('--epochs', type=int, default=60)
    return p.parse_args()


def make_trainer(use_cbam):
    class CBAMDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            if use_cbam:
                inject_cbam(model, {'A2C2f'}, mode='all')
            return model
    return CBAMDetectionTrainer


def main():
    args = parse_args()
    model = YOLO('D:/1/models/yolo12s.pt')

    TrainerCls = make_trainer(args.cbam)

    model.train(
        data='D:/2/dataset_final_bbox/data.yaml',
        epochs=args.epochs, imgsz=320, batch=8,
        optimizer='AdamW', lr0=0.001, lrf=0.005,
        momentum=0.937, weight_decay=0.0005,
        cos_lr=True, warmup_epochs=3.0,
        box=7.5, cls=0.5, dfl=1.5,
        mosaic=0.8, mixup=0.1, copy_paste=0.1,
        hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
        degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5,
        close_mosaic=5, label_smoothing=0.1, dropout=0.0,
        amp=True, seed=42, workers=0, device=0,
        project='D:/2/runs', name=args.name, exist_ok=True,
        trainer=TrainerCls,
    )


if __name__ == '__main__':
    main()
