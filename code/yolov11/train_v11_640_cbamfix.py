# -*- coding: utf-8 -*-
"""YOLOv11s + 修复版 CBAM（中心化 SA 输入 + SA conv 带 bias），640 分辨率。

与 train_v11_640.py 的唯一区别：注入的是 cbam_fixed.CBAM（见 cbam_fixed.py 文件头）。
其余超参与 train_v11_640.py 逐项一致，保证与 wire_v11_640_both 可直接对照。

运行（**必须用绝对路径解释器**，bare `python` 是系统 Python，没有 ultralytics/torch）：
    D:/yolo-env/Scripts/python.exe D:/2/code/yolov11/train_v11_640_cbamfix.py
默认即 --mode both --name wire_v11_640_cbamfix，无需传参。

参数量验收目标：9,513,416
    baseline 9,428,953 + both 注入 84,458 + 5 个 SA conv 的 bias 5 = 9,513,416
    （对比：原版 wire_v11_640_both 是 9,513,411，差 5 = 5 个 bias）
"""
import argparse
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from cbam_fixed import CBAM


def parse_args():
    p = argparse.ArgumentParser()
    # 默认 both / wire_v11_640_cbamfix，便于直接跑；仍可显式覆盖
    p.add_argument('--mode', choices=['baseline', 'backbone', 'neck_sppf', 'both'], default='both')
    p.add_argument('--name', default='wire_v11_640_cbamfix')
    p.add_argument('--epochs', type=int, default=60)
    # 与 train_v11_640.py 保持一致：中间轮次落盘，便于事后测第 20/40 轮的 z 均值。
    p.add_argument('--save-period', type=int, default=20)
    return p.parse_args()


def get_channels(m):
    for name in ('cv2', 'cv3', 'cv4'):
        conv = getattr(m, name, None)
        if conv is None:
            continue
        if isinstance(conv, nn.Conv2d):
            return conv.out_channels
        inner = getattr(conv, 'conv', None)  # ultralytics Conv 包装了 nn.Conv2d
        if isinstance(inner, nn.Conv2d):
            return inner.out_channels
    for sub in reversed(list(m.modules())):
        if isinstance(sub, nn.Conv2d):
            return sub.out_channels
    raise ValueError('no conv')


def inject_cbam_custom(model, mode):
    seq = model.model
    backbone = [2, 4, 6, 8]
    sppf = 9
    if mode == 'baseline':
        return []
    if mode == 'backbone':
        idx = backbone
    elif mode == 'neck_sppf':
        idx = [sppf]
    else:
        idx = backbone + [sppf]

    for i in idx:
        ch = get_channels(seq[i])
        cbam = CBAM(ch)  # ← 唯一实质差异：cbam_fixed.CBAM
        w = nn.Sequential(seq[i], cbam)
        w.f = seq[i].f
        w.i = seq[i].i
        if hasattr(seq[i], 'type'):
            w.type = seq[i].type
        seq[i] = w
    print(f'>>> CBAM(fixed) injected: mode={mode} layers={idx}')
    return idx


def make_trainer(mode):
    """生成一个按 mode 注入修复版 CBAM 的 DetectionTrainer 子类"""
    class CBAMFixDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            if mode != 'baseline':
                inject_cbam_custom(model, mode)
            return model
    return CBAMFixDetectionTrainer


def main():
    args = parse_args()
    model = YOLO('D:/1/models/yolo11s.pt')

    TrainerCls = make_trainer(args.mode)

    model.train(
        data='D:/2/dataset_final_bbox/data.yaml',
        epochs=args.epochs, imgsz=640, batch=8,
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
        save_period=args.save_period,
        trainer=TrainerCls,
    )


if __name__ == '__main__':
    main()
