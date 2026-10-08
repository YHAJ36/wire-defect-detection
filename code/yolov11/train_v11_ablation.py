import argparse
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from cbam import CBAM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['baseline', 'backbone', 'neck_sppf', 'both'], required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--epochs', type=int, default=60)
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
        cbam = CBAM(ch)
        w = nn.Sequential(seq[i], cbam)
        w.f = seq[i].f
        w.i = seq[i].i
        if hasattr(seq[i], 'type'):
            w.type = seq[i].type
        seq[i] = w
    print(f'>>> CBAM injected: mode={mode} layers={idx}')
    return idx


def make_trainer(mode):
    """生成一个按 mode 注入 CBAM 的 DetectionTrainer 子类"""
    class CBAMDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            if mode != 'baseline':
                inject_cbam_custom(model, mode)
            return model
    return CBAMDetectionTrainer


def main():
    args = parse_args()
    model = YOLO('D:/1/models/yolo11s.pt')

    TrainerCls = make_trainer(args.mode)

    model.train(
        data='D:/2/dataset_final_bbox/data.yaml',
        epochs=args.epochs, imgsz=320, batch=8,
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
        trainer=TrainerCls,
    )


if __name__ == '__main__':
    main()