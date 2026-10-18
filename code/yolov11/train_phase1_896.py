# -*- coding: utf-8 -*-
"""Phase 1 三组探索实验的统一训练入口（E1/E2/E3，896，seed42）。

口径：除命令行明示的改动外，与已封版 896 基线 `runs/wire_v11_896/args.yaml`
逐项一致。基线实际 args.yaml 中**没有** `label_smoothing`（本机 ultralytics
8.3.180 不支持该参数，旧脚本虽传了但未生效），因此这里也不再传。

三组：
  E1  --loss wiou_v3                  CIoU -> WIoU v3（官方默认 1.7/2.7）
  E2  --data .../data_wearos.yaml      wear 图像 2x 过采样（1593 行）
  E3  --model code/yolov11/yolo11s-p2.yaml --batch 4 --pretrained ...
                                      增加 P2 小目标检测头

冒烟：
    ... --epochs 1 --fraction 0.02 --imgsz 320 --batch 2 --name smoke
正式：
    ... --loss wiou_v3 --name wiou_v11_896_s42
    ... --data D:/2/exp_upscale/data/data_wearos.yaml --name wearos2_v11_896_s42
    ... --model D:/2/code/yolov11/yolo11s-p2.yaml --pretrained D:/1/models/yolo11s.pt \
        --batch 4 --name p2_v11_896_s42
续训：
    ... --name <run> --loss wiou_v3 --resume
"""
import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, 'D:/2/code/yolov11')

EXP = 'D:/2/exp_upscale/runs'
DATA = 'D:/2/dataset_final_bbox/data.yaml'

BASE = dict(
    data=DATA,
    optimizer='SGD', lr0=0.01, lrf=0.005,
    momentum=0.937, weight_decay=0.0005,
    cos_lr=True, warmup_epochs=3.0,
    box=7.5, cls=0.5, dfl=1.5,
    mosaic=0.8, mixup=0.1, copy_paste=0.1,
    hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
    degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5,
    close_mosaic=5, dropout=0.0,
    amp=True, seed=42, workers=0, device=0,
    project=EXP,
)


def make_p2_trainer(pretrained):
    """P2 是自定义 YAML：先按 data 的 nc 建模型，再向 backbone 迁移预训练权重。

    不能直接用 `YOLO(yaml).load(pt)`，那会把模型替换成 3 输出尺度的 .pt 结构。
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    class P2Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            from ultralytics.nn.tasks import attempt_load_one_weight
            w, _ = attempt_load_one_weight(pretrained)
            model.load(w)
            return model

    return P2Trainer


def add_wiou_logger(model):
    """每个 epoch 结束记录 WIoU 的 iou_mean，便于判断动态聚焦是否正常。"""

    def on_train_epoch_end(trainer):
        try:
            # EMA 的 __dict__ 会被 copy_attr 带上 criterion；保存前删掉，
            # 免得到处加载 checkpoint 都要 sys.path 指向 wiou_v3。
            ema = getattr(trainer, 'ema', None)
            if ema is not None and hasattr(ema.ema, 'criterion'):
                delattr(ema.ema, 'criterion')

            crit = getattr(trainer.model, 'criterion', None)
            w = getattr(getattr(crit, 'bbox_loss', None), 'wiou', None)
            if w is None:
                return
            out = Path(trainer.save_dir) / 'wiou_iou_mean.csv'
            line = '%d,%.6f\n' % (trainer.epoch + 1, float(w.iou_mean))
            with open(out, 'a', encoding='utf-8') as f:
                f.write(line)
        except Exception:
            pass

    model.add_callback('on_train_epoch_end', on_train_epoch_end)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--model', default='D:/1/models/yolo11s.pt',
                   help='.pt 或自定义 .yaml（E3 用 yolo11s-p2.yaml）')
    p.add_argument('--pretrained', default=None,
                   help='自定义 YAML 时迁移权重的 .pt（E3 用 yolo11s.pt）')
    p.add_argument('--data', default=DATA)
    p.add_argument('--loss', choices=['ciou', 'wiou_v3'], default='ciou')
    p.add_argument('--alpha', type=float, default=1.7)
    p.add_argument('--delta', type=float, default=2.7)
    p.add_argument('--momentum', type=float, default=1e-2)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=896)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='0')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--project', default=EXP)
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    if args.loss == 'wiou_v3':
        from wiou_v3 import patch_ultralytics
        cls = patch_ultralytics(alpha=args.alpha, delta=args.delta, momentum=args.momentum)
        print('>>> WIoU v3 已挂载：alpha=%.2f delta=%.2f momentum=%.4f (%s)' % (
            args.alpha, args.delta, args.momentum, cls.__name__))

    kw = dict(BASE)
    kw.update({
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'seed': args.seed,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'exist_ok': True,
    })
    if args.fraction < 1.0:
        kw['fraction'] = args.fraction

    print('=== %s ===' % args.name)
    for k in ('model', 'data', 'loss', 'epochs', 'batch', 'imgsz', 'seed', 'device'):
        print('    %-10s = %s' % (k, getattr(args, k)))

    # 固定记录本组实验的差异项（args.yaml 不会记录 alpha/delta 这类自定义超参）
    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        'name': args.name, 'model': args.model, 'pretrained': args.pretrained,
        'data': args.data, 'loss': args.loss, 'alpha': args.alpha,
        'delta': args.delta, 'momentum': args.momentum,
        'epochs': args.epochs, 'batch': args.batch, 'imgsz': args.imgsz,
        'seed': args.seed, 'device': args.device, 'workers': args.workers,
        'fraction': args.fraction, 'project': args.project,
        'note': 'baseline args.yaml has no label_smoothing; this run does not pass it either',
    }
    (run_dir / 'phase1_meta.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    if args.loss == 'wiou_v3' and not args.resume:
        (run_dir / 'wiou_iou_mean.csv').unlink(missing_ok=True)

    if args.resume:
        last = Path(args.project) / args.name / 'weights' / 'last.pt'
        if not last.exists():
            raise SystemExit('❌ 无法续训：%s 不存在' % last)
        print('>>> 续训：%s' % last)
        model = YOLO(str(last))
        model.train(resume=True)
        return

    is_yaml = str(args.model).lower().endswith(('.yaml', '.yml'))
    if is_yaml:
        if not args.pretrained:
            raise SystemExit('❌ 自定义 YAML 需要 --pretrained 迁移权重')
        model = YOLO(args.model)
        model.train(trainer=make_p2_trainer(args.pretrained), **kw)
    else:
        model = YOLO(args.model)
        if args.loss == 'wiou_v3':
            add_wiou_logger(model)
        model.train(**kw)


if __name__ == '__main__':
    main()
