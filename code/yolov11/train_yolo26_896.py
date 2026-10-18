# -*- coding: utf-8 -*-
"""YOLO26s baseline，896 分辨率 —— 用于与 YOLOv11s@896 基线做跨代对照。

来源与口径
    逐字照抄 `code/yolov11/train_v11_896.py` 的 `model.train(...)` 超参块，
    **只改三处**：① 权重 yolo11s.pt → yolo26s.pt；② project → exp_upscale/runs
    （新实验独立目录，不污染封版 runs/）；③ 新增 `--seed`（第 4 步要跑 s43/s44）。
    其余每一项（含 `label_smoothing=0.1`）原样保留 —— 见下面"刻意保留的无效项"。

本文件不碰 `train_v11_896.py`、不碰封版 `runs/`、不碰 `dataset_final_bbox`。

刻意保留的无效项（★ 别"顺手修好"，那会破坏可比性）
    `label_smoothing` 在 ultralytics 8.3.180 与 8.4.178 **都已是 deprecated**，
    `get_cfg` 只发一条 deprecation warning 后**静默丢弃**，不进 cfg
    （实测 `hasattr(cfg,'label_smoothing')` = False）。即 v11 那批**从未真正启用**标签平滑。
    WORKLOG.md:875 已记过；WORKLOG.md:1808-1811 亦论证过"两个脚本同为死代码 ⇒ 口径一致"。
    故本文件**照抄这一行**：两代同为"无效"，才是同口径。

环境（★ 必须用新环境解释器，yolo-env 的 ultralytics 8.3.180 不认识 YOLO26）
    D:/yolo26-env/Scripts/python.exe D:/2/code/yolov11/train_yolo26_896.py \
        --seed 42 --name nf_yolo26_896_s42
冒烟测试：
    D:/yolo26-env/Scripts/python.exe D:/2/code/yolov11/train_yolo26_896.py \
        --epochs 1 --name nf_yolo26_896_smoke

崩溃续训（第 4 步的串行链用，`--resume`）
    D:/yolo26-env/Scripts/python.exe D:/2/code/yolov11/train_yolo26_896.py \
        --seed 43 --name nf_yolo26_896_s43 --resume
    口径与 `train_att_640.py` 的 resume 分支一致：**只传 resume、不传任何超参 kw**，
    因为 ultralytics 会从 ckpt 里读回原始 train_args（含 save_dir / epochs / 超参），
    再传 kw 会与 ckpt 里的值打架（见 train_att_640.py:223-234 的实测注释）。
    YOLO26 无注入层，故不像 CBAM/Triplet 那条链需要"续训后核实参数量"来防注入被丢；
    但**续训会重置全局 RNG**（`init_seeds(seed+1+RANK)` 在每次训练启动时都跑，
    trainer.py:127），故"续训过的 run"与"一次跑完的 run"在随机流上并不同源 ——
    这一点在 A.9.20 已记过（同种子重跑差 1.04 pt 的两个可能根因之一）。

已知的跨代不可比点（报告判读时必须一并带上，别当成"只换了模型"）
    1. **端到端头**：yolo26s 的架构 yaml 声明 `end2end: True`（YOLO26 是 NMS-free 端到端检测器），
       但加载后的头 `end2end` 读数是 False、head 类为 `Detect`、`reg_max=1`（YOLO11 是 16）。
       架构 yaml 会在 trainer 重建时生效（CBAM 被 `train()` 重建丢弃的教训），
       故训练时**实际**走不走端到端，须以 trainer 建出的模型为准实测。
    2. 检测头损失口径：YOLO11 有 DFL（reg_max=16），YOLO26 是 reg_max=1；
       `box/cls/dfl` 三个权重是否同义，需核。
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = 'D:/2/models/yolo26s.pt'
PROJECT = 'D:/2/exp_upscale/runs'      # ★ 新实验独立目录，不落封版 runs/


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='nf_yolo26_896_s42')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=896)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--weights', default=WEIGHTS)
    p.add_argument('--project', default=PROJECT)
    p.add_argument('--resume', action='store_true',
                   help='从该 run 的 last.pt 续训（中断后接上，不从头重跑）')
    return p.parse_args()


def main():
    args = parse_args()

    if args.resume:
        # 续训：只传 resume，不传任何超参 kw —— ultralytics 从 ckpt 读回原始 train_args。
        last = Path(args.project) / args.name / 'weights' / 'last.pt'
        if not last.exists():
            raise SystemExit(f'!! 无 last.pt，无法续训：{last}')
        print(f'>>> resume from {last}')
        YOLO(str(last)).train(resume=True)
        return

    model = YOLO(args.weights)

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
        amp=True, seed=args.seed, workers=0, device=0,
        project=args.project, name=args.name, exist_ok=True,
    )


if __name__ == '__main__':
    main()
