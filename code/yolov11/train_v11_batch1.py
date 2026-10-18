# -*- coding: utf-8 -*-
"""第一批 v11 提升实验（6 组）—— 单脚本参数化，产物落 exp_upscale/runs/。

口径来源：WORKLOG 附录 A（A.0 训练口径 / A.9.5 配置表 / A.9.7 探索口径与排序）。
**本批为探索阶段：只报方向与幅度，不下显著性结论。**

与基线 `train_v11_896.py` 的关系：除各组明示的改动外，**逐项一致**
（SGD lr0=0.01 lrf=0.005 cos_lr=True warmup 3.0 / box 7.5 cls 0.5 dfl 1.5 /
 mosaic 0.8 mixup 0.1 copy_paste 0.1 / close_mosaic 5 / label_smoothing 0.1 /
 amp=True seed=42 workers=0）。

`copy_paste: 0.1` 沿用基线值以保持"只改一项"的可比性，但须知它**是无效的**
（detect 任务无 segments，见 WORKLOG A.1-1），此处仅作口径占位。

用法：
    D:/yolo-env/Scripts/python.exe D:/2/code/yolov11/train_v11_batch1.py --config c1
冒烟：
    ... --config c1 --epochs 1 --name smoke_c1
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

EXP = 'D:/2/exp_upscale/runs'
DATA = 'D:/2/dataset_final_bbox/data.yaml'

# 基线口径（除各组明示改动外，一律照抄）
BASE = dict(
    data=DATA,
    optimizer='SGD', lr0=0.01, lrf=0.005,
    momentum=0.937, weight_decay=0.0005,
    cos_lr=True, warmup_epochs=3.0,
    box=7.5, cls=0.5, dfl=1.5,
    mosaic=0.8, mixup=0.1, copy_paste=0.1,
    hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
    degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5,
    close_mosaic=5, label_smoothing=0.1, dropout=0.0,
    amp=True, seed=42, workers=0, device=0,
    project=EXP,
)

# 6 组。w = 权重；inject_triplet = 是否按 DetectionTrainer 子类注入 Triplet@neck(层9)
CONFIGS = {
    # ⓿ **c0 = 照抄基线原样重跑**：复现性检查，**不是提升实验**。
    #   动机（WORKLOG A.9.14）：c1/c4/c5 的 per-class 出现同一模式 —— break 一律掉
    #   3.5~4.7pt、thunderbolt 一律涨 0.85~1.5pt，而五个改动互不相同（分辨率/容量/
    #   增强/数据/优化器）；连"只换优化器"的 c5（同为 896）也照掉 4.66pt。
    #   ⇒ 怀疑归档基线 runs/wire_v11_896 的 0.9148 是偏好的单次抽样。
    #   本组口径与 code/yolov11/train_v11_896.py **逐项一致**（已用 ast 解析实参核过，
    #   见 A.9.15）；唯一差别是 project 指向 exp_upscale/runs（不污染已封版的 runs/）。
    #   判据：c0 的 break ≈0.87 ⇒ 本批负收益为真；c0 的 break 也 ≈0.83 ⇒ 归档基准不可复现。
    'c0': dict(w='D:/1/models/yolo11s.pt', name='wire_v11_896_repro',
               imgsz=896, batch=8),
    # ① 分辨率余量（不是"救小目标"——1024 下 <8px 占比 0.0%，见 A.9.1）
    'c1': dict(w='D:/1/models/yolo11s.pt', name='wire_v11_1024',
               imgsz=1024, batch=8),
    # ② 容量瓶颈：s→m，batch 4 但 nbs 保持 64（nbs 是 ultralytics 默认值 = 64）
    #    ⚠ 口径接近但不等同：BN micro-batch 4 vs 8 是残余差异，解读时必须注明
    'c2': dict(w='D:/1/models/yolo11m.pt', name='wire_v11m_896_b4',
               imgsz=896, batch=4, nbs=64),
    # ③ 增强：mosaic 上调 + close_mosaic 延后 + mixup 上调
    'c3': dict(w='D:/1/models/yolo11s.pt', name='wire_v11_1024_aug',
               imgsz=1024, batch=8, mosaic=1.0, close_mosaic=10, mixup=0.15),
    # ④ 手动 wear 定向 copy-paste —— 数据集由 code/upscale/make_cp_wear.py 离线生成
    #    **就地替换版**：画布图被换成增强版，故 train 仍 1428 张、break/thunderbolt
    #    实例数一字不变，唯一变化是 wear 788→1246。epochs 仍 60（步数不变）。
    'c4': dict(w='D:/1/models/yolo11s.pt', name='wire_v11_1024_cpm',
               data='D:/2/dataset_cpm_wear/data_cpm.yaml',
               imgsz=1024, batch=8),
    # ⑤ 优化器排他项
    'c5': dict(w='D:/1/models/yolo11s.pt', name='wire_v11_896_adamw',
               imgsz=896, batch=8, optimizer='AdamW', lr0=0.001),
    # ⑥ Triplet@neck —— **排除性实验，非提升实验**；排最后，可砍
    'c6': dict(w='D:/1/models/yolo11s.pt', name='wire_v11_1024_triplet',
               imgsz=1024, batch=8, inject_triplet=True),
}


def make_triplet_trainer():
    """按 CBAM 的教训注入：必须在 get_model() 重建之后，不能在 train() 之前。

    CBAM 曾因注入时机不对被 train() 内部的 yaml 重建静默丢弃。
    注入后由 code/tools/check_triplet_params.py 的闸门把关（预期 +300）。
    """
    import sys
    sys.path.insert(0, 'D:/2/code/yolov11')
    import torch.nn as nn
    from triplet_att import TripletAttention
    from ultralytics.models.yolo.detect import DetectionTrainer

    class TripletTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            seq = model.model
            i = 9                                    # SPPF，与 CBAM 的 neck_sppf 对齐
            gate = TripletAttention()
            w = nn.Sequential(seq[i], gate)
            w.f, w.i = seq[i].f, seq[i].i
            if hasattr(seq[i], 'type'):
                w.type = seq[i].type
            seq[i] = w
            n = sum(p.numel() for p in model.parameters())
            print(f'>>> Triplet@neck 注入 层{i}；全模型参数量 {n:,}（baseline 9,458,752 ⇒ 预期 9,459,052）')
            return model

    return TripletTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, choices=sorted(CONFIGS))
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--name', default=None)
    # 冒烟用覆盖（正式跑批一律不传，保持与基线同口径）
    p.add_argument('--imgsz', type=int, default=None, help='冒烟：覆盖分辨率')
    p.add_argument('--device', default=None, help='冒烟：cpu / 0')
    p.add_argument('--fraction', type=float, default=None, help='冒烟：只用一小部分数据')
    p.add_argument('--project', default=None, help='冒烟：换个输出根，不落正式 runs')
    p.add_argument('--resume', action='store_true',
                   help='从该 run 的 last.pt 续训（被中断后接上，不从头重跑）')
    args = p.parse_args()

    cfg = CONFIGS[args.config]
    kw = dict(BASE)
    kw.update({k: v for k, v in cfg.items() if k not in ('w', 'inject_triplet')})
    if args.name:
        kw['name'] = args.name
    kw['epochs'] = args.epochs
    for a in ('imgsz', 'device', 'fraction', 'project'):
        v = getattr(args, a)
        if v is not None:
            kw[a] = v

    print(f'=== 配置 {args.config} ===')
    for k in ('name', 'imgsz', 'batch', 'optimizer', 'lr0', 'nbs'):
        if k in kw:
            print(f'    {k} = {kw[k]}')
    if kw.get('data') != DATA:
        print(f'    data = {kw.get("data")}   ← 非基线数据集（仅 c4 如此）')

    if args.resume:
        # 续训：从该 run 的 last.pt 接上。ultralytics 的 resume 会从 ckpt 里读回
        # 原始 train_args（含 save_dir），故此处不传 kw，只给 resume。
        last = Path(kw.get('project', EXP)) / kw['name'] / 'weights' / 'last.pt'
        if not last.exists():
            raise SystemExit(f'❌ 无法续训：{last} 不存在')
        print(f'>>> 续训：{last}')
        YOLO(str(last)).train(resume=True)
    elif cfg.get('inject_triplet'):
        model = YOLO(cfg['w'])
        model.train(trainer=make_triplet_trainer(), **kw)
    else:
        model = YOLO(cfg['w'])
        model.train(**kw)


if __name__ == '__main__':
    main()
