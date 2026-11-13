# -*- coding: utf-8 -*-
"""探针：训练期目标分配器把正样本分给了哪个尺度。

用途 —— 回答 A.9.32 留下的那个推断（"P2 抢走正样本但自己训不动"）到底是真是假。
做法 —— 在 TaskAlignedAssigner 上挂 forward hook，直接读它每次吐出的 fg_mask，
        按预测特征的段长切分，统计每个尺度分到多少正样本、以及该尺度的分数水平。

不改任何权重行为（hook 只读），不写 exp_upscale/runs，产物落 report/_p2_diag/。

用法：
    从头（看早期分配）:
        probe_assign.py --mode scratch --arch p2v2 --name probe_p2v2_scratch
    从已训权重续跑少量 iter（看收敛后的分配）:
        probe_assign.py --mode resume --ckpt exp_upscale/runs/p2v2_v11_896_s42 \
                        --name probe_p2v2_resume
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, 'D:/2/code/yolov11')

import ultralytics.utils.loss as UL
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

import p2_transfer

DATA = 'D:/2/dataset_final_bbox/data.yaml'
PRETRAIN = 'D:/1/models/yolo11s.pt'
YAML_P2V2 = 'D:/2/code/yolov11/yolo11s-p2v2.yaml'
OUT = Path('D:/2/report/_p2_diag')
PROJ = OUT / 'probe_runs'

BASE = dict(
    data=DATA, optimizer='SGD', lr0=0.01, lrf=0.005,
    momentum=0.937, weight_decay=0.0005, cos_lr=True, warmup_epochs=3.0,
    box=7.5, cls=0.5, dfl=1.5,
    mosaic=0.8, mixup=0.1, copy_paste=0.1,
    hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
    degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5,
    close_mosaic=5, dropout=0.0,
    amp=True, seed=42, workers=0, device=0, project=str(PROJ),
    imgsz=896, val=False, plots=False, exist_ok=True, verbose=False,
)

REC = []          # 每次 loss 调用一行：[(scale_idx, n_pos, n_anchor, score_med, score_p99), ...]
STATE = {'seglen': None, 'stride': None}

# hook 必须是**模块级函数**：写成 __call__ 里的闭包会导致 trainer.save_model()
# 时 pickle 失败（Can't get local object 'ProbeLoss.__call__.<locals>.hook'）。
def _assign_hook(mod, inp, out):
    """assigner 出口只读：inp[0]=sigmoid 后分数 (b,hw,nc)，out[3]=fg_mask (b,hw)。"""
    ps, fg = inp[0], out[3]
    cum, rows = 0, []
    for i, L in enumerate(STATE['seglen']):
        seg_fg = fg[:, cum:cum + L]
        seg_sc = ps[:, cum:cum + L, :].amax(-1).float().ravel()
        rows.append((i, int(seg_fg.sum()), L,
                     float(seg_sc.median()),
                     float(torch.quantile(seg_sc, 0.99))))
        cum += L
    REC.append(rows)


class ProbeLoss(UL.v8DetectionLoss):
    """原封不动地跑 v8DetectionLoss，只在 assigner 出口读一次 fg_mask。"""

    def __call__(self, preds, batch):
        feats = preds[1] if isinstance(preds, tuple) else preds
        STATE['seglen'] = [int(f.shape[2]) * int(f.shape[3]) for f in feats]
        STATE['stride'] = [int(s) for s in self.stride]

        if not getattr(self, '_hooked', False):
            self.assigner.register_forward_hook(_assign_hook)
            self._hooked = True

        return super().__call__(preds, batch)


def patch_criterion():
    DetectionModel.init_criterion = lambda self: ProbeLoss(self)


def make_p2_trainer(pretrained, ckpt=None):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class P2Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            from ultralytics.nn.tasks import attempt_load_one_weight
            w, _ = attempt_load_one_weight(pretrained)
            model.load(w)
            p2_transfer.transfer(model, w)
            if ckpt:                                  # 续跑模式：再叠上已训权重
                c, _ = attempt_load_one_weight(ckpt)
                model.load(c)
                p2_transfer.transfer(model, c)
            return model

    return P2Trainer


def run(args):
    REC.clear()
    patch_criterion()

    # ---- resume：先按结构建模型，再灌已训权重，用极小 lr 只跑少量 iter
    if args.mode == 'resume':
        ckpt = str(Path(args.ckpt) / 'weights' / 'best.pt')
        kw = dict(BASE)
        kw.update(dict(epochs=1, batch=4, lr0=1e-4, warmup_epochs=0.0,
                       fraction=args.fraction, name=args.name))
        if args.arch == 'p2v2':
            model = YOLO(YAML_P2V2)
            model.train(trainer=make_p2_trainer(PRETRAIN, ckpt=ckpt), **kw)
        else:
            model = YOLO(ckpt)                        # 基线直接从 .pt 建，结构一致
            model.train(**kw)
    else:
        kw = dict(BASE)
        kw.update(dict(epochs=args.epochs, batch=4, name=args.name))
        if args.arch == 'p2v2':
            model = YOLO(YAML_P2V2)
            model.train(trainer=make_p2_trainer(PRETRAIN), **kw)
        else:
            model = YOLO(PRETRAIN)
            model.train(**kw)

    # ---- 落盘
    tag = args.name
    path = OUT / f'assign_{tag}.csv'
    with path.open('w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['iter', 'scale', 'stride', 'n_anchor', 'n_pos',
                     'score_med', 'score_p99'])
        for it, rows in enumerate(REC):
            for (si, npos, nanch, smed, s99) in rows:
                st = STATE['stride'][si]
                wr.writerow([it, si, st, nanch, npos, round(smed, 6), round(s99, 6)])

    # ---- 汇总打印
    n_iter = len(REC)
    nscale = len(REC[0]) if n_iter else 0
    print(f'\n===== {tag}  ({args.arch}, {args.mode})  iters={n_iter}')
    print(f'{"scale":>5} {"stride":>6} {"anchor":>7} {"正样本/iter":>11} '
          f'{"占比":>8} {"score中位":>10} {"p99":>7}')
    for si in range(nscale):
        tot_pos = sum(r[si][1] for r in REC)
        nanch = REC[0][si][2]
        st = STATE['stride'][si]
        smed = sum(r[si][3] for r in REC) / n_iter
        s99 = sum(r[si][4] for r in REC) / n_iter
        print(f'{si:>5} {st:>6} {nanch:>7} {tot_pos / n_iter:>11.2f} '
              f'{100.0 * tot_pos / (n_iter * nanch):>7.3f}% {smed:>10.4f} {s99:>7.4f}')

    # 每个真值实际分到几个正样本（fg_mask 总数 / GT 数不可直接从 hook 得，
    # 这里给正样本总数，GT 数另行统计）
    print(f'{nscale} 个尺度合计正样本/iter = '
          f'{sum(sum(r[i][1] for r in REC) for i in range(nscale)) / n_iter:.1f}')
    print(f'CSV -> {path}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['scratch', 'resume'], default='scratch')
    p.add_argument('--arch', choices=['p2v2', 'base'], default='p2v2')
    p.add_argument('--ckpt', default=None, help='resume 模式下的 run 目录')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--fraction', type=float, default=0.1,
                   help='resume 模式跑多少数据（0.1 ≈ 36 iter @batch4）')
    p.add_argument('--name', required=True)
    a = p.parse_args()
    if a.mode == 'resume' and not a.ckpt:
        raise SystemExit('resume 需要 --ckpt')
    return a


if __name__ == '__main__':
    run(parse_args())
