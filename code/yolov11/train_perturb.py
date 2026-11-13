# -*- coding: utf-8 -*-
"""数据侧扰动实验：检验「瓶颈在标注」这个判断到底成不成立。

背景（2026-10-29 诊断，code/tools/eval/iou_diagnosis.py）
    三种子在同帧同目标上的预测互相 IoU（A）与「共识 vs GT」（B）：
        break    A=0.695  B=0.662   差 +0.033   ← 模型自己就不一致
        thunder  A=0.901  B=0.678   差 +0.223   ← 模型很一致，但离标注远
        wear     A=0.794  B=0.713   差 +0.081
    所以「标注太烂」不能一概而论：
        · break 的瓶颈在**模型方差**（A 只有 0.695），不是标注
        · thunder 的瓶颈才更像标注（A−B = 0.223）
    本脚本用两个单变量去分头检验这两个说法，**都不改网络结构**。

两个模式
    1) --gt-jitter SIGMA
       每轮重新抖动训练集 GT 框（中心相对框宽高抖动 + 半幅度尺度抖动），
       让模型无法精确拟合「这一帧人怎么画的」，只能学「目标大概在这」。
       若有效，应该主要出现在 test（跨段）而不是 val（同段）——同段标注风格一致，
       拟合得准反而是优势。「val 不动 / test 涨」是本实验最强的证据形态。

    2) --drop-file TSV
       按外部清单删掉「三颗种子都不认同」的训练集 GT 框（权重置 0 的粗粒度近似）。
       这是「不可信框清洗」的直接检验。
       若无效，说明噪声是**均匀抖动**而非**局部错标**——洗不掉，这本身是结论。

    两个模式可以叠加，但正式实验建议分开跑（单变量）。

为什么单独写一个入口而不改 train_att_640.py
    后者是已验证的正式管线（跑出了 v3 的全部结果）。往里加分支的风险是：
    一次改动同时影响「历史可比性」和「注入正确性」，出问题无法二分定位。
    本脚本 import 它的 BASE 常量与注入逻辑，保证除扰动外**逐项一致**，
    并自带一份配置指纹打印，便于事后核对。

用法
    冒烟：
      python code/yolov11/train_perturb.py --att baseline --epochs 1 --fraction 0.03 \
             --gt-jitter 0.10 --name smoke_jit
    正式：
      python code/yolov11/train_perturb.py --att baseline --seed 42 --epochs 60 \
             --imgsz 896 --data D:/2/dataset_noleak/data.yaml \
             --gt-jitter 0.10 --name nf_v11_noleak_jit10_s42
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'D:/2/code/yolov11')
import train_att_640 as T          # noqa: E402  BASE / make_trainer / inject
from ultralytics import YOLO       # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402


def jitter_labels(labels, sigma, rng):
    """就地抖动一组标签的 bboxes（xywh 归一化）。返回被改的框数。

    sigma 是**中心相对偏移**（乘框宽/高）；尺度抖动取 sigma 的一半。
    这个参数化在 ceiling_analysis.py 里扫过，sigma=0.10 对应
    「抖动副本与原框」IoU 中位约 0.74，略轻于诊断出的标注偏离 B=0.695。
    """
    n = 0
    for lab in labels:
        b = lab.get('bboxes')
        if b is None or len(b) == 0:
            continue
        b = np.asarray(b, dtype=np.float32)
        w, h = b[:, 2].copy(), b[:, 3].copy()
        b[:, 0] = b[:, 0] + rng.normal(0, sigma, len(b)).astype(np.float32) * w
        b[:, 1] = b[:, 1] + rng.normal(0, sigma, len(b)).astype(np.float32) * h
        s = 1.0 + rng.normal(0, sigma * 0.5, (len(b), 2)).astype(np.float32)
        s = np.clip(s, 0.6, 1.6)
        b[:, 2] = np.clip(w * s[:, 0], 0.002, 1.0)
        b[:, 3] = np.clip(h * s[:, 1], 0.002, 1.0)
        b[:, 0] = np.clip(b[:, 0], 0.0, 1.0)
        b[:, 1] = np.clip(b[:, 1], 0.0, 1.0)
        lab['bboxes'] = b
        n += len(b)
    return n


def drop_labels(labels, drop_map, stats):
    """按 {stem: set(gt_idx)} 删除指定 GT 框。"""
    for lab in labels:
        stem = Path(lab['im_file']).stem
        kill = drop_map.get(stem)
        if not kill:
            continue
        n0 = len(lab['bboxes'])
        keep = np.array([i not in kill for i in range(n0)], bool)
        if keep.all():
            continue
        stats['dropped'] += int((~keep).sum())
        lab['bboxes'] = np.asarray(lab['bboxes'])[keep]
        lab['cls'] = np.asarray(lab['cls'])[keep]
    return stats


def read_drop_file(p):
    """TSV：stem<TAB>gt_index（0 起）。"""
    m = {}
    if not p:
        return m
    for ln in Path(p).read_text(encoding='utf-8').splitlines():
        t = ln.split('\t')
        if len(t) >= 2 and t[1].strip().isdigit():
            m.setdefault(t[0].strip(), set()).add(int(t[1]))
    return m


def make_trainer(args, state):
    """基线管线 + 两个可选的数据侧扰动。"""
    base = T.make_trainer(args.att, args.mode)
    drop_map = read_drop_file(args.drop_file)
    rng0 = np.random.default_rng(args.seed)

    class PerturbTrainer(base):
        def build_dataset(self, img_path, mode='train', batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            if mode == 'train':
                state['ds'] = ds
                state['drop_map'] = drop_map
                state['rng'] = np.random.default_rng(args.seed + 1000)
                state['stats'] = {'dropped': 0}
                if drop_map:
                    drop_labels(ds.labels, drop_map, state['stats'])
                    print(f'>>> 已删除不可信框 {state["stats"]["dropped"]} 个'
                          f'（来自 {len(drop_map)} 张图）')
                # 保存抖动前的原始标签，供每轮重抖
                state['orig'] = [np.asarray(l['bboxes'], np.float32).copy()
                                 if l.get('bboxes') is not None else None
                                 for l in ds.labels]
                if args.gt_jitter > 0:
                    n = 0
                    for lab, o in zip(ds.labels, state['orig']):
                        if o is not None:
                            lab['bboxes'] = o.copy()
                    n = jitter_labels(ds.labels, args.gt_jitter, state['rng'])
                    print(f'>>> GT 抖动已启用 sigma={args.gt_jitter}，'
                          f'首轮抖动 {n} 个框，之后每轮重抖')
            return ds

    return PerturbTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--att', choices=['baseline', 'cbam_bn', 'triplet'], default='baseline')
    p.add_argument('--mode', default='neck_sppf')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--imgsz', type=int, default=896)
    p.add_argument('--data', default='D:/2/dataset_noleak/data.yaml')
    p.add_argument('--name', default=None)
    p.add_argument('--project', default=None)
    p.add_argument('--fraction', type=float, default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--gt-jitter', type=float, default=0.0,
                   help='GT 中心相对抖动 sigma（0=关闭）；每轮重抖')
    p.add_argument('--drop-file', default=None,
                   help='TSV：stem<TAB>gt_index，要删除的训练框')
    p.add_argument('--close-mosaic', type=int, default=None)
    p.add_argument('--mosaic', type=float, default=None)
    p.add_argument('--mixup', type=float, default=None)
    p.add_argument('--copy-paste', type=float, default=None)
    p.add_argument('--scale', type=float, default=None)
    p.add_argument('--batch', type=int, default=None)
    p.add_argument('--weights', default=None,
                   help='预训练权重；默认沿用 train_att_640.WEIGHTS（yolo11s）')
    p.add_argument('--multi-scale', action='store_true',
                   help='开启 ultralytics multi_scale（BASE 默认关闭，本项目从未试过）')
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    kw = dict(T.BASE)
    kw.update(data=args.data, imgsz=args.imgsz, epochs=args.epochs, seed=args.seed)
    if args.name:
        kw['name'] = args.name
    if args.project:
        kw['project'] = args.project
    if args.fraction is not None:
        kw['fraction'] = args.fraction
    if args.device is not None:
        kw['device'] = args.device
    if args.close_mosaic is not None:
        kw['close_mosaic'] = args.close_mosaic
    if args.mosaic is not None:
        kw['mosaic'] = args.mosaic
    if args.mixup is not None:
        kw['mixup'] = args.mixup
    if args.copy_paste is not None:
        kw['copy_paste'] = args.copy_paste
    if args.scale is not None:
        kw['scale'] = args.scale
    if args.batch is not None:
        kw['batch'] = args.batch
    if args.multi_scale:
        kw['multi_scale'] = True

    tag = []
    if args.gt_jitter > 0:
        tag.append(f'jitter={args.gt_jitter}')
    if args.drop_file:
        tag.append(f'drop={Path(args.drop_file).name}')
    if args.multi_scale:
        tag.append('multi_scale')
    print(f'=== att={args.att} seed={args.seed} imgsz={args.imgsz} '
          f'epochs={args.epochs} 扰动={"/".join(tag) or "无"} ===')
    print(f'=== data={args.data}  name={kw.get("name")} ===')

    state = {'ds': None}
    model = YOLO(args.weights or T.WEIGHTS)

    if args.resume:
        last = Path(T.EXP) / (args.name or '') / 'weights/last.pt'
        if last.exists():
            YOLO(str(last)).train(trainer=make_trainer(args, state), resume=True)
            return
        print(f'⚠ 没有 {last}，改为从头训练')

    # 每轮重抖：只在 gt_jitter>0 时注册
    if args.gt_jitter > 0:
        def _rebake(trainer=None):
            ds = state.get('ds')
            if ds is None or not state.get('orig'):
                return
            for lab, o in zip(ds.labels, state['orig']):
                if o is not None:
                    lab['bboxes'] = o.copy()
            jitter_labels(ds.labels, args.gt_jitter, state['rng'])

        model.add_callback('on_train_epoch_start', _rebake)
        print('>>> 已注册 on_train_epoch_start 重抖回调')

    model.train(trainer=make_trainer(args, state), **kw)


if __name__ == '__main__':
    main()
