# -*- coding: utf-8 -*-
"""P2 尺度单独关停实验 —— 区分「推理期干扰」与「训练期损害」。

问题
    P2v2 判负（−0.68 pt）。但损失可能来自两个完全不同的机制：
      A. 训练期：多一个尺度的 loss 把 backbone/neck/其他分支带坏了；
      B. 推理期：P2 分支本身没训好，输出的低质量框在 NMS 里挤掉了 P3/P4 的正确框。
    这两种的处置完全相反：A 无解（结构性问题），B 说明问题出在框的排序/冗余上。

做法
    加载训练好的 P2v2 权重，不改任何参数，只在 Detect 的 forward 之后把
    **P2 尺度那一段（前 224×224 个 anchor）的类别分数置为负**，于是这些框在
    conf 过滤阶段全部消失 —— 等效于"训练时带了 P2、推理时不用它"。
    然后按正式口径（test / NMS 0.5）评 mAP。

判读
    关 P2 后 ≈ 基线 0.9212 ⇒ 权重本身没被带坏，损失来自 P2 框的推理期干扰（B）；
    关 P2 后仍 ≈ 0.909  ⇒ 训练期就被带坏了（A），P2 的框不是主要矛盾。

用法：
    python code/tools/diag/diag_p2_ablation.py <run> [<run> ...]
"""
import sys
from pathlib import Path

import torch

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d in sys.path:
        pass
    else:
        sys.path.insert(0, _d)

from ultralytics import YOLO  # noqa: E402

DATA = 'D:/2/dataset_final_bbox/data.yaml'
RUNS = Path('D:/2/exp_upscale/runs')
OUT = Path('D:/2/exp_upscale/eval_p2_off')
NMS_IOU = 0.5
IMGSZ = 896


def eval_run(run, disable_p2):
    """必须走 YOLO.val：手写 DetectionValidator 直传 model 会与 YOLO.val 差 1.4 pt
    （同权重同口径实测 0.9239 vs 0.9098），关停实验若用另一条路径，Δ 就不可比。"""
    ck = RUNS / run / 'weights' / 'best.pt'
    model = YOLO(str(ck))
    det = model.model.model[-1]

    if disable_p2:
        seen = []

        def hook(mod, inp, out):
            y, x = out                                    # eval 模式返回 (y, x)
            # 段边界必须按**本次真实的特征图尺寸**算：val 是 rect=True，
            # 每张图 letterbox 到 stride 倍数，实际推理尺寸并不固定是 896×896。
            # 之前按 IMGSZ//stride 硬算，段边界错位，压掉的是别的尺度，
            # 结果 mAP 从 0.91 掉到 0.46 —— 那是压错的假象，不是 P2 的贡献。
            seglen = [int(t.shape[2]) * int(t.shape[3]) for t in x]
            if not seen:
                seen.append(seglen)
            n0 = seglen[0]                                # 段 0 = stride 4 = P2
            y = y.clone()
            # 压到 -20：sigmoid ≈ 2e-9，conf 过滤阶段必然全灭
            y[:, 4:, :n0] = -20.0
            return y, x

        det.register_forward_hook(hook)

    r = model.val(data=DATA, split='test', imgsz=IMGSZ, batch=8, workers=0, device=0,
                  iou=NMS_IOU, conf=0.001, plots=False, verbose=False,
                  project=str(OUT), name=f'{run}_p2off' if disable_p2 else f'{run}_full',
                  exist_ok=True)
    det._forward_hooks.clear()
    if disable_p2 and not hasattr(eval_run, '_shown'):
        eval_run._shown = True
        print('  [P2 关停] 首个 batch 各段锚点数 =', globals().get('_last_seen'))
    b = r.box
    return float(b.map50), float(b.map), float(b.mp), float(b.mr)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    runs = sys.argv[1:] or ['p2v2_v11_896_s42']
    print('%-26s %-22s %-22s' % ('run', '完整 4 尺度', '关掉 P2 后'))
    print('%-26s %-22s %-22s' % ('', 'mAP50 / mAP50-95', 'mAP50 / mAP50-95'))
    for run in runs:
        full = eval_run(run, False)
        off = eval_run(run, True)
        print('%-26s %.4f / %.4f      %.4f / %.4f   Δ%+.2f pt' % (
            run, full[0], full[1], off[0], off[1], (off[0] - full[0]) * 100))


if __name__ == '__main__':
    main()
