# -*- coding: utf-8 -*-
"""WIoU v3 本地实现自检（E1 开跑前的闸门）。

检查：
1. 与官方 iou.py 的 WIoU v3 公式逐值一致（同一 iou_mean、eval 模式）。
2. 训练模式下 iou_mean 会按 momentum=0.01 更新。
3. 输出 shape=(N,1)，梯度有限。
4. monkey-patch 后 v8DetectionLoss 确实使用 WIoUv3BboxLoss。

运行：
    D:/yolo-env/Scripts/python.exe D:/2/code/tools/check_wiou_v3.py
"""
import sys

import torch

sys.path.insert(0, 'D:/2/code/yolov11')

from wiou_v3 import WIoUv3Loss, WIoUv3BboxLoss, patch_ultralytics


def official_wiou_v3(pred, target, iou_mean, alpha=1.7, delta=2.7):
    """官方 iou.py 中 _WIoU + _scaled_loss 的直接转写（eval 模式）。"""
    pred_xy = (pred[..., :2] + pred[..., 2:4]) / 2
    pred_wh = pred[..., 2:4] - pred[..., :2]
    target_xy = (target[..., :2] + target[..., 2:4]) / 2
    target_wh = target[..., 2:4] - target[..., :2]

    min_coord = torch.minimum(pred[..., :4], target[..., :4])
    max_coord = torch.maximum(pred[..., :4], target[..., :4])
    wh_inter = torch.relu(min_coord[..., 2:4] - max_coord[..., :2])
    s_inter = torch.prod(wh_inter, dim=-1)
    s_union = torch.prod(pred_wh, dim=-1) + torch.prod(target_wh, dim=-1) - s_inter
    wh_box = max_coord[..., 2:4] - min_coord[..., :2]
    l2_box = torch.square(wh_box).sum(dim=-1)
    l2_center = torch.square(pred_xy - target_xy).sum(dim=-1)
    iou_loss = 1 - s_inter / s_union

    loss = torch.exp(l2_center / l2_box.detach()) * iou_loss
    beta = iou_loss.detach() / iou_mean
    loss = loss * beta / (delta * torch.pow(alpha, beta - delta))
    return loss.unsqueeze(-1), iou_loss


def main():
    torch.manual_seed(0)
    n = 64
    xy = torch.rand(n, 2, requires_grad=True) * 100
    wh = torch.rand(n, 2) * 20 + 1
    target_xy = torch.rand(n, 2) * 100
    target_wh = torch.rand(n, 2) * 20 + 1
    pred = torch.cat([xy, xy + wh], 1)
    pred.retain_grad()
    target = torch.cat([target_xy, target_xy + target_wh], 1)

    ours = WIoUv3Loss(alpha=1.7, delta=2.7)
    ours.eval()
    ours.iou_mean.fill_(0.8)
    with torch.no_grad():
        y = ours(pred, target)
    ref, _ = official_wiou_v3(pred.detach(), target, torch.tensor(0.8))
    diff = (y - ref).abs().max().item()
    print('1) max |ours - official_ref| = %.3e' % diff)
    assert diff < 1e-6, diff

    # 2) training 模式 iou_mean 更新
    ours.train()
    before = ours.iou_mean.item()
    out = ours(pred, target)
    after = ours.iou_mean.item()
    _, iou_loss = official_wiou_v3(pred.detach(), target, torch.tensor(before))
    expected = before * 0.99 + 0.01 * iou_loss.mean().item()
    print('2) iou_mean: %.6f -> %.6f (expected %.6f)' % (before, after, expected))
    assert abs(after - expected) < 1e-5, (after, expected)

    # 3) shape / 梯度
    out.sum().backward()
    print('3) shape=%s grad_finite=%s' % (tuple(out.shape), bool(torch.isfinite(pred.grad).all())))
    assert out.shape == (n, 1)

    # 4) monkey-patch
    cls = patch_ultralytics(alpha=1.7, delta=2.7, momentum=0.01)
    import ultralytics.utils.loss as L
    assert L.BboxLoss is cls
    from ultralytics import YOLO
    m = YOLO('D:/1/models/yolo11s.pt')
    crit = m.model.init_criterion()
    print('4) criterion.bbox_loss =', type(crit.bbox_loss).__name__,
          '| alpha=%.1f delta=%.1f' % (crit.bbox_loss.wiou.alpha, crit.bbox_loss.wiou.delta))
    assert isinstance(crit.bbox_loss, WIoUv3BboxLoss)

    print('OK: WIoU v3 自检通过')


if __name__ == '__main__':
    main()
