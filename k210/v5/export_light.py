# -*- coding: utf-8 -*-
"""瘦身版 v5：宽度减半（0.25 -> 0.125），验证 K210 能否吃得下

背景：标准 yolov5n 编译出的 kmodel 1.94MB，kpu.load 无论从文件还是 flash 地址
      都报 Out of memory —— K210 的 KPU RAM 只有 2MB，装不下。
      这里只验证"瘦身版能不能加载"，权重是随机的，不是为了精度。
"""
import sys, os, argparse, yaml
import torch

Y5 = r'D:/2/code/baselines/yolov5'
sys.path.insert(0, Y5)
os.chdir(Y5)

from models.yolo import Model, Detect        # noqa: E402
from export_v5_raw import swap_silu, swap_upsample, patch_maxpool, raw_forward  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=float, default=0.125)
    ap.add_argument('--depth', type=float, default=0.33)
    ap.add_argument('--nc', type=int, default=3)
    ap.add_argument('--out', default=r'D:/2/k210/v5/v5_light.onnx')
    a = ap.parse_args()

    cfg = yaml.safe_load(open('models/yolov5n.yaml'))
    cfg['width_multiple'] = a.width
    cfg['depth_multiple'] = a.depth
    cfg['nc'] = a.nc
    model = Model(cfg).float().eval()

    nparam = sum(p.numel() for p in model.parameters())
    print(f'[cfg ] width={a.width} depth={a.depth} nc={a.nc}  params={nparam/1e6:.3f}M')

    print(f'[act ] SiLU->ReLU {swap_silu(model, "relu")} 处')
    print(f'[up  ] Upsample    {swap_upsample(model)} 处')
    print(f'[pool] SPPF 去池化 {patch_maxpool(model)} 处')
    Detect.forward = raw_forward

    dummy = torch.zeros(1, 3, 224, 320)
    with torch.no_grad():
        outs = model(dummy)
    for i, o in enumerate(outs):
        print(f'[head] out{i} {tuple(o.shape)}')

    torch.onnx.export(model, (dummy,), a.out, opset_version=12,
                      input_names=['images'], output_names=[f'p{i}' for i in range(len(outs))],
                      do_constant_folding=True, dynamo=False)
    print(f'[onnx] {a.out} {os.path.getsize(a.out)/1e6:.2f} MB')


if __name__ == '__main__':
    main()
