# -*- coding: utf-8 -*-
"""PC 端对照：用 best.pt 直接跑 320x224（部署尺寸），解码出框。

用途：
  1) 生成给板子看的 jpg（直接拉伸到 320x224，和部署预处理一致）
  2) 给出 PyTorch 的检结果，作为板上输出的对照基准
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

Y5 = r'D:/2/code/baselines/yolov5'
sys.path.insert(0, Y5)
os.chdir(Y5)

import export_k210_trained as E  # noqa: E402

W, H = 320, 224
STRIDES = [8, 16, 32]
ANCHORS = np.array([[[10., 13.], [16., 30.], [33., 23.]],
                    [[30., 61.], [62., 45.], [59., 119.]],
                    [[116., 90.], [156., 198.], [373., 326.]]], dtype=np.float32)
OUT = r'D:/2/k210/v5/demo'
os.makedirs(OUT, exist_ok=True)

PICS = [
    (r'D:/2/dataset_noleak/images/test/firc_xianlan_1010.jpg', [0.336, 0.378, 0.109, 0.073]),
    (r'D:/2/dataset_noleak/images/test/firc_xianlan_1011.jpg', [0.287, 0.335, 0.138, 0.087]),
    (r'D:/2/dataset_noleak/images/test/firc_xianlan_1012.jpg', [0.315, 0.406, 0.133, 0.083]),
]


def load_model(w):
    E.Detect.forward = E.raw_forward
    m = torch.load(w, map_location='cpu', weights_only=False)['model'].float()
    return m.fuse().eval()


def decode(out, conf_th=0.10):
    """out: 3 个 (1,18,ny,nx) -> 框列表 [(x1,y1,x2,y2,score)]（像素）"""
    boxes = []
    for k, (o, st, anc) in enumerate(zip(out, STRIDES, ANCHORS)):
        o = o[0].numpy()                      # (18, ny, nx)
        _, ny, nx = o.shape
        o = o.reshape(3, 6, ny, nx)           # 3 anchors x [xywh,obj,cls]
        gy, gx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        for a in range(3):
            p = o[a]
            s = 1 / (1 + np.exp(-p))          # sigmoid
            conf = s[4] * s[5]
            ys, xs = np.where(conf > conf_th)
            for y, x in zip(ys, xs):
                cx = (s[0, y, x] * 2 - 0.5 + x) * st
                cy = (s[1, y, x] * 2 - 0.5 + y) * st
                bw = (s[2, y, x] * 2) ** 2 * anc[a][0]
                bh = (s[3, y, x] * 2) ** 2 * anc[a][1]
                boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, conf[y, x]))
    return boxes


def nms(boxes, iou_th=0.45):
    if not boxes:
        return []
    b = np.array(boxes)
    x1, y1, x2, y2, sc = b.T
    area = (x2 - x1) * (y2 - y1)
    order = sc.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_th]
    return [boxes[i] for i in keep]


def main():
    model = load_model(r'D:/2/k210/v5/runs/v5n125_break_320_s2/weights/best.pt')
    for i, (path, gt) in enumerate(PICS):
        im = Image.open(path).convert('RGB').resize((W, H), Image.BILINEAR)
        im.save(os.path.join(OUT, f't{i}.jpg'), quality=95)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1)[None]
        with torch.no_grad():
            out = model(x)
        det = nms(decode(out))
        det.sort(key=lambda r: -r[4])
        # 真值框（归一化 -> 像素）
        gx, gy, gw, gh = gt
        g = ((gx - gw / 2) * W, (gy - gh / 2) * H, (gx + gw / 2) * W, (gy + gh / 2) * H)
        print(f'--- t{i}.jpg  GT=({g[0]:.0f},{g[1]:.0f},{g[2]:.0f},{g[3]:.0f})  检出 {len(det)} 个')
        for d in det[:5]:
            print(f'    ({d[0]:.0f},{d[1]:.0f},{d[2]:.0f},{d[3]:.0f})  conf={d[4]:.3f}')


if __name__ == '__main__':
    main()
