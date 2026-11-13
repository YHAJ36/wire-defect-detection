# -*- coding: utf-8 -*-
"""给板子准备一份「不用碰摄像头也能验模型」的输入 + 参考输出。

为什么要这么做：
  板上第一件事必须是「模型本身跑得对」，而不是「摄像头到模型这条链路跑得通」。
  把两者混在一起调，出问题时无法定位是模型、预处理、ai2d、还是显示。

  所以这里在 PC 上先把一张测试图**按板上的同一套预处理**做成 640×640×3 的
  uint8 RGB 裸数据（.bin），连同一份 ONNX 的参考检出（.json）一起推到板上。
  板上读 bin → 喂模型 → 解码 → 打印，数字对得上就说明模型和预处理都没问题。

用法
    D:/yolo-env/Scripts/python.exe D:/2/deploy/make_board_testcase.py --img 1175
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

NAMES = ['break', 'thunder', 'wear']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default='1175', help='test 集里文件名含这个串的那张')
    ap.add_argument('--imgs', default='D:/2/dataset_noleak/images/test')
    ap.add_argument('--onnx', default='D:/2/deploy/yolo11s_640_single.onnx')
    ap.add_argument('--outdir', default='D:/2/deploy/board_case')
    ap.add_argument('--conf', type=float, default=0.10)
    args = ap.parse_args()

    files = [p for p in Path(args.imgs).iterdir() if args.img in p.name]
    if not files:
        raise SystemExit(f'找不到含 "{args.img}" 的图')
    p = sorted(files)[0]

    im = cv2.imread(str(p))
    h, w = im.shape[:2]
    r = min(640 / h, 640 / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pt, pl = (640 - nh) // 2, (640 - nw) // 2
    canvas = np.full((640, 640, 3), 114, np.uint8)
    canvas[pt:pt + nh, pl:pl + nw] = cv2.resize(
        im, (nw, nh), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]   # BGR->RGB

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    bin_path = out / 'testcase.bin'
    bin_path.write_bytes(canvas.tobytes())

    x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    y = sess.run(None, {'images': np.ascontiguousarray(x)})[0].reshape(7, -1)

    best = y[4:].max(0)
    dets = []
    for i in np.where(best > args.conf)[0]:
        k = int(y[4:, i].argmax())
        cx, cy, bw, bh = y[0, i], y[1, i], y[2, i], y[3, i]
        # 反 letterbox 回原图坐标，便于人工核对
        x1 = (cx - bw / 2 - pl) / r
        y1 = (cy - bh / 2 - pt) / r
        x2 = (cx + bw / 2 - pl) / r
        y2 = (cy + bh / 2 - pt) / r
        dets.append({'cls': NAMES[k], 'score': round(float(best[i]), 4),
                     'xyxy_orig': [round(float(v), 1) for v in (x1, y1, x2, y2)]})
    dets.sort(key=lambda d: -d['score'])

    meta = {'src': p.name, 'size': [w, h], 'scale': round(r, 6),
            'pad_top': pt, 'pad_left': pl,
            'conf': args.conf, 'n_above_conf': len(dets),
            'top': dets[:12]}
    (out / 'expect.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                    encoding='utf-8')

    print(f'源图 {p.name}  {w}x{h}  scale={r:.4f} pad=({pt},{pl})')
    print(f'写出 {bin_path}  {bin_path.stat().st_size} 字节（应为 640*640*3=1228800）')
    print(f'ONNX 参考检出（conf>{args.conf}）共 {len(dets)} 个，前 5：')
    for d in dets[:5]:
        print(f"   {d['cls']:<8} {d['score']:.3f}  {d['xyxy_orig']}")


if __name__ == '__main__':
    main()
