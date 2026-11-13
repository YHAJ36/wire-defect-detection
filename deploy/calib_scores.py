# -*- coding: utf-8 -*-
"""量化后的分数标定：同一份 uint8 NHWC 输入，分别喂 ONNX 与 kmodel 仿真器，
收集 (PC 分, 板端分) 的配对，用来回答一个具体问题——

    「板上显示的 0.36，到底等于 PC 上的多少？」

背景：A.9.78 只核对了 top1 一个点（0.597 → 0.361，0.605×），由此推论出
「按 0.6 折算阈值」。这个推论是错的。对多张图做配对后发现压缩比不是常数：
低分区压到 0.45×，高分区只压到 0.6~0.9×。本节即把这个映射量出来。

用法
    set DOTNET_ROLL_FORWARD=LatestMajor
    D:/yolo-env/Scripts/python.exe D:/2/deploy/calib_scores.py --n 40
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault('DOTNET_ROLL_FORWARD', 'LatestMajor')
_SP = Path(sys.prefix) / 'Lib' / 'site-packages'
for _d in (_SP, _SP / 'nncase', _SP / 'nncase' / 'modules' / 'kpu'):
    if _d.is_dir():
        try:
            os.add_dll_directory(str(_d))
        except Exception:
            pass
os.environ.setdefault('NNCASE_PLUGIN_PATH', str(_SP / 'nncase' / 'modules' / 'kpu'))
os.environ['PATH'] = str(_SP) + os.pathsep + os.environ.get('PATH', '')

import nncase  # noqa: E402
import onnxruntime as ort  # noqa: E402


def letterbox(im, new=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new - nh) // 2, (new - nw) // 2
    return cv2.copyMakeBorder(im, top, new - nh - top, left, new - nw - left,
                              cv2.BORDER_CONSTANT, value=color)


def prep(path, size=640):
    """uint8 NHWC，与板上喂进 KPU 的那块内存完全同格式。"""
    im = cv2.imread(str(path))
    if im is None:
        return None
    im = letterbox(im, size)[:, :, ::-1]          # BGR -> RGB
    return np.ascontiguousarray(im[None], dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kmodel', default='D:/2/deploy/yolo11n_640_u8r.kmodel')
    ap.add_argument('--onnx', default='D:/2/deploy/yolo11n_640_u8.onnx')
    ap.add_argument('--imgs', default='D:/2/dataset_noleak/images/test')
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--topk', type=int, default=100)
    ap.add_argument('--out', default='D:/2/deploy/score_pairs.npz')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    def new_sim():
        s = nncase.Simulator()
        s.load_model(Path(args.kmodel).read_bytes())
        return s

    sim = new_sim()
    print(f'kmodel {args.kmodel}  {Path(args.kmodel).stat().st_size / 1e6:.2f} MB')
    print('  in ', [sim.get_input_shape(i) for i in range(sim.inputs_size)])
    print('  out', [sim.get_output_shape(i) for i in range(sim.outputs_size)])
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    print('onnx  ', [(i.name, i.shape, i.type) for i in sess.get_inputs()])

    files = sorted(p for p in Path(args.imgs).iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    step = max(1, len(files) // args.n)
    picked = files[::step][:args.n]
    print(f'\n配对 {len(picked)} 张\n')

    SO, SK, IMG = [], [], []
    for k, p in enumerate(picked):
        x = prep(p)
        if x is None:
            continue
        raw = sess.run(None, {'images': x})[0]
        yo = raw.reshape(raw.shape[-2], raw.shape[-1])      # (7, 8400)

        # k230 仿真器是子进程，偶发 "io error" 后句柄即失效，只能重建
        yk = None
        for attempt in range(3):
            try:
                sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(x))
                sim.run()
                yk = sim.get_output_tensor(0).to_numpy().reshape(-1)
                break
            except Exception as e:
                print(f'      ! 仿真失败({attempt + 1}/3): {e}，重建')
                try:
                    sim = new_sim()
                except Exception:
                    pass
        if yk is None:
            print(f'      × 跳过 {p.name}')
            continue
        yk = yk.reshape(yo.shape)

        so = yo[4:7].max(0)
        sk = yk[4:7].max(0)
        idx = np.argsort(-so)[:args.topk]
        SO.append(so[idx].astype(np.float32))
        SK.append(sk[idx].astype(np.float32))
        IMG.append(np.full(len(idx), k, np.int16))
        print(f'  [{k + 1:3d}/{len(picked)}] {p.name[:36]:<38} '
              f'top1 {so[idx[0]]:.4f} -> {sk[idx[0]]:.4f}  '
              f'x{sk[idx[0]] / max(so[idx[0]], 1e-9):.3f}')
        # 边跑边存，防止中途崩掉丢掉前面全部结果
        np.savez_compressed(args.out, s_pc=np.concatenate(SO),
                            s_km=np.concatenate(SK), img=np.concatenate(IMG))

    SO = np.concatenate(SO)
    SK = np.concatenate(SK)
    IM = np.concatenate(IMG)
    np.savez_compressed(args.out, s_pc=SO, s_km=SK, img=IM)
    print(f'\n配对点 {len(SO)} 个 -> {args.out}')


if __name__ == '__main__':
    main()
