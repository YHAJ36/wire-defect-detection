# -*- coding: utf-8 -*-
"""上板之前在 PC 上验证 kmodel：nncase Simulator vs ONNX。

为什么必须先做这一步：
  INT8 量化会引入误差，但**不会报错**。如果直接拷到板子上跑，
  看到的掉点分不清是"量化掉的"还是"板上解码写错了"。
  这里先用模拟器把两者分开：
    · Simulator 的输出  vs  ONNX 输出     → 纯量化误差
    · 解码后的框一致性                     → 后处理是否对齐

用法
    set DOTNET_ROLL_FORWARD=LatestMajor
    D:/yolo-env/Scripts/python.exe D:/2/deploy/verify_kmodel.py \
        --kmodel D:/2/deploy/yolo11s_640.kmodel --n 20
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault('DOTNET_ROLL_FORWARD', 'LatestMajor')
# Windows 上 nncase.simulator.k230.dll 的依赖分布在 nncase 子目录里，
# 不注册会报 "The specified module could not be found"（误导性报错，其实是找不到依赖）
_SP = Path(sys.prefix) / 'Lib' / 'site-packages'
for _d in (_SP, _SP / 'nncase', _SP / 'nncase' / 'modules' / 'kpu'):
    if _d.is_dir():
        try:
            os.add_dll_directory(str(_d))
        except Exception:
            pass
os.environ.setdefault('NNCASE_PLUGIN_PATH', str(_SP / 'nncase' / 'modules' / 'kpu'))
# k230 模拟器实际是子进程调用 nncase.simulator.k230.sc.exe，
# 不把它所在目录塞进 PATH 会报一个极具误导性的 "io error"
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
    im = cv2.imread(str(path))
    if im is None:
        return None
    im = letterbox(im, size)[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(im.transpose(2, 0, 1)[None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kmodel', default='D:/2/deploy/yolo11s_640.kmodel')
    ap.add_argument('--onnx', default='D:/2/deploy/yolo11s_640_single.onnx')
    ap.add_argument('--imgs', default='D:/2/dataset_noleak/images/test')
    ap.add_argument('--n', type=int, default=20)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    sim = nncase.Simulator()
    sim.load_model(Path(args.kmodel).read_bytes())
    print(f'kmodel: {args.kmodel}  {Path(args.kmodel).stat().st_size/1e6:.2f} MB')
    n_in, n_out = sim.inputs_size, sim.outputs_size
    print('  inputs :', n_in, [sim.get_input_shape(i) for i in range(n_in)])
    print('  outputs:', n_out, [sim.get_output_shape(i) for i in range(n_out)])

    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])

    files = sorted([p for p in Path(args.imgs).iterdir()
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    step = max(1, len(files) // args.n)
    picked = files[::step][:args.n]
    print(f'\n对比 {len(picked)} 张（{args.imgs}）\n')

    diffs, rels, agree = [], [], []
    for p in picked:
        x = prep(p)
        if x is None:
            continue
        yo = sess.run(None, {'images': x})[0]

        sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(x))
        sim.run()
        yk = sim.get_output_tensor(0).to_numpy()
        if yk.ndim == 3 and yk.shape[0] == 1:
            yk = yk[0]
        yk = yk.reshape(yo.shape)

        d = float(np.abs(yo - yk).max())
        scale = float(np.abs(yo).max()) or 1.0
        diffs.append(d)
        rels.append(d / scale)

        # 类别置信度层面的一致性：取每类最大 score
        co, ck = yo[0, 4:, :], yk[0, 4:, :]
        agree.append(float(np.abs(co.max(1) - ck.max(1)).mean()))

        print(f'  {p.name[:42]:<44} max|Δ|={d:.4f}  rel={d/scale*100:5.2f}%  '
              f'cls_score_Δ={agree[-1]:.4f}')

    print(f'\n汇总  max|Δ| 中位 {np.median(diffs):.4f}  最大 {max(diffs):.4f}')
    print(f'      相对误差 中位 {np.median(rels)*100:.3f}%  最大 {max(rels)*100:.3f}%')
    print(f'      类别分数平均偏差 {np.mean(agree):.4f}')
    print('\n判读：相对误差 < 2% 视为量化正常；> 5% 说明校准集或预处理有问题。')


if __name__ == '__main__':
    main()
