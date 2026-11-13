# -*- coding: utf-8 -*-
"""在 ONNX 图前面插一段「uint8 NHWC 0..255 → float32 NCHW 0..1」的转换。

为什么要这么绕：
  nncase 2.11.0 的 `preprocess=True` 这条路上有 bug —— 显式 mean/std 会在
  FoldNopBinary 折叠「减 0 / 乘 1」时报
  `Only The Expr Have CheckedType Can Get It's Shape`；
  改成 input_range=[0,1] 也在 import_onnx 阶段同一个 pass 崩掉。
  两条路都试过，都挂在 nncase 自身，跟我们的模型无关。

  既然编译器的预处理通道走不通，就把它写进图里：
  板子把 ai2d 出来的 uint8 RGB888（NHWC，1.2 MB）直接喂进去，
  Cast→Div→Transpose 由 nncase 排在 CPU 上做，MicroPython 侧零逐像素开销。

用法
    D:/yolo-env/Scripts/python.exe D:/2/deploy/make_uint8_onnx.py \
        --src D:/2/deploy/yolo11s_640_single.onnx \
        --out D:/2/deploy/yolo11s_640_u8.onnx

    # NCHW 版：与官方 libs/AI2D.py（set_ai2d_dtype 用 NCHW_FMT）对齐，
    # 图里只需要 Cast -> Div，不需要 Transpose
    python make_uint8_onnx.py --layout nchw --out yolo11s_640_u8nchw.onnx
"""
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='D:/2/deploy/yolo11s_640_single.onnx')
    ap.add_argument('--out', default='D:/2/deploy/yolo11s_640_u8.onnx')
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--layout', choices=['nhwc', 'nchw'], default='nhwc',
                    help='输入内存布局。nhwc 给自写 ai2d 用；nchw 给官方 '
                         'libs/AI2D.py 用（set_ai2d_dtype 默认 NCHW_FMT）。')
    args = ap.parse_args()

    m = onnx.load(args.src)
    g = m.graph
    old_in = g.input[0]
    print('原输入:', old_in.name, [d.dim_value for d in old_in.type.tensor_type.shape.dim])

    # 把原输入改名，挂到新链条后面
    orig_name = old_in.name + '_f32'
    for node in g.node:
        for i, n in enumerate(node.input):
            if n == old_in.name:
                node.input[i] = orig_name

    # 新输入：uint8
    if args.layout == 'nhwc':
        new_in = helper.make_tensor_value_info(
            'images', TensorProto.UINT8, [1, args.size, args.size, 3])
    else:
        new_in = helper.make_tensor_value_info(
            'images', TensorProto.UINT8, [1, 3, args.size, args.size])

    # Cast -> Div(255) [-> Transpose(NHWC->NCHW)]
    c_cast = helper.make_node('Cast', ['images'], ['x_f32'],
                              to=TensorProto.FLOAT, name='u8_cast')
    c_scale = numpy_helper.from_array(np.array(255.0, dtype=np.float32), 'scale_255')
    c_div = helper.make_node('Div', ['x_f32', 'scale_255'], ['x_01'], name='u8_norm')
    head = [c_cast, c_div]
    if args.layout == 'nhwc':
        head.append(helper.make_node('Transpose', ['x_01'], [orig_name],
                                     perm=[0, 3, 1, 2], name='u8_hwc2chw'))
    else:
        head.append(helper.make_node('Identity', ['x_01'], [orig_name],
                                     name='u8_nchw_passthrough'))

    del g.node[:]  # 清空后按顺序重排：新节点在前
    g.node.extend(head)
    m2 = onnx.load(args.src)
    for node in m2.graph.node:
        for i, n in enumerate(node.input):
            if n == m2.graph.input[0].name:
                node.input[i] = orig_name
    g.node.extend(m2.graph.node)

    del g.input[:]
    g.input.append(new_in)
    # 常量 scale_255 要注册进 initializer
    g.initializer.append(c_scale)

    onnx.checker.check_model(m)
    onnx.save(m, args.out)
    print('写出:', args.out)
    shape = ('[1,%d,%d,3]' % (args.size, args.size)) if args.layout == 'nhwc' \
        else ('[1,3,%d,%d]' % (args.size, args.size))
    print('输入:', 'images uint8 %s (%s)' % (shape, args.layout))
    print('输出:', [o.name for o in g.output], [d.dim_value for d in g.output[0].type.tensor_type.shape.dim])


if __name__ == '__main__':
    main()
