# -*- coding: utf-8 -*-
"""读修复版 SA conv 的 bias，验证"修复版残余 DC 是不是 bias"。

理论预测（来自 cbam_fixed.py 的中心化改动）：
    改动 2 把 SA 输入 y 逐通道去空间均值，故 y 每通道的空间均值恒为 0。
    conv 是 2→1、kernel=7、padding=3。忽略零 padding 的边缘效应时，
        mean_spatial(z) = sum_c w_c * mean(y_c) + bias = 0 + bias = bias
    ⇒ **修复版 z 的空间均值应当精确等于 bias**，与输入图像无关。

    这就是"噪声对照 ≈ 图像 DC"的原因（见 logs/cbam_fixed0_probe.txt：
    五层 DC 与噪声对照差均 < 0.002）。

    推论：若实测 mean(z) ≈ bias 成立，则修复版 DC 的增长**完全等价于 bias 的增长**，
    即中心化改动把 DC 的来源从"输入均值 × 权重"换成了"bias 本身"。
    bias 是自由参数、无任何机制约束它 → 边界 B（延后 vs 修好）问的就是
    "bias 会不会在 60 轮内长到原版的量级"。

用法：
    D:/yolo-env/Scripts/python.exe D:/2/code/tools/check_sa_bias.py
    # 指定检查点（tag=路径，可多个）：读 60 轮那趟的中间权重
    D:/yolo-env/Scripts/python.exe D:/2/code/tools/check_sa_bias.py \
        --ckpt ep20=D:/2/runs/wire_v11_640_cbamfix/weights/epoch20.pt \
        --ckpt ep40=D:/2/runs/wire_v11_640_cbamfix/weights/epoch40.pt
"""
import argparse
import sys

sys.path.insert(0, 'D:/2/code/yolov11')

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# 标签**不写死训练轮数**：best.pt 是按 fitness 选的、且训练在跑时它一直在变。
# 实测踩过的坑：曾把 wire_v11_640_cbamfix/weights/best.pt 标为 'fixed@60'，
# 而它在 ep41 那一刻与 epoch40.pt **SHA256 完全相同**（那时最好轮就是 ep40），
# 结果输出里 'fixed@60' 与 'ep40' 两行逐位相同 —— 若照此下结论就是错的。
# 故改成从检查点**自读 epoch**并打印，标签只标来源。
CKPTS = [
    ('fixed@0 ', 'D:/2/logs/_probe0/fixed0.pt'),
    ('fixed@20', 'D:/2/runs/wire_v11_640_cbamfix_probe20/weights/best.pt'),
    ('fix60run', 'D:/2/runs/wire_v11_640_cbamfix/weights/best.pt'),
]
# 对照：同一工具实测的 z 空间均值（去 padding，8 张测试图）
MEASURED = {
    'fixed@0 ': {2: 0.0193, 4: -0.0418, 6: -0.0806, 8: 0.1149, 9: 0.0776},
    'fixed@20': {2: 0.1173, 4: 0.7874, 6: 0.5214, 8: 0.1149, 9: -0.0862},
}

LAYERS = [2, 4, 6, 8, 9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', default=[], metavar='TAG=PATH',
                    help='额外的检查点（可重复）；不给则用内置的 fixed@0/@20/@60 三档')
    args = ap.parse_args()
    ckpts = list(CKPTS)
    for spec in args.ckpt:
        tag, _, path = spec.partition('=')
        ckpts.append((tag, path))

    print(f'{"ckpt":<10} {"ckpt内epoch":>11} {"layer":>5} {"bias":>14} {"实测 mean(z)":>14} {"差":>12}')
    for tag, path in ckpts:
        # 先直接读检查点字典取 epoch —— 不要相信标签里的轮数
        ep = '?'
        try:
            ck = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(ck, dict):
                ep = ck.get('epoch', '?')
        except Exception:
            pass
        # best.pt 里存的 epoch 是 -1（ultralytics 未回填），故显式标出"轮数未知"
        if ep == -1 or (isinstance(path, str) and path.replace('\\', '/').endswith('best.pt')):
            ep = 'best?'
        try:
            m = YOLO(path).model
        except Exception as e:  # 目录不存在（如某档尚未训）
            print(f'{tag}: 跳过（{type(e).__name__}: {e}）')
            continue
        seq = m.model
        for i in LAYERS:
            layer = seq[i]
            sub = layer[1] if hasattr(layer, '__len__') else layer
            b = getattr(getattr(sub, 'sa', None), 'conv', None)
            if b is None or b.bias is None:
                print(f'{tag} layer {i}: 无 sa.conv.bias')
                continue
            bias = float(b.bias.detach().reshape(-1)[0])
            meas = MEASURED.get(tag, {}).get(i)
            if meas is None:
                print(f'{tag:<10}{str(ep):>11} layer {i}: bias = {bias:+.6f}   （无实测对照）')
            else:
                print(f'{tag:<10}{str(ep):>11} layer {i}: bias = {bias:+.6f}   '
                      f'实测 = {meas:+.6f}   差 = {bias - meas:+.6f}')


if __name__ == '__main__':
    main()
