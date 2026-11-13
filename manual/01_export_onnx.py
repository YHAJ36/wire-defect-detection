# -*- coding: utf-8 -*-
"""第 2 步：把 best.pt 导成 ONNX。

为什么要自己写这几行，而不是用官方的 yolo export？
  因为 ultralytics 8.3.180 默认走"新导出器"，它需要 onnxscript 这个包，
  本机没有 → 报 No module named 'onnxscript'。
  老导出器不需要它，但必须手动调 torch.onnx.export(..., dynamo=False)。

跑法（PowerShell）：
  & D:/yolo-env/Scripts/python.exe D:/2/manual/01_export_onnx.py
"""
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

# ---- 三个路径，想换模型只改这里 ----
PT = 'D:/2/exp_upscale/runs/nf_v11n_nl640opt_s42/weights/best.pt'
OUT = 'D:/2/manual/yolo11n_640.onnx'
SIZE = 640                      # 必须等于训练时的 imgsz，否则精度会崩


# ① 加载权重，拿到里面的"纯网络"（YOLO 这个对象是外壳，.model 才是 nn.Module）
#    .float()  把所有参数转成 fp32 —— 有些权重存的是 fp16，导出前统一
#    .eval()   切到推理模式 —— 关掉 dropout，BN 用训练时统计好的均值方差
#    .fuse()   把 Conv 和紧跟的 BN 合并成一个 Conv，推理更快，数值上等价
model = YOLO(PT).model.float().eval().fuse()

# ② 改检测头的四个开关。这一步最容易漏，漏了导出的东西不能用：
for m in model.modules():                 # 遍历所有子模块
    if isinstance(m, Detect):              # 只挑检测头（YOLO11 只有一个）
        m.inplace = False                  # 别原地改张量，导出时 tracing 会崩
        m.export = True                    # 输出"解码后"的格式：cx,cy,w,h + 各类分数
        m.dynamic = False                  # 固定输入尺寸，不做动态 shape
        m.format = 'onnx'                  # 告诉检测头按 onnx 的方式组织输出

# ③ 造一个假输入。它不参与计算，只用来"跑一遍、把计算过程记下来"（叫 tracing）
#    形状含义：[批量, 通道, 高, 宽]。batch=1，RGB 三通道，640x640
im = torch.zeros(1, 3, SIZE, SIZE)

# ④ 真正导出。每个参数是什么意思：
#    model, im            要导的模型 + 假输入
#    OUT                  输出文件路径
#    opset_version=12     算子集版本。nncase 2.11 认到 12 左右，太高会不认
#    input_names          给输入张量起名，板上喂数据时按这个名字对
#    output_names         给输出张量起名
#    do_constant_folding  把能提前算的常量提前算掉，图更小
#    dynamo=False         ★ 关键：走老导出器，不需要 onnxscript
torch.onnx.export(model, im, OUT, opset_version=12,
                  input_names=['images'], output_names=['output0'],
                  do_constant_folding=True, dynamo=False)

print('导出完成 ->', OUT)
print('文件大小 :', round(__import__('os').path.getsize(OUT) / 1048576, 2), 'MB')

# ⑤ 打开导出的文件，看看它长什么样。这一输出是后面所有推理代码的依据
import onnx
g = onnx.load(OUT).graph
for t in g.input:
    print('输入 :', t.name, [d.dim_value for d in t.type.tensor_type.shape.dim])
for t in g.output:
    print('输出 :', t.name, [d.dim_value for d in t.type.tensor_type.shape.dim])
