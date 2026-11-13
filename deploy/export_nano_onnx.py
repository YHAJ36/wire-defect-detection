"""把 yolo11n best.pt 导成 ONNX（legacy 导出路子，绕开 onnxscript）。

ultralytics 8.3.180 的 m.export() 走 dynamo 分支，需要 onnxscript，本机装不上；
直接调 torch.onnx.export(dynamo=False) 走旧的 symbolic 路径。
导出后若有多输出，只留 [1, 7, 8400] 那一个（后面 make_uint8_onnx 只认单输出）。
"""
import copy
import sys
from pathlib import Path

import onnx
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

PT = Path(sys.argv[1] if len(sys.argv) > 1 else
          'D:/2/exp_upscale/runs/nf_v11n_nl640opt_s42/weights/best.pt')
SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 640
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else 'D:/2/deploy/yolo11n_640.onnx')

model = copy.deepcopy(YOLO(str(PT)).model).float().eval()
model = model.fuse()
for m in model.modules():
    if isinstance(m, Detect):
        m.inplace = False
        m.export = True
        m.dynamic = False
        m.format = 'onnx'

im = torch.zeros(1, 3, SIZE, SIZE)
with torch.no_grad():
    y = model(im)
print('torch 前向输出:', [tuple(t.shape) for t in y] if isinstance(y, (list, tuple)) else tuple(y.shape))

torch.onnx.export(model, im, str(OUT), opset_version=12,
                  input_names=['images'], output_names=['output0'],
                  do_constant_folding=True, dynamo=False)
print('写出', OUT, OUT.stat().st_size // 1024, 'KB')

# 只留检测头那一路
g = onnx.load(str(OUT)).graph
keep = None
for o in g.output:
    d = [x.dim_value for x in o.type.tensor_type.shape.dim]
    print('  输出', o.name, d)
    if len(d) == 3 and d[0] == 1 and d[1] == 7:
        keep = o
if keep is not None and len(g.output) > 1:
    name = keep.name
    del g.output[:]
    g.output.append(keep)
    onnx.save(onnx.load(str(OUT)), str(OUT))
    print('  已裁成单输出:', name)
