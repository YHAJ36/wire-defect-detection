import nncase_runtime as nn
import ulab.numpy as np
print("module funcs:", [a for a in dir(nn) if not a.startswith("_")])
print("ai2d class:", [a for a in dir(nn.ai2d) if not a.startswith("_")])
a = nn.ai2d()
a.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
a.set_pad_param(True, [0, 0, 0, 0, 80, 80, 0, 0], 0, (114, 114, 114))
a.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
r = a.build([1, 3, 480, 640], [1, 3, 640, 640])
print("build returned:", type(r), r)
if r is not None:
    print("  its attrs:", [x for x in dir(r) if not x.startswith("_")])
print("kpu class:", [x for x in dir(nn.kpu) if not x.startswith("_")])
k = nn.kpu()
print("kpu inst:", [x for x in dir(k) if not x.startswith("_")])
