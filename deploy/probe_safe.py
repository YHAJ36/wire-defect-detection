# 零风险探测：不初始化屏幕/摄像头，只问「API 长什么样」。
# 用于确定 ai2d 到底该喂什么类型的输入、PipeLine/YOLO 官方封装有哪些方法。
import sys
import nncase_runtime as nn

def show(tag, obj):
    try:
        print("%-22s %s" % (tag, [a for a in dir(obj) if not a.startswith("_")]))
    except Exception as e:
        print("%-22s FAIL %s" % (tag, type(e).__name__))

import image
show("image.Image", image.Image)

from media.sensor import Sensor
print("Sensor 常量:", [a for a in dir(Sensor) if a.isupper()])

try:
    from libs import PipeLine as PL
    show("PipeLine", PL)
    show("PipeLine.PipeLine", PL.PipeLine)
except Exception as e:
    print("PipeLine import FAIL", type(e).__name__, e)

try:
    from libs import YOLO as Y
    show("YOLO module", Y)
    show("YOLO.YOLO", Y.YOLO)
except Exception as e:
    print("YOLO import FAIL", type(e).__name__, e)

try:
    from libs import AI2D as A
    show("AI2D module", A)
except Exception as e:
    print("AI2D import FAIL", type(e).__name__, e)

show("nn.ai2d", nn.ai2d)
a = nn.ai2d()
b = a.build([1, 3, 480, 640], [1, 3, 640, 640])
show("ai2d_builder", b)
try:
    b.release()
except Exception:
    pass
print("=== 探测结束 ===")
