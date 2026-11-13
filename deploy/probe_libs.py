import os
for d in ("/sdcard", "/sdcard/libs", "/sdcard/examples"):
    try:
        print(d, "->", os.listdir(d))
    except Exception as e:
        print(d, "N/A", type(e).__name__)
import nncase_runtime as nn
print("nn attrs:", [a for a in dir(nn) if not a.startswith("_")])
try:
    print("ai2d_format:", [a for a in dir(nn.ai2d_format) if not a.startswith("_")])
except Exception as e:
    print("no ai2d_format", e)
