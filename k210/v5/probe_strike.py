# -*- coding: utf-8 -*-
"""定阈值用：挑真实雷击图，传板上跑，看 obj 响应范围。

流程：
  1. 从 test 集挑 3 张含雷击的图（按框面积挑最明显的），resize 到 320x224
  2. 上传到板子 /sd/s0.jpg ~ s2.jpg
  3. 板上加载 strike.kmodel 推理，打印三层 obj 最大值

为什么必须板上跑：PC 模型是 fp32，板上是 uint8 量化，
obj 绝对数值会有偏移，阈值必须按板上实测来定。
"""
import glob
import os
import sys

from PIL import Image

SRC = r'D:/2/dataset_noleak'
TMP = r'D:/2/k210/v5/demo_strike'
N = 3
CLS = 1                     # 雷击

sys.path.insert(0, r'D:/2/k210')
from k210 import K210  # noqa: E402


def pick_images():
    """挑含雷击且框面积最大的 N 张图。"""
    cand = []
    for f in sorted(glob.glob(os.path.join(SRC, 'images/test/*'))):
        stem = os.path.splitext(os.path.basename(f))[0]
        lab = os.path.join(SRC, 'labels/test', stem + '.txt')
        if not os.path.exists(lab):
            continue
        best_area = 0.0
        for ln in open(lab):
            p = ln.split()
            if len(p) >= 5 and int(float(p[0])) == CLS:
                best_area = max(best_area, float(p[3]) * float(p[4]))
        if best_area > 0:
            cand.append((best_area, f, stem))
    cand.sort(reverse=True)
    return cand[:N]


def main():
    os.makedirs(TMP, exist_ok=True)
    picks = pick_images()
    print(f'[pick] {len(picks)} 张雷击图（按框面积排序）')

    files = []
    for i, (area, f, stem) in enumerate(picks):
        im = Image.open(f).convert('RGB').resize((320, 224), Image.BILINEAR)
        out = os.path.join(TMP, f's{i}.jpg')
        im.save(out, quality=92)
        files.append(out)
        print(f'  s{i}.jpg  {stem}  框面积比 {area*100:.2f}%')

    d = K210()
    for i, p in enumerate(files):
        data = open(p, 'rb').read()
        d.write_file_batch(f'/sd/s{i}.jpg', data, chunk=4096, per=4)
        got = d.exec(f"import os\nprint(os.stat('/sd/s{i}.jpg')[6])").strip()
        print(f'[up ] s{i}.jpg {len(data)} -> {got} {"OK" if got == str(len(data)) else "FAIL"}')

    code = '''
import image, gc
import KPU as kpu

task = kpu.load("/sd/strike.kmodel")
kpu.set_outputs(task, 0, 28, 40, 18)
kpu.set_outputs(task, 1, 14, 20, 18)
kpu.set_outputs(task, 2, 7, 10, 18)
gc.collect()
HEADS = ((28,40), (14,20), (7,10))
print("free after load", gc.mem_free())

for i in range(3):
    gc.collect()
    img = image.Image("/sd/s%d.jpg" % i)
    img.pix_to_ai()          # 这一步吃掉 320*224*3 字节，之后堆只剩约 150KB
    kpu.forward(task, img)
    line = "s%d" % i
    for hi in range(3):
        ny, nx = HEADS[hi]
        NS = ny * nx
        o = kpu.get_output(task, hi)      # 每次分配 ny*nx*18*4 字节浮点数组
        mx = -99.0
        for a in range(3):
            base = (a * 6 + 4) * NS
            for k in range(NS):
                v = o[base + k]
                if v > mx:
                    mx = v
        line += "  h%d=%.2f" % (hi, mx)
        del o                              # 必须立刻释放，否则三层叠加必 OOM
        gc.collect()
    print(line)
    del img
    gc.collect()
kpu.deinit(task)
print("PROBE DONE")
'''
    print(d.exec(code, timeout=180))
    d.close()


if __name__ == '__main__':
    main()
