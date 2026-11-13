# 04_verify_kmodel.py —— 编译完了，怎么知道能不能用？
#
# 不能直接拷到板子上试。原因是：量化一定有误差，而它**不会报错**。
# 如果板上框歪了，你分不清是"量化掉的"还是"板上解码写错了"。
# 所以在 PC 上先用 nncase 自带的模拟器（Simulator）跑同一个 kmodel，
# 和 ONNX 喂同一张图对比，误差就被单独隔离出来了：
#
#   Simulator 的 kmodel 输出    vs    ONNX 的输出
#              ↑ 两者唯一差别就是量化。差多少 = 纯量化误差。
#
# 用哪张图：testcase.bin —— 板上跑过的同一张，这样 PC 和板端可以对着看。
#
# 运行（PowerShell，一行）：
#   $env:DOTNET_ROLL_FORWARD="LatestMajor"; & D:/yolo-env/Scripts/python.exe D:/2/manual/04_verify_kmodel.py

import os
import sys
from pathlib import Path

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

KMODEL = r'D:/2/manual/yolo11n_640_manual.kmodel'
ONNX = r'D:/2/manual/yolo11n_640.onnx'
BIN = r'D:/2/deploy/board_case/testcase.bin'
NAMES = ['break', 'thunderbolt', 'wear']

# ---- 输入：和上次检查 ONNX 时一模一样 ----
raw = np.fromfile(BIN, dtype=np.uint8).reshape(640, 640, 3)
x = (raw.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)

# ---- 标准答案：ONNX（float32，几乎没有误差）----
sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
yo = sess.run(None, {'images': x})[0]          # (1, 7, 8400)

# ---- 待测：kmodel（已经压成 int8）----
sim = nncase.Simulator()
sim.load_model(Path(KMODEL).read_bytes())
print(f'kmodel  {KMODEL}  {Path(KMODEL).stat().st_size / 1e6:.2f} MB')
print(f'  输入 {sim.get_input_shape(0)}   输出 {sim.get_output_shape(0)}')

sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(x))
sim.run()
yk = sim.get_output_tensor(0).to_numpy()
if yk.ndim == 3 and yk.shape[0] == 1:
    yk = yk[0]
yk = yk.reshape(yo.shape)

# ---- 比较 ----
d = float(np.abs(yo - yk).max())
scale = float(np.abs(yo).max())
print(f'\n最大绝对误差 {d:.4f}   相对 {d / scale * 100:.2f}%')
print('判读：相对误差 < 2% 说明量化正常（int8 只有 256 个格，必然有误差，不是 0 才对）')

# 框的位置有没有歪？框参数（前 4 行）是像素，直接比像素差
box_o, box_k = yo[0, :4], yk[0, :4]
diff = np.abs(box_o - box_k)
print(f'  框参数最大偏移 {diff.max():.2f} px')
print(f'  但 8400 个格子里，绝对误差的中位数 {np.median(diff):.3f} px，'
      f'99% 的格子小于 {np.percentile(diff, 99):.2f} px')
print(f'  误差超过 5 px 的格子 {int((diff > 5).sum())} 个 / 共 {diff.size} 个（大多是背景格子，'
      f'它们本来就不该出框）')

# 分数掉了多少？类别分数（后 3 行）
so, sk = yo[0, 4:].max(0), yk[0, 4:].max(0)
k = int(so.argmax())
print(f'\n最高的那个框：')
print(f'  ONNX   类别 {NAMES[int(yo[0, 4:, k].argmax())]}  分数 {so[k]:.3f}  '
      f'中心 ({yo[0, 0, k]:.1f}, {yo[0, 1, k]:.1f})')
print(f'  kmodel 类别 {NAMES[int(yk[0, 4:, k].argmax())]}  分数 {sk[k]:.3f}  '
      f'中心 ({yk[0, 0, k]:.1f}, {yk[0, 1, k]:.1f})')
print(f'  分数比 {sk[k] / so[k]:.3f}   中心差 {abs(yo[0, 0, k] - yk[0, 0, k]):.2f} / '
      f'{abs(yo[0, 1, k] - yk[0, 1, k]):.2f} px')
print(f'\n对照：板上实测中心 (342.3, 257.3)')
print(f'所以这个 kmodel 的预期表现：中心约 ({yk[0, 0, k]:.1f}, {yk[0, 1, k]:.1f})，'
      f'分数约 {sk[k]:.2f}')
