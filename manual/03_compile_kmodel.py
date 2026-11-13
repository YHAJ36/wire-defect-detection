# 03_compile_kmodel.py —— 把 ONNX 编译成 kmodel（K230 能读的格式）
#
# 这一步是整个转换里唯一"重"的一步，也是唯一必须看代码的一步。
# 但真正要做的事只有四件，代码按这四件事分成四段：
#
#   第 1 段  准备校准图：从训练集里抽 50 张，按训练时的方式处理好
#   第 2 段  声明输入：告诉编译器"我要喂什么形状、什么类型、什么排布"
#   第 3 段  交给它校准（PTQ）：让它看这 50 张图，统计每一层数值的范围
#   第 4 段  编译 + 存盘：算出 int8 权重和 scale，输出成 .kmodel
#
# 为什么非要那 50 张图（这就是上一条我没听到回答的问题）：
#   训练好的权重本来是 float32，数值范围是连续的。要压进 int8（只有 -128~127
#   共 256 个格），必须知道"这一层的数值大概在什么范围里"，才能决定 256 个格
#   怎么铺。范围猜小了，超出的部分全被削平；猜大了，256 个格挤在一小段上，
#   分辨率全浪费。这个"范围"不是猜的，是喂几十张真实图片统计出来的——这就叫
#   校准（calibration）。所以那几十张图不是用来"再训练一次"，是用来定标尺的。
#   标尺定歪了，模型照样跑得动，但分数会整体偏低或者高分段被削平，且不报错。
#
# 运行（PowerShell，一行）：
#   $env:DOTNET_ROLL_FORWARD="LatestMajor"; & D:/yolo-env/Scripts/python.exe D:/2/manual/03_compile_kmodel.py

import argparse
import io
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---- 下面 8 行是 Windows 上跑 nncase 模拟器/编译器必须的，不是模型相关的东西 ----
# 不设 DOTNET_ROLL_FORWARD：.NET 找不到它要的 7.0 就直接拒绝启动
os.environ.setdefault('DOTNET_ROLL_FORWARD', 'LatestMajor')
# nncase 的 dll 分散在几个子目录里，不把它们登记进 dll 搜索路径，
# 会报 "The specified module could not be found"（其实是找不到依赖，不是找不到自己）
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


def letterbox(im, new=640, color=(114, 114, 114)):
    """把任意尺寸的图缩放到 640x640，保持长宽比，空出来的地方填灰色。

    训练时用的就是这个方式（ultralytics 默认），推理必须一模一样。
    如果你推理时改成"直接拉伸到 640"，模型看到的东西和训练时不一样，框会歪。
    """
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new - nh) // 2, (new - nw) // 2
    return cv2.copyMakeBorder(im, top, new - nh - top, left, new - nw - left,
                              cv2.BORDER_CONSTANT, value=color)


def build_calib(folder, n, size=640):
    """第 1 段：抽 n 张图，按训练时的方式处理成 (1,3,640,640) 的 float32。"""
    files = sorted(p for p in Path(folder).iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not files:
        raise SystemExit(f'校准目录里没有图片：{folder}')
    step = max(1, len(files) // n)          # 等距抽，避免全挤在同一段视频里
    picked = files[::step][:n]

    data = []
    for p in picked:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = letterbox(im, size)[:, :, ::-1]            # BGR -> RGB（opencv 读进来是 BGR）
        im = im.astype(np.float32) / 255.0              # 归一化到 0~1
        data.append(np.ascontiguousarray(im.transpose(2, 0, 1)[None]))  # HWC -> CHW + batch
    print(f'校准图 {len(data)} 张（从 {len(files)} 张里每 {step} 张取一张）')
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', default='D:/2/manual/yolo11n_640.onnx')
    ap.add_argument('--out', default='D:/2/manual/yolo11n_640_manual.kmodel')
    ap.add_argument('--calib-dir', default='D:/2/dataset_noleak/images/train')
    ap.add_argument('--calib', type=int, default=50)
    ap.add_argument('--method', default='NoClip', help='Kld 还是 NoClip，见下面注释')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    cali = build_calib(args.calib_dir, args.calib)

    # ---------------- 第 2 段：声明输入 ----------------
    o = nncase.CompileOptions()
    o.target = 'k230'                 # 目标芯片，决定指令集和内存布局
    o.input_shape = [1, 3, 640, 640]  # 逻辑形状：1 张、3 通道、640x640
    o.output_layout = 'NCHW'
    o.input_type = 'float32'          # 我们喂 float（图里已经除过 255）
    o.input_layout = 'NCHW'
    o.preprocess = False              # 不让 nncase 再做一遍归一化，做两遍不报错但掉分
    o.dump_ir = False
    o.dump_asm = False

    c = nncase.Compiler(o)
    c.import_onnx(Path(args.onnx).read_bytes(), nncase.ImportOptions())
    print('import_onnx OK  （ONNX 语法和算子都合法）')

    # ---------------- 第 3 段：校准 ----------------
    q = nncase.PTQTensorOptions()
    q.samples_count = len(cali)
    # 校准方法决定"标尺"怎么定：
    #   Kld  —— 主动裁掉长尾，求整体误差最小。但它会把 YOLO 的高分框一起削掉，
    #           实测分数封顶在 0.544，所有高分框被压平，板上看起来"置信度只有五十"。
    #   NoClip —— 不裁剪，按最大绝对值定标尺。高分保住，低分区间精度差一点。
    # 检测任务关注高分框，所以选 NoClip。
    q.calibrate_method = args.method
    q.quant_type = 'uint8'            # 激活用 uint8：ReLU 后全非负，白赚一位精度
    q.w_quant_type = 'uint8'
    q.cali_data = [nncase.RuntimeTensor.from_numpy(a) for a in cali]
    c.use_ptq(q)
    print(f'use_ptq OK  samples={q.samples_count}  method={args.method}')

    # ---------------- 第 4 段：编译 + 存盘 ----------------
    t0 = time.time()
    c.compile()
    print(f'compile OK  {time.time() - t0:.1f}s')

    sink = io.BytesIO()
    c.gencode(sink)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(sink.getvalue())
    print(f'kmodel: {out}  {out.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
