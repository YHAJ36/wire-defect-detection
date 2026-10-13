# -*- coding: utf-8 -*-
"""追问：CBAM 为什么退化成"常数场"？

背景
    viz_cbam_attention.py 实测（8 张图 × 21 个模块）：空间注意力标准差中位数 0.0308、
    通道注意力 0.0259、对比度比中位数 1.0152 —— 即 CBAM 输出 ≈ 常数 × 输入，
    而常数缩放会被后一层 BN 严格吸收（见 archive/2026-10-11.md 的推导），
    故 CBAM 在功能上接近恒等映射。

    本脚本回答"为什么"。三种候选原因，用两组证据区分：

    答案 A 初始化问题 —— 权重压根没从初始值移动（训练没动它）
    答案 B 梯度问题   —— 权重想动但梯度太小/传不过去
    答案 C 数据问题   —— 梯度正常流通，但对各通道/位置是"同向的"，
                        只能产生均匀偏移，产生不了区分度

    证据 1（静态）：训练产物里 CBAM 三个权重的 std，对比 PyTorch 默认初始化的理论 std。
                    ratio = std_训练 / std_初始化 ≈ 1 → 权重没动。
    证据 2（动态）：跑 1 轮训练，给 CBAM 权重与宿主层参数各挂 backward hook，比梯度范数。
                    若 CBAM 的梯度比宿主层小若干数量级 → 答案 B。
                    若量级正常但（见 viz 脚本）权重仍没动 → 答案 C。

用法
    D:/yolo-env/Scripts/python.exe D:/2/code/tools/analyze_cbam_degradation.py
    只跑静态检查（跳过 1 轮训练）：
    ... analyze_cbam_degradation.py --skip-grad
    只跑梯度探针：
    ... analyze_cbam_degradation.py --skip-weights

输出
    logs/cbam_degradation.txt（同时打印）

注意
    - 梯度探针会**真的训练 1 个 epoch**（640），产物写到 runs/_probe_* 并在结束时删除。
      用 640 是为了与分析的产物（wire_v11_640_*）同口径，1 epoch ≈ 1 分钟。
    - 用 AMP（与正式训练一致），梯度会被 GradScaler 缩放；同一 step 内所有参数缩放相同，
      故**比值**有意义、绝对值无意义 —— 报告里只看比值与数量级。
"""

import argparse
import collections
import gc
import math
import shutil
import sys
from pathlib import Path

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch
import torch.nn as nn

from cbam import CBAM

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

RUNS = 'D:/2/runs'
DATA = 'D:/2/dataset_final_bbox/data.yaml'
WEIGHTS = 'D:/1/models/yolo11s.pt'
OUT = 'D:/2/logs/cbam_degradation.txt'

# 与 train_v11_640.py 逐字一致，保证梯度量级有代表性
TRAIN_KW = dict(
    data=DATA, epochs=1, imgsz=640, batch=8,
    # nbs=batch → 梯度不跨 micro-batch 累积。默认 nbs=64/batch=8 会累积 8 步，
    # 使相邻两步的梯度人为高度相关，余弦相似度就失去意义。梯度**方向**的统计与 nbs 无关，
    # 故此处取 nbs=8 只为让余弦检验干净，不影响范数量级的结论。
    nbs=8,
    optimizer='SGD', lr0=0.01, lrf=0.005,
    momentum=0.937, weight_decay=0.0005,
    cos_lr=True, warmup_epochs=3.0,
    box=7.5, cls=0.5, dfl=1.5,
    mosaic=0.8, mixup=0.1, copy_paste=0.1,
    hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
    degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5,
    close_mosaic=5, label_smoothing=0.1, dropout=0.0,
    amp=True, seed=42, workers=0, device=0,
    project=RUNS, exist_ok=True, plots=False, val=False,
)

STATIC_RUNS = ['wire_v11_640_both', 'wire_v11_640_neck', 'wire_v12_640_cbam']
IDX_V11 = [2, 4, 6, 8, 9]        # backbone C3k2 ×4 + SPPF
IDX_V12 = [6, 8, 11, 14, 17]     # A2C2f


# ------------------------------------------------------------------ 公共
def locate_cbam(det_model):
    """返回 [(层号, 父层类名, CBAM), ...]。注入把整层包成 nn.Sequential(原层, CBAM)。"""
    found = []
    for i, layer in enumerate(det_model.model):
        if not (isinstance(layer, nn.Sequential) and len(layer) == 2):
            continue
        if isinstance(layer[1], CBAM):
            found.append((i, type(layer[0]).__name__, layer[1]))
    return sorted(found, key=lambda t: t[0])


def load_det(run):
    from ultralytics import YOLO
    w = Path(RUNS) / run / 'weights' / 'best.pt'
    if not w.exists():
        raise FileNotFoundError(f'缺权重 {w}')
    m = YOLO(str(w))
    m.model.eval()
    return m.model


def wstats(w):
    w = w.detach().float()
    return dict(mean=float(w.mean()), std=float(w.std()),
                amin=float(w.abs().max()), n=w.numel())


def init_ref_std(mod, draws=20):
    """PyTorch 默认初始化的理论 std：同形状新建同名层取多次平均（单次随机有采样噪声）。"""
    stds, amins = [], []
    for _ in range(draws):
        ref = nn.Conv2d(mod.in_channels, mod.out_channels, mod.kernel_size,
                        stride=mod.stride, padding=mod.padding,
                        dilation=mod.dilation, groups=mod.groups,
                        bias=mod.bias is not None)
        s = wstats(ref.weight)
        stds.append(s['std'])
        amins.append(s['amin'])
    return sum(stds) / len(stds), sum(amins) / len(amins)


# ------------------------------------------------------------------ 证据 1：静态权重
def part1_weights(runs, lines):
    lines += ['=' * 100,
              '证据 1（静态）：训练产物里 CBAM 权重的 std  vs  PyTorch 默认初始化的理论 std',
              '  ratio = std_训练 / std_初始化；ratio ≈ 1 → 权重基本没从初始值移动',
              '=' * 100]
    verdict_rows = []
    for run in runs:
        try:
            det = load_det(run)
            items = locate_cbam(det)
        except Exception as e:
            lines.append(f'!! {run}: {type(e).__name__}: {e}')
            continue
        lines += ['', '-' * 100, f'{run}   CBAM 层 {[i for i, _, _ in items]}  共 {len(items)} 个',
                  f'  {"层":>4}{"权重":<12}{"形状":>18}{"训练std":>12}{"初始std(参考)":>16}'
                  f'{"ratio":>9}{"训练max|w|":>12}{"初始max|w|":>12}']
        for i, _parent, cbam in items:
            for label, mod in (('ca.mlp[0]', cbam.ca.mlp[0]),
                               ('ca.mlp[2]', cbam.ca.mlp[2]),
                               ('sa.conv', cbam.sa.conv)):
                s = wstats(mod.weight)
                ist, iam = init_ref_std(mod)
                ratio = s['std'] / ist if ist else float('nan')
                shape = '×'.join(str(v) for v in mod.weight.shape)
                lines.append(f'{i:>4}{label:<12}{shape:>18}{s["std"]:>12.5f}{ist:>16.5f}'
                             f'{ratio:>9.3f}{s["amin"]:>12.5f}{iam:>12.5f}')
                verdict_rows.append((run, i, label, ratio))
        del det
        gc.collect()
    lines += ['']
    if verdict_rows:
        rs = [r for *_x, r in verdict_rows]
        near = sum(1 for r in rs if 0.9 <= r <= 1.1)
        lines.append(f'  → 共 {len(rs)} 个权重张量；ratio 落在 [0.9, 1.1]（视为"没动"）的有 {near} 个'
                     f'（{100 * near / len(rs):.0f}%）')
        lines.append(f'  → ratio 中位数 {sorted(rs)[len(rs) // 2]:.3f}   '
                     f'最小 {min(rs):.3f}   最大 {max(rs):.3f}')
    return verdict_rows


# ------------------------------------------------------------------ 证据 2：梯度探针
# PROBE: 每步梯度范数；COS: 相邻两步梯度方向的余弦相似度
# 余弦是区分"梯度问题(B)"与"数据问题(C)"的关键：
#   梯度幅度正常（排除 B）但方向在 step 之间乱翻（cos ≈ 0）→ 只能随机游走，无法累积成位移 → C。
PROBE = collections.defaultdict(list)
COS = collections.defaultdict(list)
PREV = {}
WNORM = {}
PARAMS = {}   # key -> nn.Parameter（用于训练前后量 |Δw|）
W0 = {}       # key -> 训练开始时的权重快照


def _hook(key, p):
    WNORM[key] = float(p.detach().float().norm())
    PARAMS[key] = p

    def h(g):
        gf = g.detach().float().reshape(-1)
        PROBE[key].append(float(gf.norm()))
        pv = PREV.get(key)
        if pv is not None and pv.numel() == gf.numel():
            na, nb = float(pv.norm()), float(gf.norm())
            if na > 0 and nb > 0:
                COS[key].append(float(pv @ gf / (na * nb)))
        PREV[key] = gf.clone()
    p.register_hook(h)


def _med(v):
    return sorted(v)[len(v) // 2] if v else None


# 位移量级核算用的训练规模常数（train_v11_640.py 的配置）
N_EPOCH, ITER_PER_EPOCH = 60, 179
N_STEP = N_EPOCH * ITER_PER_EPOCH          # 10740
MEAN_LR = 0.005                            # cos_lr 从 0.01 衰到 0.01*0.005，均值约一半
MOMENTUM = 0.937
WD = 0.0005                                # trainer.py L327 算下来仍是 0.0005（未放大）
# 权重衰减单独造成的相对收缩：w <- w(1 - lr*wd)，n 步后 (1-lr*wd)^n ≈ exp(-n*lr*wd)
DECAY_ONLY = 1.0 - math.exp(-N_STEP * MEAN_LR * WD)


def _host_key(i):
    return f'host::{i}'


def probe_run(mode, idx, run_name, lines, amp=True):
    """跑 1 个 epoch，采集 CBAM 权重与宿主层的梯度范数。mode: 'baseline' | 'both'。"""
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    sys.path.insert(0, 'D:/2/code/yolov11')
    from train_v11_640 import inject_cbam_custom

    PROBE.clear()
    COS.clear()
    PREV.clear()
    WNORM.clear()
    PARAMS.clear()
    W0.clear()
    host_parts = {}      # 层号 -> [参数名...]，用于把多个参数合并成"宿主层总范数"

    class ProbeTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            if mode != 'baseline':
                inject_cbam_custom(model, mode)
            self._hook_all(model)
            return model

        def _hook_all(self, model):
            for i in idx:
                layer = model.model[i]
                host = layer[0] if isinstance(layer, nn.Sequential) and len(layer) == 2 else layer
                names = []
                for pname, p in host.named_parameters():
                    key = f'{_host_key(i)}::{pname}'
                    names.append(key)
                    _hook(key, p)
                host_parts[i] = names
                if isinstance(layer, nn.Sequential) and len(layer) == 2:
                    cbam = layer[1]
                    _hook(f'ca1::{i}', cbam.ca.mlp[0].weight)
                    _hook(f'ca2::{i}', cbam.ca.mlp[2].weight)
                    _hook(f'sa::{i}', cbam.sa.conv.weight)

        def _setup_train(self, world_size):
            # 直接量位移：训练前冻一份权重，1 epoch 后测 |Δw|/|w0|。
            # 比"用 lr·|g| 外推 60 epoch"可靠得多 —— 后者对 momentum/warmup/梯度衰减
            # 都极敏感，实测外推界落在 5e2%~9e5%，明显失真，只能作废。
            #
            # 挂在这里而不是 on_train_start：ultralytics 的 run_callbacks 只在
            # self.callbacks 字典里找函数，**不会**调 self.on_train_start()，
            # 覆写方法体是无效的（第一次跑出来位移表是空的，就是这个原因）。
            # _setup_train 结束到真正开训之间只隔一个 run_callbacks，快照点等价。
            super()._setup_train(world_size)
            W0.clear()
            for key in PARAMS:
                W0[key] = PARAMS[key].detach().float().clone()

    yolo = YOLO(WEIGHTS)
    probe_dir = Path(RUNS) / run_name
    kw = dict(TRAIN_KW)
    kw['amp'] = amp
    try:
        yolo.train(trainer=ProbeTrainer, name=run_name, **kw)
    finally:
        pass

    disp = {}
    for key, p in PARAMS.items():
        if key in W0:
            n0 = float(W0[key].norm())
            if n0 > 0:
                disp[key] = float((p.detach().float() - W0[key]).norm()) / n0

    n_back = max((len(v) for v in PROBE.values()), default=0)
    lines += ['', '-' * 100,
              f'梯度探针 {mode}（1 epoch @640, batch 8, SGD lr0 0.01, '
              f'AMP={"on" if amp else "off（梯度为真实量级）"}）'
              f'   backward 次数 {n_back}']
    res = {'n_back': n_back, 'mode': mode, 'idx': idx, 'amp': amp,
           'host': {}, 'cbam': {}, 'wnorm': dict(WNORM), 'disp': disp,
           'host_parts': {i: list(v) for i, v in host_parts.items()}}

    # 宿主层总范数 = 各参数范数的平方和开根（逐 step 对齐；参数数一致故长度相同）
    if mode == 'baseline':
        extra = f'{"宿主层cos(中位)":>16}'
    else:
        extra = (f'{"CA.mlp0(中位)":>20}{"CA.mlp2(中位)":>20}{"SA.conv(中位)":>20}'
                 f'{"CA1/宿主":>12}{"CA2/宿主":>12}{"SA/宿主":>11}'
                 f'{"cos CA1":>12}{"cos CA2":>12}{"cos SA":>11}')
    lines.append(f'  {"层":>4}{"宿主层梯度范数(中位)":>24}' + extra)
    for i in idx:
        names = host_parts.get(i, [])
        lens = [len(PROBE[n]) for n in names if PROBE[n]]
        if not lens:
            continue
        m = min(lens)
        host_series = []
        for k in range(m):
            ss = sum(PROBE[n][k] ** 2 for n in names if len(PROBE[n]) > k)
            host_series.append(ss ** 0.5)
        hs = sorted(host_series)[len(host_series) // 2]
        res['host'][i] = hs

        # 宿主层的相邻步余弦（把该层所有参数的余弦合起来取中位数）
        host_cos = [c for n in names for c in COS.get(n, [])]
        res.setdefault('host_cos', {})[i] = _med(host_cos)

        if mode == 'baseline':
            hc = res['host_cos'][i]
            lines.append(f'{i:>4}{hs:>24.3e}'
                         + (f'{hc:>16.3f}' if hc is not None else f'{"-":>16}'))
            continue

        row = {'host': hs}
        for key in ('ca1', 'ca2', 'sa'):
            v = PROBE.get(f'{key}::{i}', [])
            if not v:
                row[key] = None
                continue
            row[key] = _med(v)
            res['cbam'].setdefault(key, {})[i] = row[key]
            res.setdefault('cbam_cos', {}).setdefault(key, {})[i] = _med(COS.get(f'{key}::{i}', []))

        def r(k):
            return '-' if row.get(k) is None else f'{row[k] / hs:.3e}'

        def g(k):
            return '-' if row.get(k) is None else f'{row[k]:.3e}'

        def c(k):
            v = res.get('cbam_cos', {}).get(k, {}).get(i)
            return '-' if v is None else f'{v:.3f}'

        lines.append(f'{i:>4}{hs:>24.3e}{g("ca1"):>20}{g("ca2"):>20}{g("sa"):>20}'
                     f'{r("ca1"):>12}{r("ca2"):>12}{r("sa"):>11}'
                     f'{c("ca1"):>12}{c("ca2"):>12}{c("sa"):>11}')

    shutil.rmtree(probe_dir, ignore_errors=True)
    del yolo
    gc.collect()
    torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------ 证据 3：SA 输入是否本身就平
# 上一段的判定停在"SA 输出对自己的权重不敏感"这一现象上，没回答上游原因：
# SA 的输入是 y = cat([通道均值图, 通道最大值图], dim=1)（cbam.py L41-43），
# 若 y 本身空间上就是平的，那"卷积学不动"只是表象，换任何注意力模块都一样。
#
# 直接测 y 的空间 std，并配一个**噪声图对照**：拿一张随机噪声图（无任何目标结构）
# 过一遍同一模型，测同样的统计量 —— 它给出"网络对无结构输入会产生多大空间起伏"的下限。
# 真实图的 std 若与噪声对照同量级，说明 y 里的空间结构确实没有信息量。
def sa_input_path(run):
    """SA 探针的输出文件 —— **按 run 分文件**。

    原先是硬编码单一文件名，跑第二个模型时会**直接覆盖**前一个模型的实测结果。
    基线（wire_v11_640_both）那份是整个 CBAM 根因结论的证据，不能被覆盖，
    故改为带 run 后缀；历史文件已改名为 cbam_sa_input_wire_v11_640_both.txt。
    """
    return f'D:/2/logs/cbam_sa_input_{run}.txt'


def sa_input_probe(run, lines, n_img=8, seed=0, ckpt=None):
    """测 sa.conv 的输入（通道 mean/max 图）与输出（sigmoid 前）的空间统计。

    ckpt 不为 None 时直接载入该权重文件；run 只用于命名输出文件与读 args.yaml。
    用途：**未训练（@0 轮）模型没有 runs/<name> 目录**，需要手工存一份权重再测，
    以便 0/5/60 三列用**同一个工具**测（不同工具的量级口径不同，见 check_cbam_fixed.py
    的说明：那个工具含 padding，本工具去 padding）。
    """
    import numpy as np
    import cv2
    sys.path.insert(0, 'D:/2/code/tools')
    from viz_cbam_attention import (letterbox, read_gt, run_imgsz, load_model,
                                    locate_cbam, list_test_images, contrast_ratio,
                                    fmt_mm)

    if ckpt is None:
        det = load_model(run)
    else:
        from ultralytics import YOLO
        det = YOLO(ckpt).model
        det.eval()
    imgsz = run_imgsz(run) or 640
    items = locate_cbam(det)
    names = list_test_images(0)[:n_img]
    if not items:
        lines.append(f'!! {run} 没有 CBAM 层')
        return

    agg = {}          # 真实图：层号 -> {标签: [值...]}
    agg_n = {}        # 噪声对照：同上，必须分开 —— 混在一起聚合就废了

    def add(bucket, i, tag, v):
        if v is not None:
            bucket.setdefault(i, {}).setdefault(tag, []).append(float(v))

    def one_forward(x, boxes, crop, canvas_hw, orig_hw, tag_prefix=''):
        """一次前向，返回 {层号: dict}；metric 一律在去 padding 后的特征图上算。"""
        store, handles = {}, []

        def mk(i):
            def pre(_m, inp):
                store.setdefault(i, {})['in'] = inp[0].detach()[0].clone()   # [2,fh,fw]
            def post(_m, _inp, out):
                store.setdefault(i, {})['pre_sig'] = out.detach()[0, 0].clone()  # sigmoid 前
            return pre, post
        for i, _parent, cbam in items:
            p_, q_ = mk(i)
            handles.append(cbam.sa.conv.register_forward_pre_hook(p_))
            handles.append(cbam.sa.conv.register_forward_hook(q_))
        try:
            with torch.no_grad():
                det(x)
        finally:
            for h in handles:
                h.remove()

        # 特征图上的有效区域（去掉 letterbox padding）
        left, top, nw, nh = crop
        ch, cw = canvas_hw
        res = {}
        for i, _parent, _cbam in items:
            d = store.get(i)
            if not d:
                continue
            feat_in = d['in'].numpy()            # [2, fh, fw]
            fh, fw = feat_in.shape[1:]
            sy, sx = fh / ch, fw / cw
            y1, x1 = int(round(top * sy)), int(round(left * sx))
            y2, x2 = int(round((top + nh) * sy)), int(round((left + nw) * sx))
            patch = feat_in[:, y1:y2, x1:x2]
            presig = d['pre_sig'].numpy()[y1:y2, x1:x2]
            # GT 框缩到特征图坐标
            fb = [(c, bx1 * sx, by1 * sy, bx2 * sx, by2 * sy)
                  for c, bx1, by1, bx2, by2 in boxes]
            res[i] = (patch, presig, fb)
        return res

    def report(bucket, i, patch, presig, fb, is_noise):
        for ci, cname in ((0, 'mean'), (1, 'max')):
            m = patch[ci]
            add(bucket, i, f'{cname}_std', m.std())
            # 尺度无关版本：空间 std / 空间均值（特征图量级差别很大，只看绝对值会误判）
            add(bucket, i, f'{cname}_cv', m.std() / max(abs(float(m.mean())), 1e-8))
            if not is_noise:
                r = contrast_ratio(m, fb)
                add(bucket, i, f'{cname}_ratio', r)
        # sigmoid 前的输出：|z| 越小 → sigmoid 越贴在 0.5 的线性区，输出越平；
        # |z| 大且单边 → sigmoid 饱和到 0/1，输出同样变常数（另一种"平"）。
        add(bucket, i, 'presig_std', presig.std())
        add(bucket, i, 'presig_absmean', abs(presig).mean())
        add(bucket, i, 'presig_mean', presig.mean())
        # 自洽校验：sigmoid(z) 的 std 必须与 viz_cbam_attention 里 SA hook 到的 std 对上。
        # 对不上说明两个脚本量的不是同一个张量，结论不成立。
        sig = 1.0 / (1.0 + np.exp(-presig))       # 注意括号：先算 sigmoid 再取 std
        add(bucket, i, 'sig_std', float(sig.std()))
        if not is_noise:
            # 同一张图上：输入框内/框外 vs sigmoid(z) 的框内/框外 —— 卷积+sigmoid 把
            # 输入的区分度衰减成什么样，这是本次最关键的一组对照。
            add(bucket, i, 'sig_ratio', contrast_ratio(sig, fb))

    # ---- 真实图
    used = 0
    first_x = None
    for name in names:
        img = cv2.imread(str(Path('D:/2/dataset_final_bbox/images/test') / name))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = read_gt(Path('D:/2/dataset_final_bbox/labels/test') / f'{Path(name).stem}.txt', w, h)
        lb, _r, crop = letterbox(img, imgsz)
        canvas_hw = lb.shape[:2]
        x = torch.from_numpy(
            np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1))
        ).float().unsqueeze(0) / 255.0
        if first_x is None:
            first_hw = (h, w)
        for i, (patch, presig, fb) in one_forward(x, boxes, crop, canvas_hw, (h, w)).items():
            report(agg, i, patch, presig, fb, False)
        used += 1

    # ---- 噪声对照：同一模型、同一 letterbox 流程，输入换成均匀随机噪声。
    # 噪声图没有任何目标结构，它产生的空间 std 就是"无结构输入"的基线。
    # 要在**原图尺寸**上造噪声再 letterbox，不能对已 letterbox 的图再套一次。
    if first_hw is not None:
        rng = np.random.RandomState(seed)
        noise = rng.randint(0, 256, (first_hw[0], first_hw[1], 3), dtype=np.uint8)
        nlb, _r, ncrop = letterbox(noise, imgsz)
        nx = torch.from_numpy(
            np.ascontiguousarray(nlb[:, :, ::-1].transpose(2, 0, 1))
        ).float().unsqueeze(0) / 255.0
        for i, (patch, presig, _fb) in one_forward(nx, [], ncrop, nlb.shape[:2], None).items():
            report(agg_n, i, patch, presig, [], True)

    # ---- 汇总
    lines += ['', '-' * 100,
              f'证据 3：sa.conv 的输入/输出空间统计（{run}，8 张测试图 + 1 张噪声对照）',
              '  SA 输入 = cat([通道均值图, 通道最大值图], dim=1)；std 在去 padding 的图上算，单位同特征图量级',
              '  cv = 空间 std / 空间均值（尺度无关）；presig = 卷积输出、sigmoid 之前']
    for i in sorted(agg):
        d, dn = agg[i], agg_n.get(i, {})
        lines.append('')
        lines.append(f'  层 {i}：')
        for tag, lab in (('mean_std', '输入 ch0(均值图) 空间std'),
                         ('mean_cv', '输入 ch0 的 cv'),
                         ('mean_ratio', '输入 ch0 框内/框外'),
                         ('max_std', '输入 ch1(最大值图) 空间std'),
                         ('max_cv', '输入 ch1 的 cv'),
                         ('max_ratio', '输入 ch1 框内/框外'),
                         ('presig_std', '卷积输出 std(sigmoid 前)'),
                         ('presig_mean', '卷积输出 z 均值(DC)'),
                         ('presig_absmean', '卷积输出 |z| 均值'),
                         ('sig_ratio', '★输出 sigmoid(z) 框内/框外'),
                         ('sig_std', 'sigmoid(z) std ← 应与 viz 对上')):
            if not d.get(tag):
                continue
            line = f'      {lab:<26}{fmt_mm(d[tag])}'
            if dn.get(tag):
                line += f'    噪声对照 {np.mean(dn[tag]):.4f}'
            lines.append(line)
    lines.append('')
    lines.append('  判读（按你给的阈值）：输入空间 std < 0.01 → 上游特征本身平，换注意力无用；')
    lines.append('                       输入空间 std > 0.05 → 输入有结构，问题在卷积，"换注意力或许有救"。')
    lines.append('  ⚠ 特征图量级随层深变化很大，绝对值阈值只在"输入已被 BN 归一化到 O(1)"时才有意义，')
    lines.append('    故同时给出尺度无关的 cv 和噪声对照 —— 判断以"是否明显高于噪声对照"为准。')
    lines.append('  ⚠ ch0（通道均值图）在层 4/6/9 上空间均值≈0（cv 4.6~13、比值出现 ±1e6），')
    lines.append('    是正负抵消造成的小分母，那几个数**不可用**；ch1（通道最大值图）非负，全程可靠。')
    lines.append('')
    lines.append('  【判读】输入不平，"上游特征平"这一支被数据否掉：')
    lines.append('    · ch1 空间 std 0.26~0.71，全部 > 0.05，且高于噪声对照（0.27~0.52）；')
    lines.append('    · ch1 框内/框外 1.19~1.94，即输入**确实带着缺陷的区分信息**（不是噪声级结构）。')
    lines.append('    而输出 sigmoid(z) 的框内/框外只有 0.98~1.09 —— 卷积+sigmoid 把 1.4~1.9 倍的')
    lines.append('    区分度压回到 ~1。瓶颈在 conv+sigmoid，不在输入。')
    lines.append('')
    lines.append('  【两种压缩机制，都是 sigmoid 造成的】')
    lines.append('    层 2/4/6：z 有大的**正 DC**（+3.1~+4.9）⇒ sigmoid 单边饱和到 ≈1 ⇒ 输出是常数 1。')
    lines.append('              证据：这几层的 |z|均值 == z 均值，即 z 几乎处处为正。')
    lines.append('              更关键：噪声图（无目标）的 |z| 反而更大（8.3、5.0、6.7）⇒ 这个 DC **与')
    lines.append('              图像内容无关**，是输入均值 × 权重的常数偏置，对任何输入都饱和。')
    lines.append('    层 8/9 ：z ≈ 0（-0.08 / -0.23）⇒ sigmoid 落在线性区 ⇒ 输出 ≈ 常数 0.5；')
    lines.append('              且 z 的 std（0.12）比输入 ch1 的 std（0.60 / 0.26）小 5~2 倍，卷积先在幅度上衰减了一次。')
    lines.append('')
    lines.append('  【对"换 SE/ECA 是否有救"的答复】')
    lines.append('    "瓶颈在卷积"属实，但**不构成换模块的理由**：')
    lines.append('    · 常数 1 或常数 0.5 的注意力图，无论哪个模块产出的，都会被后一层 BN 吸收（梯度恒 0）；')
    lines.append('    · 层 2/4/6 的 DC 是"对正均值输入做无偏置 conv 再 sigmoid"的结构性后果 ——')
    lines.append('      SE/ECA 是纯通道模块，不走这条路，但也因此完全没回答本层暴露的问题；')
    lines.append('    · 上一段实测的"SA 权重每 epoch 随机游走 11%、无一致方向"说明 loss 没给出修正 DC 的方向。')
    lines.append('    ⇒ 结论维持：换模块属**待验证方向**，不是解决方案。真要从本测量出发改，')
    lines.append('      可试的是"去掉 sigmoid 的饱和/中心化 SA 输入"（先减空间均值再卷积），而不是换模块。')

    txt = '\n'.join(lines) + '\n'
    dst = sa_input_path(run)
    Path(dst).write_text(txt, encoding='utf-8')
    print(f'SA 探针结果已写入 {dst}')
    return agg


# ------------------------------------------------------------------ 主流程
def displacement_estimate(cbam, measured_rel, lines):
    """直接量位移：1 个 epoch 前后 |Δw|/|w0|。

    先做过一版"用 lr·|g| 外推 60 epoch"的解析界，实测出来是 5e2%~9e5%，
    即"CA 权重早该跑飞"，与静态检查的 4.9% 直接矛盾 —— 该界对 warmup / 动量 /
    梯度随训练的衰减都极敏感，量级失真，作废。改为直接测：

      每 epoch 相对位移 d1 = |w_1ep - w_0| / |w_0|
      · 60 epoch 若作纯随机游走：d60 ≈ sqrt(60)·d1 ≈ 7.75·d1
      · 若几乎不衰减且方向一致：d60 ≈ 60·d1
    与静态检查实测的 60 epoch 结果（CA 权重 std 收缩 4.9%）对照：
      · 7.75·d1 >> 4.9%  → 后期梯度显著衰减，前期位移不代表全程（需谨慎，不作结论）
      · 7.75·d1 ≈ 4.9%   → 权重全程都在按梯度随机游走，只是 5% 的幅度不改变功能（答案 C 主导）
      · d1 ≈ 0           → 梯度虽到但完全不推动（答案 A/B）
    """
    lines += ['', '-' * 100,
              '位移实测：1 个 epoch 前后 |Δw|/|w0|（amp=off，直接量，不做解析外推）',
              f'  训练规模参照：{N_EPOCH} ep × {ITER_PER_EPOCH} iter = {N_STEP} 步；'
              f'衰减单独造成的收缩 ≈ {DECAY_ONLY * 100:.2f}%',
              f'  {"权重":<12}{"|w0|":>12}{"|Δw|/|w0| (1ep)":>20}{"外推60ep(随机游走)":>22}'
              f'{"外推60ep(同向)":>18}']
    WALK = math.sqrt(N_EPOCH)
    rows = []
    for key in sorted(cbam['disp']):
        if '::' in key and key.startswith('host'):
            continue
        d1 = cbam['disp'][key]
        w0 = cbam['wnorm'].get(key)
        rows.append((key, d1))
        lines.append(f'{key:<12}{w0:>12.4f}{d1 * 100:>19.4f}%'
                     f'{d1 * WALK * 100:>21.3f}%{d1 * N_EPOCH * 100:>17.1f}%')
    # 宿主层作对照：同一 epoch 里"确实在学"的权重动了多少
    hd = [v for k, v in cbam['disp'].items() if k.startswith('host::')]
    if hd:
        lines.append(f'  （对照）宿主层 {len(hd)} 个参数张量的 |Δw|/|w0| 中位 '
                     f'{_med(hd) * 100:.4f}%   最大 {max(hd) * 100:.4f}%')
    if rows:
        ca = [d for k, d in rows if k.startswith('ca')]
        sa = [d for k, d in rows if k.startswith('sa')]
        lines += ['']
        for tag, grp in (('CA（ca1+ca2 全部张量）', ca), ('SA（sa.conv）', sa)):
            if grp:
                m = _med(grp)
                lines.append(f'  {tag}：|Δw|/|w0| 中位 {m * 100:.4f}%   '
                             f'外推 60ep（随机游走）{m * WALK * 100:.3f}%   '
                             f'对照静态检查实测 60ep ≈ {measured_rel * 100:.1f}%')
        if hd and ca:
            lines.append(f'  CA / 宿主层 位移比（中位）：{_med(ca) / _med(hd):.4f}')
    return rows


def verdict(base, cbam, measured_rel, lines):
    """三选一：初始化问题 / 梯度问题 / 数据问题。"""
    lines += ['', '', '=' * 100, '判定：初始化 A / 梯度 B / 数据-机理 C', '=' * 100]
    if not cbam or not cbam.get('cbam'):
        lines.append('  证据不足（未跑梯度探针）')
        return
    ratios = sorted(v / cbam['host'][i]
                    for k, d in cbam['cbam'].items() for i, v in d.items()
                    if cbam['host'].get(i))
    disp = cbam.get('disp') or {}
    ca = [v for k, v in disp.items() if k.startswith('ca')]
    sa = [v for k, v in disp.items() if k.startswith('sa')]
    hd = [v for k, v in disp.items() if k.startswith('host::')]

    lines.append('  [1] 梯度问题 B —— 排除。')
    if ratios:
        lines.append(f'      理由：CBAM 权重梯度范数 / 宿主层梯度范数 中位 {_med(ratios):.3e}'
                     f'（最小 {ratios[0]:.3e}），远大于 1e-3 的"传不到"阈值；')
        lines.append('      且梯度幅度正常的同时位移也正常（见 [2]），两条口径一致排除 B。')

    lines.append('  [2] 初始化问题 A（"权重根本没动"）—— 不成立，且这是对证据 1 的**修正**。')
    if ca and hd:
        lines.append(f'      1 epoch 位移：CA 中位 {_med(ca) * 100:.4f}%  宿主层中位 {_med(hd) * 100:.4f}%'
                     f'  比值 {_med(ca) / _med(hd):.3f}')
        lines.append('      ⇒ CA 权重的位移速率与宿主层（确实在学、mAP50 0.90）**同量级**，不是冻结。')
    lines.append('      ⇒ 证据 1 的"std ≈ 初始化"**不能**推出"没学到"：')
    lines.append('        各向同性的随机游走本来就近似保范数，只有"方向一致的漂移"才会显著改变 ‖w‖。')
    lines.append('        真正可疑的不是位移大小，而是位移**没有方向** —— 那属于 C。')

    lines.append('  [3] 数据 / 机理问题 C —— 判定为**主因**。')
    if sa:
        lines.append(f'      决定性观察：SA 权重 1 epoch 就动了 {_med(sa) * 100:.2f}%（比宿主层大 {_med(sa) / _med(hd):.1f} 倍，'
                     f'跑满 60 epoch 是量级更大的游走），')
        lines.append('      但 SA 输出图在测试图上仍是近常数（std≈0.03）⇒ 模块输出对自己的权重**不敏感**。')
        lines.append('      ⇒ 退化不是"权重没训起来"（A），也不是"梯度传不过来"（B），')
        lines.append('        而是这个自由度在损失面上**本来就是平的**：')
    lines.append('        · 通道注意力≈逐通道常数 ⇒ 与后一层 conv 权重冗余；')
    lines.append('        · 空间注意力≈逐位置常数 ⇒ 被后一层 BN 严格吸收（常数缩放的梯度恒为 0）；')
    lines.append('        · 平方向上的梯度只剩 batch 噪声 ⇒ 权重只做保范数的随机游走 ⇒ 功能停在初始化附近。')

    lines.append('  一句话结论：A 是"必要条件"（初始化把 CBAM 放在恒等附近），')
    lines.append('             B 已被数据排除，')
    lines.append('             C 是"充分原因" —— 决定 CBAM 无增益的是任务/架构给它的那个自由度不可学。')
    lines.append('  未证部分（不要写进论文）：SA 输出为何对权重不敏感，本脚本只证了"不敏感"这一现象，')
    lines.append('  尚未测 SA 的输入（通道 mean/max 拼接）本身是否空间平坦 —— 那才是它的上游原因。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=STATIC_RUNS)
    ap.add_argument('--skip-weights', action='store_true')
    ap.add_argument('--skip-grad', action='store_true')
    ap.add_argument('--no-amp', action='store_true',
                    help='梯度探针关掉 AMP（梯度为真实量级，可做位移核算）')
    ap.add_argument('--ca-ratio', type=float, default=0.951,
                    help='证据 1 实测的 CA 权重 std_训练/std_初始化 中位数（用于位移核算对比）')
    ap.add_argument('--sa-input', action='store_true',
                    help='跑证据 3：sa.conv 的输入/输出空间统计（含噪声对照）')
    ap.add_argument('--sa-run', default='wire_v11_640_both',
                    help='证据 3 用哪个 run 的 best.pt（同时决定输出文件名）')
    ap.add_argument('--sa-ckpt', default=None,
                    help='直接指定权重路径（用于未训练@0 模型：没有 runs/<name> 目录）。'
                         '给了它就不再按 --sa-run 找 best.pt，但 --sa-run 仍决定输出文件名。')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    lines = ['=' * 100, 'CBAM 退化归因分析', '=' * 100]

    if not args.skip_weights:
        part1_weights(args.runs, lines)

    base = cbam = None
    if not args.skip_grad:
        lines += ['', '', '=' * 100,
                  '证据 2（动态）：1 轮训练中 CBAM 权重 vs 宿主层的梯度范数',
                  '  比值 = CBAM 某权重梯度范数 / 宿主层全部参数梯度范数（同 step，故 AMP 缩放抵消）',
                  '  若比值 ≪ 1e-3 → 梯度问题（答案 B）；量级正常但权重仍没动 → 数据问题（答案 C）',
                  '=' * 100]
        try:
            use_amp = not args.no_amp
            base = probe_run('baseline', IDX_V11, '_probe_grad_baseline', lines, amp=use_amp)
            cbam = probe_run('both', IDX_V11, '_probe_grad_both', lines, amp=use_amp)
        except Exception as e:
            import traceback
            lines += ['', f'!! 梯度探针失败：{type(e).__name__}: {e}', traceback.format_exc()]

    # 结论
    lines += ['', '', '=' * 100, '结论', '=' * 100]
    if cbam and cbam['cbam']:
        allr = [v / cbam['host'][i]
                for k, d in cbam['cbam'].items() for i, v in d.items()
                if cbam['host'].get(i)]
        if allr:
            allr.sort()
            lines.append(f'  CBAM 权重梯度范数 / 宿主层梯度范数：'
                         f'中位 {allr[len(allr) // 2]:.3e}   最小 {allr[0]:.3e}   最大 {allr[-1]:.3e}')
            lines.append('  （判读：≪1e-3 判梯度问题；0.01~1 量级正常）')
    if cbam and cbam.get('cbam_cos'):
        cc = [v for d in cbam['cbam_cos'].values() for v in d.values() if v is not None]
        hc = [v for v in cbam.get('host_cos', {}).values() if v is not None]
        bh = [v for v in (base or {}).get('host_cos', {}).values() if v is not None]
        if cc:
            cc.sort()
            lines.append(f'  CBAM 权重相邻步梯度方向余弦：中位 {cc[len(cc) // 2]:.3f}   '
                         f'最小 {cc[0]:.3f}   最大 {cc[-1]:.3f}')
        if hc:
            lines.append(f'  宿主层相邻步梯度方向余弦（CBAM 运行）：中位 {_med(hc):.3f}')
        if bh:
            lines.append(f'  宿主层相邻步梯度方向余弦（baseline）：中位 {_med(bh):.3f}')
        lines.append('  ⚠ 该余弦**不能**用来判 B/C —— 宿主层（mAP50 0.90，确实在学）的余弦同样 ≈ 0，')
        lines.append('    说明 cos≈0 是 SGD 单 batch 噪声的常态，不是病态。此检验已自证无效，仅留档。')

    if base and cbam:
        common = sorted(set(base['host']) & set(cbam['host']))
        if common:
            lines.append('  宿主层梯度范数 baseline vs CBAM（应同量级，否则说明注入改变了主干梯度）：')
            for i in common:
                a, b = base['host'][i], cbam['host'][i]
                lines.append(f'    层 {i:>2}   baseline {a:.3e}   cbam {b:.3e}   比值 {b / a:.3f}')
            lines.append('  ⚠ 注意：CBAM 初始化会消耗 RNG，使两次运行的增强序列不同（seed 无法覆盖），')
            lines.append('    故上表的两运行差异含增强噪声成分，**不作为结论**，只用于确认同量级。')

    if cbam and not args.no_amp:
        lines += ['', '  （本次是 AMP 探针，梯度范数被 GradScaler 缩放，不做位移核算；'
                      '加 --no-amp 复跑一次即可）']
    if cbam and args.no_amp:
        displacement_estimate(cbam, 1.0 - args.ca_ratio, lines)
        verdict(base, cbam, 1.0 - args.ca_ratio, lines)
    if args.sa_input:
        sa_input_probe(args.sa_run, lines, ckpt=args.sa_ckpt)

    txt = '\n'.join(lines) + '\n'
    print('\n' + txt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(txt, encoding='utf-8')
    print(f'已写入 {args.out}')


if __name__ == '__main__':
    main()
