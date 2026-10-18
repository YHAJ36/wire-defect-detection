# -*- coding: utf-8 -*-
"""追问3 取证：Triplet 是否走 CBAM 的**第二条**失效路径 —— 幅度衰减（输出≈常数 0.5）。

用户指出：前一轮只排除了「DC 饱和」（第一机制），没验「幅度衰减」（第二机制）。
CBAM 的第二机制证据（`cbam_fixed.py` 头部 + `logs/cbam_degradation.txt`，实测自
**训练后**的 wire_v11_640_both）：

    层 8（SPPF）: z均值 −0.23（未饱和！）  输出 std 0.0307  框内外比 0.981
    层 9（SPPF）: z均值 −0.08（未饱和！）  输出 std 0.0306  框内外比 0.995
    ⇒ sigmoid 落在**线性区**，输出 ≈ 常数 0.5，衰减成"无区分度"的均匀缩放。

判定规则（用户给定）：不走衰减路径 → Triplet 保留；走 → 降级或砍掉。

⚠ 方法学前提（不可省）：CBAM 的 0.03 是在**训练后**权重上测的，Triplet 没有训练后权重，
   直接用随机初始化对拍**不可比**。故本探针跑四组：
     A. cbam_trained  —— 载入 wire_v11_640_both/best.pt，**先复现 0.03/0.98 验证量具**
     B. cbam_init     —— 同骨干（yolo11s.pt）随机初始化 CBAM
     C. cbamfix_init  —— 同骨干 + 修好偏置/中心化的 CBAM（上限参照）
     D. triplet_init  —— 同骨干 + 官方 Triplet Attention
   **只有 B vs D 是严格可比的**（同骨干、同图、同种子、同层）；A 是量具校验，C 是上界参照。

测的层与 CBAM 完全一致：[2, 4, 6, 8, 9]（= 训练脚本的 mode='both'）。
量具全部复用 `code/tools/viz_cbam_attention.py`（letterbox / read_gt / contrast_ratio /
map_to_original），保证与已发表的 CBAM 读数同口径。

Triplet 的"空间可比分支"取 `hw`（直接作用于 x、不做 permute 的那一支），
与 CBAM 的 `sa` 语义对应。注意 Triplet 另有两支 cw/hc 是跨通道/跨高与跨宽的门，
其输出分布本就不该与 sa 直接比 —— 本探针只比 hw，并在结论里注明。

输出：stdout + D:/2/exp_upscale/logs/probe_triplet_vs_cbam.txt
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'D:/2/code/yolov11')
sys.path.insert(0, 'D:/2/code/tools')

from viz_cbam_attention import (  # noqa: E402
    IMAGES, LABELS, letterbox, read_gt, list_test_images, contrast_ratio, map_to_original)

N_IMAGES = 12          # 跨图聚合，单图下不了结论（与 viz 同思路）
SEED = 42
LAYERS = [2, 4, 6, 8, 9]   # 与 train_v11_640_both.py 的 mode='both' 完全一致
IMGSZ = 640


# ============================ Triplet Attention（官方源码复刻 + 取值口） ============
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    """官方 AttentionGate；额外存 last_scale / last_z 供测量（计算与官方逐字一致）。"""

    def __init__(self):
        super().__init__()
        k = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, k, stride=1, padding=(k - 1) // 2, relu=False)
        self.last_scale = None
        self.last_z = None

    def forward(self, x):
        zc = self.compress(x)
        zo = self.conv(zc)                 # conv(bias=False) → BatchNorm2d
        scale = torch.sigmoid(zo)          # 等价官方 sigmoid_（此处不用 in-place）
        self.last_scale = scale.detach()
        self.last_z = zo.detach()
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super().__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out11 = self.cw(x_perm1).permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out21 = self.hc(x_perm2).permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            return 1 / 3 * (x_out + x_out11 + x_out21)
        return 1 / 2 * (x_out11 + x_out21)


# ================================ 建模型 / 注入 ====================================
def get_channels(m):
    for name in ('cv2', 'cv3', 'cv4'):
        conv = getattr(m, name, None)
        if conv is None:
            continue
        if isinstance(conv, nn.Conv2d):
            return conv.out_channels
        inner = getattr(conv, 'conv', None)
        if isinstance(inner, nn.Conv2d):
            return inner.out_channels
    for sub in reversed(list(m.modules())):
        if isinstance(sub, nn.Conv2d):
            return sub.out_channels
    raise ValueError('no conv')


def inject(seq, gate_factory, idx):
    """把 seq[i] 包成 nn.Sequential(seq[i], gate)，与 train_v11_640_both.py 逐字同法。"""
    made = []
    for i in idx:
        gate = gate_factory(get_channels(seq[i]))
        w = nn.Sequential(seq[i], gate)
        w.f, w.i = seq[i].f, seq[i].i
        if hasattr(seq[i], 'type'):
            w.type = seq[i].type
        seq[i] = w
        made.append((i, type(seq[i][0]).__name__, gate))
    return made


def build(kind):
    """返回 (det_model, items)；items = [(层号, 父层类名, 取 scale 的函数)]。"""
    from ultralytics import YOLO

    if kind == 'cbam_trained':
        det = YOLO('D:/2/runs/wire_v11_640_both/weights/best.pt').model
        det.eval()
        items = []
        for i, layer in enumerate(det.model):
            if isinstance(layer, nn.Sequential) and len(layer) == 2 and hasattr(layer[1], 'sa'):
                items.append((i, type(layer[0]).__name__, layer[1].sa, 'cbam'))
        return det, sorted(items)

    det = YOLO('D:/1/models/yolo11s.pt').model       # 预训练骨干 —— 特征分布才真实

    if kind in ('cbam_init', 'cbamfix_init'):
        if kind == 'cbam_init':
            from cbam import CBAM as G
        else:
            from cbam_fixed import CBAM as G
        torch.manual_seed(SEED)
        made = inject(det.model, lambda c: G(c), LAYERS)
        items = [(i, p, g.sa, 'cbam') for i, p, g in made]
    elif kind in ('triplet_init', 'triplet_trainbn'):
        torch.manual_seed(SEED)
        made = inject(det.model, lambda c: TripletAttention(), LAYERS)
        items = [(i, p, g.hw, 'triplet') for i, p, g in made]
    else:
        raise ValueError(kind)

    det.eval()
    if kind == 'triplet_trainbn':
        # 只把 **gate 内部**的 BN 置 train（用 batch 统计），其余全网络保持 eval。
        # 这样隔离出"训练中 BN 生效"这一个变量，不与骨干 BN 行为混淆。
        for _i, _p, g in made:
            for m in g.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    return det, sorted(items)


# ================================== 测量 ==========================================
def measure(kind, img_names):
    det, items = build(kind)
    store = {}
    handles = []
    for i, parent, holder, gk in items:
        def mk(i, gk, holder):
            def hook(_m, _inp, out):
                if gk == 'cbam':
                    store[i] = out.detach()[0, 0].clone()          # sa 直接返回 scale 图
                else:
                    store[i] = holder.last_scale[0, 0].clone()      # triplet 的 hw 分支
                    store.setdefault(f'z_{i}', holder.last_z[0, 0].clone())
            return hook
        handles.append(holder.register_forward_hook(mk(i, gk, holder)))

    agg = {}
    used = 0
    try:
        for name in img_names:
            p = Path(IMAGES) / name
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            boxes = read_gt(Path(LABELS) / f'{p.stem}.txt', w, h)
            lb, _r, crop = letterbox(img, IMGSZ)
            canvas_hw = lb.shape[:2]
            x = torch.from_numpy(
                np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1))
            ).float().unsqueeze(0) / 255.0

            store.clear()
            with torch.no_grad():
                det(x)

            for i, parent, _h, gk in items:
                if i not in store:
                    continue
                sc = store[i].numpy()
                zkey = f'z_{i}'
                sc_orig = map_to_original(sc, canvas_hw, crop, (h, w))
                r = contrast_ratio(sc_orig, boxes)
                a = agg.setdefault(i, {'parent': parent, 'std': [], 'mean': [],
                                       'ratio': [], 'zstd': [], 'zmean': []})
                a['std'].append(float(sc.std()))
                a['mean'].append(float(sc.mean()))
                if r is not None:
                    a['ratio'].append(float(r))
                if zkey in store:
                    z = store[zkey].numpy()
                    a['zstd'].append(float(z.std()))
                    a['zmean'].append(float(z.mean()))
            used += 1
    finally:
        for hd in handles:
            hd.remove()
        del det
        torch.cuda.empty_cache()
    return agg, used


def fmt(vals, nd=4):
    if not vals:
        return '—'
    return f'{np.mean(vals):.{nd}f}[{min(vals):.{nd}f}~{max(vals):.{nd}f}]'


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    names = list_test_images(prefer_class=0)[:N_IMAGES]
    say('=' * 108)
    say('追问3 取证：Triplet 是否走 CBAM 的第二失效路径「幅度衰减（输出≈常数）」')
    say(f'  {len(names)} 张测试图（break 优先，与 viz_cbam_attention 同口径） / imgsz {IMGSZ} / 层 {LAYERS}')
    say('=' * 108)
    say(f'{"配置":<16}{"层":>4}{"父层":>10}{"scale均值":>24}{"scale标准差":>24}{"框内外比":>24}')
    say('-' * 108)

    results = {}
    for kind in ('cbam_trained', 'cbam_init', 'cbamfix_init',
                 'triplet_init', 'triplet_trainbn'):
        try:
            agg, used = measure(kind, names)
        except Exception as e:
            say(f'{kind:<16} 失败: {type(e).__name__}: {str(e)[:60]}')
            continue
        results[kind] = agg
        for i in sorted(agg):
            a = agg[i]
            say(f'{kind:<16}{i:>4}{a["parent"]:>10}'
                f'{fmt(a["mean"]):>24}{fmt(a["std"]):>24}{fmt(a["ratio"]):>24}')
        say('-' * 108)

    say('\n【CBAM 训练后的已知读数（cbam_fixed.py 头部，供校验量具）】')
    say('  层 8: z均值 −0.23  输出std 0.0307  框内外比 0.981')
    say('  层 9: z均值 −0.08  输出std 0.0306  框内外比 0.995')
    say('  ⇒ 若上面 cbam_trained 的层 8/9 对得上，说明本量具可信，triplet_init 的数才可用。')

    say('\n【Triplet 的 sigmoid 之前 z 的统计（init = BN 恒等  vs  trainbn = BN 生效）】')
    for tag in ('triplet_init', 'triplet_trainbn'):
        if tag not in results:
            continue
        say(f'\n  -- {tag} --')
        say(f'  {"层":>4}{"父层":>10}{"z均值(DC)":>24}{"z标准差":>24}{"scale标准差":>24}{"scale均值":>24}')
        for i in sorted(results[tag]):
            a = results[tag][i]
            say(f'{i:>4}{a["parent"]:>10}{fmt(a["zmean"]):>24}{fmt(a["zstd"]):>24}'
                f'{fmt(a["std"]):>24}{fmt(a["mean"]):>24}')

    say('\n' + '=' * 108)
    say('判读要点（写进 WORKLOG 前请核对）：')
    say('  1. cbam_trained 的层 8/9 复现已知读数 ⇒ 量具可信。')
    say('  2. 第二机制 = 输出塌成常数（std 极小），与"饱和"（均值贴 0/1）是两回事；')
    say('     cbam_trained 层 8/9 均值 0.44/0.48（未饱和）而 std 仅 0.032 ⇒ 正是第二机制。')
    say('  3. Triplet 的 BN 在 **train** 下把 z 归一到单位尺度 ⇒ scale 的标准差由 BN 钉住，')
    say('     不继承 conv 的原始幅度 ⇒ 结构上不该出现同样的塌陷。看 triplet_trainbn 的 std 列。')
    say('  4. ⚠ init 行的"框内外比"**不可用于判定机制**：门尚未训练，对 CBAM 与 Triplet')
    say('     都是噪声（cbam_init 也散在 0.77~1.10）。区分度只能训练后测。')

    OUT = Path('D:/2/exp_upscale/logs/probe_triplet_vs_cbam.txt')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
