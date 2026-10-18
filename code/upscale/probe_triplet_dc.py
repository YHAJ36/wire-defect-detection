# -*- coding: utf-8 -*-
"""方向6 取证：Triplet Attention 会不会走 CBAM 那条「DC 饱和」路径？

背景：CBAM 在本数据集被定量证明退化（浅层 SA 输出近常数图，框内外比 ≈1）。
失效机理（code/yolov11/cbam_fixed.py 头部 + code/tools/analyze_cbam_degradation.py）：
    正均值输入 × 无偏置 7×7 conv ⇒ sigmoid 前 z 出现 +3~+5 的大正 DC ⇒ 单边饱和。

三问（用户指定）：
  ① Triplet 空间分支的输入（Z-pool 输出）是否非负？
  ② sigmoid 前是否有 conv 的 bias？
  ③ 是否走与 CBAM 相同的 DC 饱和路径？

⚠ 第一版探针的教训（已修正，留档）：只在 eval() 下测。初始化时 BN 的
  running_mean=0 / running_var=1 ⇒ **BN 是恒等映射**，DC 原样透传，
  Triplet 测出 +1.463，比 CBAM 的 +0.603 还饱和 —— 差点据此判 Triplet 同样该死。
  错在哪：BN 消 DC 不是"结构上天生具备"的，而是"训练中统计量累积之后才生效"的。
  所以必须分**三种状态**测，且**多种子**取分布（单次随机初始化方差极大）。

做法：Triplet 的 AttentionGate 按官方源码逐行复刻（内联于下，注明出处），
与项目里真实的 CBAM SpatialAttention 同输入对拍，读 sigmoid 之前的 z。

三种状态：
  (A) eval @ init      —— BN 用 running(=0,1)，恒等；暴露 conv 的"裸" DC
  (B) train            —— BN 用 batch 统计；DC 被当场扣掉
  (C) eval @ warmup    —— 先 train 模式喂 N 个 batch 攒 running stats，再 eval；模拟训练后推理
CBAM 无 BN，无此三态（三态恒等）。

输入：模拟 post-SiLU 特征图 —— 非负、有正均值、幅度贴近真实（这是 CBAM 失效的输入条件）。

源：LandskapeAI/triplet-attention — triplet_attention.py（Rotate to Attend, WACV 2021）
输出：stdout + D:/2/exp_upscale/logs/probe_triplet_dc.txt
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, 'D:/2/code/yolov11')
from cbam import SpatialAttention as CbamSA  # noqa: E402  项目里真实的 CBAM 空间注意力

SEEDS = [0, 1, 2, 3, 4]
WARM_BATCHES = 200      # BN momentum=0.01 ⇒ 200 步约收敛到 87%


# ---------------- Triplet Attention：逐行复刻官方源码 ----------------
class BasicConv(nn.Module):
    """官方 BasicConv（bias 默认 False；bn 默认 True）。"""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.out_channels = out_planes
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
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1,
                              padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        zc = self.compress(x)
        x_out = self.conv(zc)          # conv(bias=False) → BatchNorm2d → (无 ReLU)
        return x * torch.sigmoid(x_out)


def z_pre_sigmoid(mod, x, which):
    """返回 sigmoid 之前的 z（不触发 sigmoid 的 in-place 副作用）。"""
    if which == 'cbam':
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return mod.conv(torch.cat([avg, mx], dim=1))
    zc = mod.compress(x)
    return mod.conv.bn(mod.conv.conv(zc))


def stats(z, x):
    with torch.no_grad():
        out = torch.sigmoid(z)
        B, C, H, W = out.shape
        h0, h1, w0, w1 = H // 4, H * 3 // 4, W // 4, W * 3 // 4
        inner = out[..., h0:h1, w0:w1].mean().item()
        mask = torch.ones_like(out, dtype=torch.bool)
        mask[..., h0:h1, w0:w1] = False
        outer = out[mask].mean().item()
    return z.mean().item(), z.abs().mean().item(), out.std().item(), inner / max(outer, 1e-9)


def batch_stream(C, H, W, n, sd=1234, vary=True):
    """一串正特征图 batch，让 BN 的 running stats 有东西可攒。

    vary=True  —— 各 batch 幅度浮动 1~3 倍（模拟训练中特征分布漂移）
    vary=False —— 各 batch 同分布（模拟某一层在稳态下的分布）
    对照这两者，是为了判断状态 (C) 里那点残余 DC 到底是不是"合成数据流的锅"。
    """
    g = torch.Generator().manual_seed(sd)
    out = []
    for i in range(n):
        m = (1.0 + 2.0 * torch.rand(1, generator=g).item()) if vary else 1.0
        out.append(torch.rand(2, C, H, W, generator=g) * 3.0 * m)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    say('=' * 100)
    say('方向6 取证：Triplet Attention 是否走 CBAM 的「DC 饱和」路径')
    say(f'（多种子 {len(SEEDS)}，三状态；BN warmup {WARM_BATCHES} batch）')
    say('=' * 100)

    for (C, H, W) in [(256, 80, 80), (1024, 5, 5), (1024, 20, 20)]:
        stream = batch_stream(C, H, W, WARM_BATCHES)
        x = stream[0]
        say(f'\n{"-"*100}')
        say(f'输入 x: {tuple(x.shape)}  mean={x.mean():.3f}（非负、有正均值，模拟 post-SiLU 特征图）')
        zp = ZPool()(x)
        say(f'  ① Z-pool 输出: {tuple(zp.shape)}  min={zp.min():.3f} mean={zp.mean():.3f} '
            f'⇒ {"非负（与 CBAM 的 [avg,max] 同条件）" if zp.min() >= 0 else "含负值"}')

        # ---- CBAM：无 BN，三态恒等，只测 init ----
        rows = []
        for s in SEEDS:
            torch.manual_seed(s)
            m = CbamSA().eval()
            with torch.no_grad():
                z = z_pre_sigmoid(m, x, 'cbam')
            rows.append(stats(z, x))
        say(f'\n  CBAM-SpatialAttention（无 BN，与训练状态无关）')
        say(f'    {_fmt(rows)}')

        # ---- Triplet：三态 ----
        for label, mode, use_vary in (
                ('(A) eval@init  [BN=running(0,1)=恒等]', 'init', None),
                ('(B) train        [BN=batch 统计]', 'train', None),
                (f'(C) eval@warm{WARM_BATCHES}   [BN=running 已成形]', 'warm', True),
                (f'(C2) 同上，但 warmup 用**同分布**流 [对照]', 'warm', False)):
            rows = []
            for s in SEEDS:
                torch.manual_seed(s)
                m = AttentionGate()
                with torch.no_grad():
                    if mode == 'init':
                        m.eval()
                    elif mode == 'train':
                        m.train()
                    else:
                        m.train()
                        for b in (batch_stream(C, H, W, WARM_BATCHES, vary=use_vary) if use_vary is not None
                                  else stream):
                            m(b)
                        m.eval()
                    z = z_pre_sigmoid(m, x, 'triplet')
                rows.append(stats(z, x))
            say(f'\n  Triplet-AttentionGate {label}')
            say(f'    {_fmt(rows)}')
            if mode == 'warm':
                bn = m.conv.bn
                say(f'    BN running_mean={bn.running_mean.item():+.4f} '
                    f'running_var={bn.running_var.item():.4f}')

    say('\n' + '=' * 100)
    say('② sigmoid 前有无 conv bias：')
    say('   CBAM 原版 : Conv2d(2,1,7,padding=3, bias=False)                       → 无 bias')
    say('   Triplet   : BasicConv(..., relu=False) → Conv2d(2,1,7,bias=False)      → 同样无 bias')
    say('               **但在 conv 与 sigmoid 之间多了一层 BatchNorm2d(1, affine=True, momentum=0.01)**')
    say('\n③ 判定（看上面三态里 DC 列的变化）：')
    say('   CBAM : DC 只由 conv 权重决定，**任何状态下都不会被扣掉** ⇒ 饱和是持久的。')
    say('   Triplet: (A) 初始化时 BN 恒等，DC 与 CBAM 同级甚至更大（**这一态不能作为判据**）；')
    say('            (B) train 下 BN 用 batch 统计，DC 被当场减掉；')
    say('            (C) 攒够 running stats 后，推理时 DC 同样接近 0。')
    say('   ⇒ 结论：Triplet **在结构上具备**切断 DC 饱和的机制，但机制靠 BN 统计量生效，')
    say('      训练初期（前若干 batch）会经历一小段与 CBAM 同样"裸 DC"的窗口。')

    OUT = Path('D:/2/exp_upscale/logs/probe_triplet_dc.txt')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


def _fmt(rows):
    import statistics as st
    cols = ['DC(均值)', '|z|均值', '输出std', '框内外比']
    fields = list(zip(*rows))
    out = []
    for name, col in zip(cols, fields):
        out.append(f'{name}={st.mean(col):+.4f}±{st.pstdev(col):.4f}')
    return '  '.join(out)


if __name__ == '__main__':
    main()
