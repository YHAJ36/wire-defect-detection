# -*- coding: utf-8 -*-
"""显存探测：v11s / v11m 在 896 / 1024 下、不同 batch 的峰值显存。

目的：回答"v11m@896 batch8 在 8GB 卡会不会 OOM""1024 下 batch 能开多少"。
不改任何已封版文件，不写 runs/，只读 yaml 建架构 + 一次 forward/backward。

方法：
  - 用 ultralytics 的 yaml 从零建架构（不下载预训练权重——显存由**架构+尺寸+batch**决定，
    与权重数值无关，故无需联网）。
  - train 模式下 amp autocast 前向 + 反向，读 torch.cuda.max_memory_allocated。
  - 每个组合用独立进程跑（子进程），避免显存碎片跨组合累积。
  - OOM 由子进程捕获并上报，不中断整轮。

用法：
    D:/yolo-env/Scripts/python.exe D:/2/code/upscale/probe_vram.py
输出：stdout + D:/2/exp_upscale/logs/probe_vram.txt
"""
import subprocess
import sys
from pathlib import Path

PY = 'D:/yolo-env/Scripts/python.exe'
OUT = Path('D:/2/exp_upscale/logs/probe_vram.txt')

# (架构 yaml, 简称, 参数量级说明)
ARCHS = [('yolo11s.yaml', 'v11s'), ('yolo11m.yaml', 'v11m')]
IMGSZ = [896, 1024]
BATCHES = [8, 6, 4, 2]

CHILD = r'''
import sys, torch, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
arch, imgsz, batch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
from ultralytics import YOLO
m = YOLO(arch).model.train()
n_par = sum(p.numel() for p in m.parameters())
dev = "cuda:0"
m = m.to(dev)
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
x = torch.randn(batch, 3, imgsz, imgsz, device=dev)
try:
    with torch.autocast("cuda", dtype=torch.float16):
        out = m(x)
        if isinstance(out, (list, tuple)):
            loss = sum(t.sum() for t in out if isinstance(t, torch.Tensor) and t.requires_grad)
        else:
            loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(json.dumps({"ok": True, "peak_mib": round(peak, 1), "params": n_par}))
except torch.cuda.OutOfMemoryError as e:
    print(json.dumps({"ok": False, "err": "OOM", "params": n_par}))
except RuntimeError as e:
    emsg = str(e)[:120]
    print(json.dumps({"ok": False, "err": ("OOM" if "memory" in emsg.lower() else emsg), "params": n_par}))
'''


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    say('=' * 92)
    say('显存探测：forward+backward 峰值（amp fp16，train 模式，RTX 4060 Laptop 8188 MiB）')
    say('=' * 92)
    say(f'{"架构":<8}{"imgsz":>7}{"batch":>7}{"峰值MiB":>12}{"占 8188 的":>12}  结论')
    say('-' * 92)
    tot = 8188
    for arch, short in ARCHS:
        for sz in IMGSZ:
            for b in BATCHES:
                p = subprocess.run([PY, '-c', CHILD, arch, str(sz), str(b)],
                                   capture_output=True, text=True, timeout=900)
                last = [l for l in p.stdout.strip().splitlines() if l.strip().startswith('{')]
                if not last:
                    say(f'{short:<8}{sz:>7}{b:>7}{"—":>12}{"—":>12}  探测失败: '
                        f'{(p.stderr.strip().splitlines() or ["(无输出)"])[-1][:60]}')
                    continue
                import json
                d = json.loads(last[-1])
                if not d['ok']:
                    say(f'{short:<8}{sz:>7}{b:>7}{"—":>12}{"—":>12}  ✗ {d["err"]}')
                    continue
                pk = d['peak_mib']
                pct = pk / tot * 100
                verdict = '安全' if pct < 78 else ('紧绷' if pct < 92 else '危险')
                say(f'{short:<8}{sz:>7}{b:>7}{pk:>12.0f}{pct:>11.1f}%  {verdict}')
    say('-' * 92)
    say('注：峰值=torch.cuda.max_memory_allocated（不含 CUDA context ~0.4G 与缓存碎片）。')
    say('    实跑时 dataloader/mosaic 的 CPU→GPU 张量会再加约 0.2~0.5G；')
    say('    故 >78% 判"紧绷"、>92% 判"危险"，实跑须留余量。')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
