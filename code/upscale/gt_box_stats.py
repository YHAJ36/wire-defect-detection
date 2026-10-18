# -*- coding: utf-8 -*-
"""第 0 步：GT 框尺寸分布 —— 判定 1024 是否需要加 P2(stride=4) 浅层检测分支。

背景：方向2（896→1024）要不要加 P2，取决于目标在特征图上的实际像素尺度。
不加 P2 时最小检测层是 P3(stride=8)。若绝大多数目标在 1024 下远大于 8px，
P3 足够，加 P2 只是白烧算力（P2 特征图是 P3 的 4 倍面积）。

方法：读 dataset_final_bbox/labels/train|val|test 的 YOLO 归一化标注，
还原为像素尺寸（按各 split 图像原始尺寸），再按 896 / 1024 缩放后统计。
纯 CPU 只读，无副作用。

输出：stdout + D:/2/exp_upscale/logs/gt_box_stats.txt
"""
import sys
from pathlib import Path

ROOT = Path('D:/2/dataset_final_bbox')
IMGSZ = [640, 896, 1024]
# P3 = stride 8；"小于一个 P3 格"= min 边 < 8px；常用小目标阈值 32px
EDGES = [4, 8, 16, 32, 64]


def img_size(p: Path):
    """用 PIL 只读文件头拿 (w,h)，避免整图解码。"""
    from PIL import Image
    with Image.open(p) as im:
        return im.size


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    say('=' * 88)
    say('第 0 步：GT 框尺寸分布（判定 1024 是否需要 P2/stride=4）')
    say('=' * 88)
    for split in ('train', 'val', 'test'):
        ldir = ROOT / 'labels' / split
        idir = ROOT / 'images' / split
        if not ldir.exists():
            say(f'{split}: 无标注目录'); continue
        recs = []
        for lf in sorted(ldir.glob('*.txt')):
            img = None
            for ext in ('.jpg', '.png', '.jpeg', '.bmp'):
                cand = idir / (lf.stem + ext)
                if cand.exists():
                    img = cand; break
            if img is None:
                continue
            W, H = img_size(img)
            for ln in lf.read_text(encoding='utf-8').splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                try:
                    w, h = float(p[3]) * W, float(p[4]) * H
                except Exception:
                    continue
                recs.append((w, h, W, H))
        if not recs:
            say(f'{split}: 无有效框'); continue
        say(f'\n--- {split}：{len(recs)} 个框，{len(list(ldir.glob("*.txt")))} 个标注文件')
        for sz in IMGSZ:
            # 缩放：letterbox 近似按长边缩放
            mins, areas = [], []
            for w, h, W, H in recs:
                s = sz / max(W, H)
                mins.append(min(w, h) * s)
                areas.append((w * s) * (h * s))
            mins.sort()
            n = len(mins)
            say(f'  [imgsz={sz}] min 边(px): '
                f'p5={mins[int(n*0.05)]:.1f} p25={mins[int(n*0.25)]:.1f} '
                f'中位={mins[n//2]:.1f} p75={mins[int(n*0.75)]:.1f} p95={mins[int(n*0.95)]:.1f}')
            frac = {e: sum(1 for m in mins if m < e) / n * 100 for e in EDGES}
            say('          小于阈值占比: ' +
                '  '.join(f'<{e}px={frac[e]:.1f}%' for e in EDGES))
    say('\n' + '=' * 88)
    say('判读：')
    say('  P3(stride=8) 的每个格子覆盖 8px。若 <8px 占比极低 ⇒ P3 足够，不必加 P2。')
    say('  惯例：<32px 称"小目标"。若 <32px 占比很高但 <8px 很低 ⇒ 仍可只用 P3。')
    say('  只有出现大量 <8px（P3 格子装不下）时，P2 才是必需的。')

    OUT = Path('D:/2/exp_upscale/logs/gt_box_stats.txt')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
