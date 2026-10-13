# -*- coding: utf-8 -*-
"""P0-F 补测：把 val 侧的置换种子从 2 个补到 5 个（B-3）。

只做 val，不重跑 test。自包含（连同 s1/s2、full、true 一起在同一设备重评），
输出 5 个种子的 val 零分布，直接给出最终 p 值。
输出：logs/p0f_val_extra.txt
用法：D:/yolo-env/Scripts/python.exe D:/2/code/tools/eval/p0f_val_extra.py
"""
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, 'D:/2/code/yolov11')
sys.path.insert(0, 'D:/2/code/yolov12')

from ultralytics import YOLO  # noqa: E402

SRC = Path('D:/2/dataset_final_bbox')
DST = Path('D:/2/dataset_leak_subsets/_perm')
OUT = Path('D:/2/logs/p0f_val_extra.txt')
NAMES = ['break', 'thunderbolt', 'wear']
SP = 'val'
K = 54                       # val_leak 的图数（与真实划分一致）
SEEDS = [1, 2, 3, 4, 5]
MODELS = [('v11@896', 'D:/2/runs/wire_v11_896', 896),
          ('v11@640', 'D:/2/runs/wire_v11_640', 640)]


def build():
    files = sorted(p.name for p in (SRC / 'images' / SP).iterdir() if p.is_file())
    for s in SEEDS:
        names = list(files)
        random.Random(s).shuffle(names)
        parts = {'_permleak': names[:K], '_permclean': names[K:]}
        assert len(parts['_permleak']) + len(parts['_permclean']) == len(files)
        for tag, ns in parts.items():
            root = DST / f'{SP}_s{s}{tag}'
            if (root / 'data.yaml').exists():
                continue
            (root / 'images').mkdir(parents=True, exist_ok=True)
            (root / 'labels').mkdir(parents=True, exist_ok=True)
            for n in ns:
                shutil.copy2(SRC / 'images' / SP / n, root / 'images' / n)
                lab = n.rsplit('.', 1)[0] + '.txt'
                src = SRC / 'labels' / SP / lab
                if src.exists():
                    shutil.copy2(src, root / 'labels' / lab)
            (root / 'data.yaml').write_text(
                f'path: {root.as_posix()}\ntrain: images\nval: images\ntest: images\n'
                f'nc: 3\nnames: {NAMES}\ntask: detect\n', encoding='utf-8')


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    build()
    say('=' * 104)
    say('P0-F 补测：val 侧置换种子 2 → 5（B-3）')
    say('=' * 104)

    R = {}

    def run(tag, root, imgsz, data, name):
        m = YOLO(f'{root}/weights/best.pt')
        r = m.val(data=str(data), split=SP, imgsz=imgsz, batch=8, workers=0,
                  device=0, project='D:/2/runs/_leak_eval', name=f'{name}_{tag}',
                  exist_ok=True, verbose=False, plots=False)
        R[(tag, name)] = float(r.box.map50)
        say(f'  [done] {name:>16} {tag:<8} mAP50={r.box.map50:.4f}')
        return R[(tag, name)]

    for tag, root, imgsz in MODELS:
        run(tag, root, imgsz, f'{SRC}/data.yaml', 'full_val')
        run(tag, root, imgsz, DST.parent / 'val_leak' / 'data.yaml', 'true_leak')
        run(tag, root, imgsz, DST.parent / 'val_clean' / 'data.yaml', 'true_clean')
        for s in SEEDS:
            for t in ('_permleak', '_permclean'):
                run(tag, root, imgsz, DST / f'{SP}_s{s}{t}' / 'data.yaml', f's{s}{t}')

    NC = len(sorted((SRC / 'images' / SP).iterdir())) - K   # clean 图数 = 178 - 54 = 124

    say()
    for tag, _, _ in MODELS:
        full = R[(tag, 'full_val')]
        tl, tc = R[(tag, 'true_leak')], R[(tag, 'true_clean')]
        rec_t = (NC * tc + K * tl) / (NC + K)
        gap_t = tl - tc
        devs, gaps = [], []
        for s in SEEDS:
            l, c = R[(tag, f's{s}_permleak')], R[(tag, f's{s}_permclean')]
            rec = (NC * c + K * l) / (NC + K)
            devs.append(rec - full)
            gaps.append(l - c)
        say()
        say(f'--- {tag} val（完整 {full:.4f}；真实 leak {tl:.4f} / clean {tc:.4f}）---')
        say(f'  {"划分":<10}{"leak":>9}{"clean":>9}{"重建":>9}{"dev":>10}{"leak−clean":>12}')
        say(f'  {"真实":<10}{tl:>9.4f}{tc:>9.4f}{rec_t:>9.4f}{rec_t - full:>+10.4f}{gap_t:>+12.4f}')
        for s in SEEDS:
            l, c = R[(tag, f's{s}_permleak')], R[(tag, f's{s}_permclean')]
            say(f'  {"随机s" + str(s):<10}{l:>9.4f}{c:>9.4f}'
                f'{(NC * c + K * l) / (NC + K):>9.4f}'
                f'{(NC * c + K * l) / (NC + K) - full:>+10.4f}{l - c:>+12.4f}')
        mu_d, lo_d, hi_d = sum(devs) / len(devs), min(devs), max(devs)
        mu_g, lo_g, hi_g = sum(gaps) / len(gaps), min(gaps), max(gaps)
        p_d = sum(1 for d in devs if abs(d) >= abs(rec_t - full)) / len(devs)
        p_g = sum(1 for g in gaps if abs(g) >= abs(gap_t)) / len(gaps)
        say(f'  重建偏差 dev：观测 {rec_t - full:+.4f}；随机 n={len(devs)} 均值 {mu_d:+.4f} '
            f'/ 范围 [{lo_d:+.4f}, {hi_d:+.4f}]；p ≈ {p_d:.2f}')
        say(f'  leak−clean：观测 {gap_t:+.4f}；随机 n={len(gaps)} 均值 {mu_g:+.4f} '
            f'/ 范围 [{lo_g:+.4f}, {hi_g:+.4f}]；p ≈ {p_g:.2f}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
