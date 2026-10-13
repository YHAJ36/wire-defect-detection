# -*- coding: utf-8 -*-
"""把 val / test 各拆成 leak / clean 两个子集，供"泄漏量化"评估使用。

定义（关键，决定结论的含义）：
    **leak**  = 该评估图在 **train** 里存在经定案的近似重复（dHash 候选 + 逐像素 MAD ≤ 20）
    **clean** = 该评估图在 train 里找不到近似重复
    只算 train→eval 方向的泄漏（这才是"评估图不是没见过"的来源）。
    val↔test 之间的近似对**不算**泄漏（两边都不是训练数据，是同一划分内的冗余）。

为什么要拆：报告的 mAP 是 leak+clean 的**混合值**。若不拆开说明，
"数据集有跨 split 近似重复"这条就只能定性说，答辩时无法回答"到底抬了多少"。

复用 `check_near_dup.py` 的 dhash / pixel_mad（import 而非复制，避免两处阈值漂移）。

输出：
    D:/2/dataset_leak_subsets/<split>_{leak,clean}/{images,labels}/   （复制，非软链，Windows 软链要管理员权限）
    D:/2/dataset_leak_subsets/<split>_{leak,clean}/data.yaml
    logs/leak_subsets.txt                                             （名单与计数）
用法：D:/yolo-env/Scripts/python.exe D:/2/code/tools/make_leak_subsets.py
"""
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, 'D:/2/code/tools')
import check_near_dup as cnd  # noqa: E402

SRC = Path('D:/2/dataset_final_bbox')
DST = Path('D:/2/dataset_leak_subsets')
OUT = Path('D:/2/logs/leak_subsets.txt')
NAMES = ['break', 'thunderbolt', 'wear']


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    # ---- 1. 收集 train 的 dHash ----
    say('=' * 88)
    say('val / test 的 leak / clean 拆分')
    say('=' * 88)
    say()
    tr = {p.name: cnd.dhash(p) for p in sorted((SRC / 'images' / 'train').iterdir()) if p.is_file()}
    say(f'  train 已哈希 {len(tr)} 张；判据 MAD ≤ {cnd.MAD_TH:.0f}（同 check_near_dup.py）')
    say()

    report = {}
    for split in ('test', 'val'):
        d = SRC / 'images' / split
        ev = {p.name: cnd.dhash(p) for p in sorted(d.iterdir()) if p.is_file()}
        leak = {}
        for n2, h2 in ev.items():
            best = None
            for n1, h1 in tr.items():
                if cnd.ham(h1, h2) <= 2:
                    m, _ = cnd.pixel_mad('train', n1, split, n2)
                    if m <= cnd.MAD_TH and (best is None or m < best[0]):
                        best = (m, n1)
            if best is not None:
                leak[n2] = best
        clean = [n for n in ev if n not in leak]
        report[split] = (leak, clean)

        say(f'--- {split} ---')
        say(f'  总数 {len(ev)} → **leak {len(leak)} / clean {len(clean)}**'
            f'（泄漏率 {100*len(leak)/len(ev):.1f}%）')
        worst = sorted(leak.items(), key=lambda kv: kv[1][0])[:8]
        say(f'  最相似的 8 对（MAD 最小 = 最像）：')
        for n2, (m, n1) in worst:
            say(f'      MAD={m:>6.2f}  train/{n1}   {split}/{n2}')
        say()

        # ---- 建子集目录 ----
        for tag, names in (('leak', sorted(leak)), ('clean', sorted(clean))):
            root = DST / f'{split}_{tag}'
            (root / 'images').mkdir(parents=True, exist_ok=True)
            (root / 'labels').mkdir(parents=True, exist_ok=True)
            for n in names:
                shutil.copy2(SRC / 'images' / split / n, root / 'images' / n)
                lab = n.rsplit('.', 1)[0] + '.txt'
                src_lab = SRC / 'labels' / split / lab
                if src_lab.exists():
                    shutil.copy2(src_lab, root / 'labels' / lab)
            (root / 'data.yaml').write_text(
                f'path: {root.as_posix()}\n'
                'train: images\nval: images\ntest: images\n'
                f'nc: 3\nnames: {NAMES}\ntask: detect\n',
                encoding='utf-8')
            say(f'  [子集] {root}  图 {len(names)} 张')

        # 名单留档（便于复核与复现）
        (DST / f'{split}_leak_names.txt').write_text('\n'.join(sorted(leak)) + '\n', encoding='utf-8')
        (DST / f'{split}_clean_names.txt').write_text('\n'.join(clean) + '\n', encoding='utf-8')

        # ---- 噪声基准：把 clean 随机对半切 ----
        # 为什么需要：leak 子集只有 ~50 张图，在 50 张上算 AP 的抽样误差本身就有几个点，
        # 若不给出"同条件下两份的 AP 差"这个经验噪声尺度，就无法判断 leak − clean 是真是假。
        rnd = random.Random(42)
        cl = sorted(clean)
        rnd.shuffle(cl)
        for tag, names in (('clean_h1', cl[:len(cl) // 2]), ('clean_h2', cl[len(cl) // 2:])):
            root = DST / f'{split}_{tag}'
            (root / 'images').mkdir(parents=True, exist_ok=True)
            (root / 'labels').mkdir(parents=True, exist_ok=True)
            for n in names:
                shutil.copy2(SRC / 'images' / split / n, root / 'images' / n)
                lab = n.rsplit('.', 1)[0] + '.txt'
                src_lab = SRC / 'labels' / split / lab
                if src_lab.exists():
                    shutil.copy2(src_lab, root / 'labels' / lab)
            (root / 'data.yaml').write_text(
                f'path: {root.as_posix()}\n'
                'train: images\nval: images\ntest: images\n'
                f'nc: 3\nnames: {NAMES}\ntask: detect\n',
                encoding='utf-8')
        say(f'  [噪声基准] {split}_clean 对半切：h1 {len(cl)//2} 张 / h2 {len(cl)-len(cl)//2} 张')
        say()

        # ---- 阈值敏感性：更严的 MAD<=10 一档，只留铁板钉钉的重复 ----
        strict = {}
        for n2 in leak:
            m, n1 = leak[n2]
            if m <= 10.0:
                strict[n2] = (m, n1)
        say(f'  [敏感性] MAD≤10 的严格档：leak {len(strict)} 张'
            f'（相比 MAD≤20 的 {len(leak)} 张）')
        say()

    say()
    say('=' * 88)
    say('下一步')
    say('=' * 88)
    say('  跑 code/tools/eval/eval_leak_subsets.py，对 v11@896 与 v11@640 各算')
    say('  leak / clean 两组的 mAP50 与 mAP50-95，比较 leak − clean。')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
