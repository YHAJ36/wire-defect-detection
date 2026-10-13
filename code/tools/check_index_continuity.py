# -*- coding: utf-8 -*-
"""编号区间的连续性检查：相邻编号的图是不是"内容连续"（同一条视频/同一批连拍）。

为什么查：数据集的文件名带编号（如 firc_xianlan_1044.jpg）。若数据是从视频按帧
抽出来的，那么**编号相邻 ⇒ 画面几乎一样**，这会产生两个后果：
  1. "随机划分"其实带块状结构：同一小段编号的图会被整体分到同一侧，
     使 val 与 test 之间出现**系统性**的（而非随机的）内容差
     —— 这正是"val 与 test 有 +1.6~2.8 点系统性偏置"的一个候选来源。
  2. 近重复检测（dHash ≤2 且 MAD ≤20）本该抓到这些对，但要确认它没漏。
     本脚本用**同一套判据**（dHash 汉明距离）直接量"编号距离"与"内容相似度"的关系。

方法：对每个 split 内部，把编号差 1、2、3 的图对与**随机图对**作对照，
各抽样若干对，比较 dHash 汉明距离的分布。
判定：若"编号差 1"的汉明距离分布**显著低于**随机图对 ⇒ 编号连续 ⇒ 存在块状结构。

只读，纯 CPU，不碰 GPU，不写任何数据集。
输出：logs/index_continuity.txt
用法：D:/yolo-env/Scripts/python.exe D:/2/code/tools/check_index_continuity.py
"""
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path('D:/2/dataset_final_bbox/images')
OUT = Path('D:/2/logs/index_continuity.txt')
SPLITS = ['train', 'val', 'test']
NUM = re.compile(r'(\d+)(?!.*\d)')     # 文件名里最后一段数字


def dhash(p, size=8):
    im = Image.open(p).convert('L').resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def ham(a, b):
    return int(np.count_nonzero(a != b))


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    say('=' * 96)
    say('编号区间连续性检查（同一 split 内部）')
    say('=' * 96)

    for sp in SPLITS:
        files = sorted(p for p in (SRC / sp).iterdir() if p.is_file())
        idx = {}
        for p in files:
            m = NUM.search(p.stem)
            if m:
                idx.setdefault(int(m.group(1)), []).append(p)
        # 同一编号有多张（多源图共享编号）时只取第一张，避免把"同编号"当成"相邻编号"
        one = {i: v[0] for i, v in idx.items()}
        nums = sorted(one)
        say()
        say(f'--- {sp}：{len(files)} 张，带编号 {len(one)} 个，'
            f'编号范围 {nums[0]}~{nums[-1]} ---')
        if len(one) < 40:
            say('  图太少，跳过。')
            continue

        hs = {i: dhash(one[i]) for i in nums}

        # 编号差 1/2/3 的对（只取编号确实相邻的）
        res = {}
        for gap in (1, 2, 3):
            ds = [ham(hs[nums[i]], hs[nums[i + gap]])
                  for i in range(len(nums) - gap) if nums[i + gap] - nums[i] == gap]
            res[gap] = ds
        # 随机对照：随机配 3000 对
        rnd = random.Random(0)
        rnd_pairs = [(rnd.randrange(len(nums)), rnd.randrange(len(nums))) for _ in range(3000)]
        rnd_pairs = [(a, b) for a, b in rnd_pairs if a != b]
        res[0] = [ham(hs[nums[a]], hs[nums[b]]) for a, b in rnd_pairs]

        say(f'{"对比":<16}{"对数":>8}{"平均汉明":>11}{"中位":>8}{"≤2（近重复）占比":>18}')
        for gap in (1, 2, 3, 0):
            a = np.array(res[gap], dtype=np.float64)
            lab = f'编号差 {gap}' if gap else '随机图对'
            say(f'{lab:<16}{a.size:>8}{a.mean():>11.2f}{np.median(a):>8.0f}'
                f'{(a <= 2).mean() * 100:>17.1f}%')
        g1, rndv = np.array(res[1], float), np.array(res[0], float)
        dd = rndv.mean() - g1.mean()
        say()
        say(f'  判读：编号差 1 的平均汉明距离 {g1.mean():.2f}，'
            f'随机图对 {rndv.mean():.2f}，差 {dd:+.2f}')
        if dd >= 2:
            say('  ⇒ 编号相邻的图**明显比随机图对更相似** ⇒ **存在块状结构**。')
            say('     后果：随机划分会按编号块成片切分 ⇒ val/test 之间可能出现系统性内容差；')
            say('     这也**降低**了"划分流程无偏"这一条的说服力（流程无偏 ≠ 产物无偏）。')
            say('     ⚠ 但注意：这解释的是"为什么 val/test 会有系统性差"，'
                '**不解释方向**——要解释"test 反而更高"仍需单独论证。')
        else:
            say('  ⇒ 编号相邻与随机图对的内容相似度**没有明显差别** ⇒ 编号不带块状结构。')
            say('     则"随机划分的产物有系统性内容差"这一候选被削弱。')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
