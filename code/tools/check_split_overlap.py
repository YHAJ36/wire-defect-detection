# -*- coding: utf-8 -*-
"""检查 dataset_final_bbox 的 split 之间有没有图片泄漏（内容级去重），并统计类别分布。

**为什么需要**：896 组测出反向异常 —— test 的 mAP50 比 val 高
（v11 +0.0158：0.8990 → 0.9148；v5 +0.0220：0.9090 → 0.9310）。
正常情况 test ≤ val（val 参与模型选择，test 是"没见过的"）。
最可疑的解释是 val/test 划分有重叠图片 ⇒ 需先排除数据泄漏，再谈"896 结论"。

判据用**文件内容的 SHA256**而非文件名：同一张图可能换了名字分别落进两个 split，
只比文件名会漏掉。反之同名文件内容相同则是同一张图。

顺带统计三个 split 的类别分布（每类 image 数 / instance 数）——若重叠为 0，
分布差异是"test 更难/更易"的下一个解释候选。

输出：logs/split_overlap.txt（同时打印到 stdout）
用法：D:/yolo-env/Scripts/python.exe D:/2/code/tools/check_split_overlap.py
"""
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('D:/2/dataset_final_bbox')
SPLITS = ['train', 'val', 'test']
IMAGES = ROOT / 'images'
LABELS = ROOT / 'labels'
OUT = Path('D:/2/logs/split_overlap.txt')
NAMES = ['break', 'thunderbolt', 'wear']


def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    # Windows 控制台默认 GBK，✅/❌ 之类字符会 UnicodeEncodeError 把脚本打死
    # （而文件写入在打印之后 ⇒ 整个结果丢失）。这里强制 stdout 用 UTF-8 + 替换。
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    # ---- 1. 逐 split 算图片内容哈希 ----
    hashes = {}          # split -> {sha: [文件名, ...]}
    for s in SPLITS:
        d = IMAGES / s
        if not d.is_dir():
            say(f'⚠ 目录不存在：{d}')
            continue
        m = defaultdict(list)
        for p in sorted(d.iterdir()):
            if p.is_file():
                m[sha256(p)].append(p.name)
        hashes[s] = dict(m)

    say('=' * 100)
    say('图片内容哈希（SHA256）重叠检查 —— 判据是内容而非文件名')
    say('=' * 100)
    say()
    for s in SPLITS:
        if s in hashes:
            say(f'  {s:<6} 图片 {len(hashes[s])} 个（唯一内容）  '
                f'文件 {sum(len(v) for v in hashes[s].values())} 个')

    # ---- 2. 两两重叠 ----
    say()
    say('-' * 100)
    say('两两重叠')
    say('-' * 100)
    total_overlap = 0
    details = {}
    for a, b in [('val', 'test'), ('train', 'test'), ('train', 'val')]:
        if a not in hashes or b not in hashes:
            continue
        common = sorted(set(hashes[a]) & set(hashes[b]))
        details[(a, b)] = common
        say()
        say(f'  {a} ∩ {b}：**{len(common)} 张**')
        if common:
            for h in common[:10]:
                say(f'      {h[:12]}…  {a}: {hashes[a][h]}   {b}: {hashes[b][h]}')
            if len(common) > 10:
                say(f'      …（共 {len(common)} 张，只列前 10）')
        if (a, b) == ('val', 'test'):
            total_overlap = len(common)

    # ---- 3. 类别分布 ----
    say()
    say('=' * 100)
    say('类别分布（逐 split）')
    say('=' * 100)
    say()
    say(f'{"split":<7}{"图片数":>8}{"instances":>11}   ' +
        '   '.join(f'{n:>16}' for n in NAMES))
    say(f'{"":<7}{"":>8}{"":>11}   ' +
        '   '.join(f'{"图/实例":>16}' for _ in NAMES))
    for s in SPLITS:
        d = LABELS / s
        if not d.is_dir():
            continue
        inst = Counter()
        imgcnt = Counter()
        n_img = 0
        for p in sorted(d.iterdir()):
            if not p.is_file() or p.suffix != '.txt':
                continue
            n_img += 1
            seen = set()
            for ln in p.read_text(encoding='utf-8', errors='ignore').splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                c = int(float(ln.split()[0]))
                inst[c] += 1
                seen.add(c)
            for c in seen:
                imgcnt[c] += 1
        cells = '   '.join(f'{imgcnt.get(i,0):>7}/{inst.get(i,0):<8}' for i in range(3))
        say(f'{s:<7}{n_img:>8}{sum(inst.values()):>11}   {cells}')

    # 占比（按 instance）—— 便于看"某类在 test 里占比更高"
    say()
    say('按 instance 的占比（%）：')
    for s in SPLITS:
        d = LABELS / s
        if not d.is_dir():
            continue
        inst = Counter()
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix == '.txt':
                for ln in p.read_text(encoding='utf-8', errors='ignore').splitlines():
                    ln = ln.strip()
                    if ln:
                        inst[int(float(ln.split()[0]))] += 1
        tot = sum(inst.values()) or 1
        pcs = '  '.join(f'{NAMES[i]}: {100*inst.get(i,0)/tot:>5.1f}%' for i in range(3))
        say(f'  {s:<7} {pcs}   (共 {tot})')

    # ---- 4. 判读 ----
    say()
    say('=' * 100)
    say('判读')
    say('=' * 100)
    if total_overlap > 0:
        say(f'  [泄漏] **val ∩ test = {total_overlap} 张，存在数据泄漏**')
        say('     ⇒ 896 组"test > val"的反常由此可以解释，896 结论待修正：')
        say('        test 读数不再是"没见过的数据"，不能用来支持"896 泛化更好"。')
    else:
        say('  [通过] val ∩ test = 0，**无 val/test 泄漏**。')
        say('     ⇒ 896 的"test > val"需另找解释（见上面的类别分布差异）。')
    tt = details.get(('train', 'test'), [])
    tv = details.get(('train', 'val'), [])
    if tt or tv:
        say(f'  [注意] 另：train ∩ test = {len(tt)} 张，train ∩ val = {len(tv)} 张'
            '（超标即训练集与评估集泄漏，比 val/test 泄漏更严重）')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
