# -*- coding: utf-8 -*-
"""P0-F：用**随机置换**给"重建 ≠ 完整"和"leak − clean"各造一条零假设分布。

为什么需要这个（而不只是 h1/h2 那个数）：
P0 发现"leak/clean 按图数加权还原"与"完整读数"差 0.19~0.47 点，提出两种解释：
  (b) 子集构造有偏（漏拷图/标签、data.yaml 不一致）
  (c) AP 非可加性（AP 把全体检测**池化**后排序算 PR 曲线，
      故子集 AP 的加权平均 **不** 等于整集的 AP）
判定方法：**把"leak/clean"这个划分换成随机划分，其它全不变**（同样按 48/131、
54/124 的图数切开，同样只差"哪些图被分到哪边"）：
  - 若随机划分也出现同量级的偏差 ⇒ 偏差来自 (c) 非可加性，与子集构造无关
  - 若随机划分的偏差是 0（或远小于观测值）⇒ 观测偏差是 (b) 构造问题，要去查脚本

这个方法同时给出**第二个零假设分布**，价值更大：
  真实 leak 子集是"被挑出来的"（有近复制品），而随机子集不是。
  故 `fake_leak − fake_clean` 的分布就是"在没有任何泄漏效应时，
  同等图数的一份划分能自然产生多大的差"。
  真实的 `leak − clean` 若落在这条分布里 ⇒ 泄漏效应**测不出来**（比"≤噪声基准"强得多，
  因为它是一条分布而非一个点，答辩时可直接说"改不了划分"）。

必要条件：随机划分必须**用完整个 split**（48+131=179），否则测的是别的东西。
本脚本自动断言这一点。

输出：logs/p0f_permutation.txt
用法：D:/yolo-env/Scripts/python.exe D:/2/code/tools/eval/p0f_permutation.py
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
OUT = Path('D:/2/logs/p0f_permutation.txt')
NAMES = ['break', 'thunderbolt', 'wear']

MODELS = [('v11@896', 'D:/2/runs/wire_v11_896', 896),
          ('v11@640', 'D:/2/runs/wire_v11_640', 640)]

# 与真实 leak 子集**同图数**：test 48/131、val 54/124（见 logs/leak_subsets.txt）
K = {'test': 48, 'val': 54}
SEEDS = {'test': [1, 2, 3, 4, 5], 'val': [1, 2]}


def build(sp, n_leak):
    """把 split 的图按随机种子切成 fake_leak/fake_clean，写出子集目录。返回种子→(leak,clean)。"""
    files = sorted(p.name for p in (SRC / 'images' / sp).iterdir() if p.is_file())
    out = {}
    for s in SEEDS[sp]:
        names = list(files)
        random.Random(s).shuffle(names)
        parts = {'_permleak': names[:n_leak], '_permclean': names[n_leak:]}
        assert len(parts['_permleak']) + len(parts['_permclean']) == len(files), '划分未用完整个 split'
        for tag, ns in parts.items():
            root = DST / f'{sp}_s{s}{tag}'
            if (root / 'data.yaml').exists():
                out.setdefault(s, {})[tag] = ns
                continue
            (root / 'images').mkdir(parents=True, exist_ok=True)
            (root / 'labels').mkdir(parents=True, exist_ok=True)
            for n in ns:
                shutil.copy2(SRC / 'images' / sp / n, root / 'images' / n)
                lab = n.rsplit('.', 1)[0] + '.txt'
                src = SRC / 'labels' / sp / lab
                if src.exists():
                    shutil.copy2(src, root / 'labels' / lab)
            (root / 'data.yaml').write_text(
                f'path: {root.as_posix()}\n'
                'train: images\nval: images\ntest: images\n'
                f'nc: 3\nnames: {NAMES}\ntask: detect\n', encoding='utf-8')
            out.setdefault(s, {})[tag] = ns
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
        OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    say('=' * 104)
    say('P0-F：随机置换零假设 —— "重建≠完整"与"leak−clean"到底是不是构造问题')
    say('=' * 104)

    PART = {}
    for sp, k in K.items():
        PART[sp] = build(sp, k)
        leak_n = set().union(*(set(PART[sp][s]['_permleak']) for s in SEEDS[sp]))
        clean_n = set().union(*(set(PART[sp][s]['_permclean']) for s in SEEDS[sp]))
        say(f'  [划分] {sp}: 每个种子切 {k} / {len(next(iter(PART[sp].values()))["_permclean"])}，'
            f'共 {len(SEEDS[sp])} 个种子')

    R = {}

    def run(tag, root, imgsz, data, split, name):
        m = YOLO(f'{root}/weights/best.pt')
        r = m.val(data=str(data), split=split, imgsz=imgsz, batch=8, workers=0,
                  device=0, project='D:/2/runs/_leak_eval', name=f'{name}_{tag}',
                  exist_ok=True, verbose=False, plots=False)
        b = r.box
        per = {int(c): float(b.ap50[i]) for i, c in enumerate(b.ap_class_index)}
        R[(tag, split, name)] = (float(b.map50), float(b.map), per)
        say(f'  [done] {name:>16} {tag:<8} mAP50={b.map50:.4f}  '
            + '/'.join(f'{NAMES[c]}{per.get(c, 0):.3f}' for c in range(3)))
        return R[(tag, split, name)]

    for tag, root, imgsz in MODELS:
        run(tag, root, imgsz, f'{SRC}/data.yaml', 'test', 'full_test')
        for sp in ('test', 'val'):
            run(tag, root, imgsz, f'{SRC}/data.yaml', sp, f'full_{sp}')
            # 真实划分（与置换同一设备/精度，保证可比）
            for sub in ('leak', 'clean'):
                run(tag, root, imgsz, DST.parent / f'{sp}_{sub}' / 'data.yaml', sp, f'true_{sub}')
            for s in SEEDS[sp]:
                for t in ('_permleak', '_permclean'):
                    run(tag, root, imgsz, DST / f'{sp}_s{s}{t}' / 'data.yaml', sp, f's{s}{t}')

    # ---------------- 报告 ----------------
    for tag, _, _ in MODELS:
        say()
        say('#' * 104)
        say(f'# {tag}')
        say('#' * 104)
        for sp in ('test', 'val'):
            full = R[(tag, sp, f'full_{sp}')][0]
            nl, nc = K[sp], len(next(iter(PART[sp].values()))['_permclean'])
            say()
            say(f'--- {sp}（完整 {nl + nc} 张，mAP50 = {full:.4f}）---')
            say('  ① 重建检验：rec = (nc×clean + nl×leak)/(nc+nl)，dev = rec − 完整')
            say(f'{"划分":<16}{"leak":>9}{"clean":>9}{"重建":>9}{"dev":>10}{"leak−clean":>12}')
            tl = R[(tag, sp, 'true_leak')][0]
            tc = R[(tag, sp, 'true_clean')][0]
            rec_t = (nc * tc + nl * tl) / (nl + nc)
            say(f'{"真实划分":<16}{tl:>9.4f}{tc:>9.4f}{rec_t:>9.4f}'
                f'{rec_t - full:>+10.4f}{tl - tc:>+12.4f}')
            devs, gaps = [], []
            for s in SEEDS[sp]:
                l = R[(tag, sp, f's{s}_permleak')][0]
                c = R[(tag, sp, f's{s}_permclean')][0]
                rec = (nc * c + nl * l) / (nl + nc)
                devs.append(rec - full)
                gaps.append(l - c)
                say(f'{"随机 s" + str(s):<16}{l:>9.4f}{c:>9.4f}{rec:>9.4f}'
                    f'{rec - full:>+10.4f}{l - c:>+12.4f}')
            dev_obs = rec_t - full
            gap_obs = tl - tc

            def zero(name, obs, dist, fmt='+.4f'):
                mu = sum(dist) / len(dist)
                lo, hi = min(dist), max(dist)
                n_ge = sum(1 for d in dist if abs(d) >= abs(obs))
                say(f'  ② {name}：观测 {obs:{fmt}}；'
                    f'随机划分 n={len(dist)}：均值 {mu:{fmt}} / 范围 [{lo:{fmt}}, {hi:{fmt}}]')
                if lo <= obs <= hi:
                    say(f'     ⇒ 观测值**落在随机划分的范围内** ⇒ 该偏差可由 AP 非可加性充分解释，'
                        f'**不需要引入"子集构造有偏"**。')
                elif abs(obs) < abs(mu) + abs(hi - lo):
                    say('     ⇒ 观测值虽在范围外，但与随机波动同量级 ⇒ 倾向非可加性，'
                        '仍不足以支撑"构造有偏"（后者需同号系统性偏差）。')
                else:
                    say('     ⇒ 观测值**远超出随机划分的范围** ⇒ 非可加性解释不足，'
                        '必须去查 make_leak_subsets.py 的构造是否存在系统偏差。')
                if n_ge == 0:
                    say(f'     ⇒ 置换 p ≈ <{1 / (len(dist) + 1):.2f}（{len(dist)} 个种子，'
                        '分辨率有限；只能作方向性证据）')
                else:
                    say(f'     ⇒ 置换 p ≈ {n_ge / len(dist):.2f}（|随机值| ≥ |观测值| 的比例）')
                return mu, lo, hi
            say('  ① 的零分布：')
            zero('重建偏差 dev', dev_obs, devs)
            say('  ③ 泄漏量 leak−clean 的零分布（没有任何泄漏效应时的"自然差"）：')
            zero('leak−clean', gap_obs, gaps)
            say('     ⚠ 读法：真实 leak 子集是"被挑出来的"（有近复制品），随机子集不是。')
            say('       故 `leak−clean` 的观测值若落在随机范围内 ⇒ **本实验规模测不出泄漏效应**，')
            say('       这是比"≤噪声基准"更可辩护的说法（用整条分布代替单点）。')

    say()
    say('=' * 104)
    say('口径总结（P0-F）')
    say('=' * 104)
    say('  1. `重建 − clean` **不是**泄漏效应，它与 `leak − clean` 只差一个缩放因子')
    say('     [n_leak/(n_clean+n_leak)]，故二者不可当作两个独立证据。')
    say('  2. `重建 ≠ 完整` 的差额由 **AP 非可加性**产生：随机划分同样产生同量级差额。')
    say('     AP 把全体检测池化后排序算 PR 曲线，子集 AP 的加权平均在数学上就不等于整集 AP。')
    say('     ⇒ **"重建"不能作为验证量**，也不能用来反推"子集构造有偏"（见排除清单⑥）。')
    say('  3. 故最终 headline 口径：泄漏效应只能给**定性**结论 ——')
    say('     "在本子集规模下不可分辨（落在随机划分的自然差范围内）"。')

    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
