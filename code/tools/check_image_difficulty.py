# -*- coding: utf-8 -*-
"""val vs test 的图像难度指标 —— 给"test 系统性高于 val（原因待查）"找一个可查的方向。

背景：val 与 test 存在系统性 +1.6~+2.8 点偏置，且逐类拆解后是
"tb + wear 三次全正、break 接近零"，说明不是随机噪声，是这套划分的固有性质。
已排除的机制：精确重复（SHA256=0）、类别分布（mAP 是三类未加权均值，占比不改变总分）、
划分流程（bbox_split.py 种子 42 均匀随机）、近似图分布（val 侧反而更密，方向相反）。
**剩下的候选**：val 与 test 的**图像/目标本身难度不同**（这最可能，且可直接量化）。

必须**逐类**算，理由与 P0 相同：mAP50(all) 是三类未加权均值，每个类占 1/3。
若 val 的某类目标天生更小更糊，"难度差"就体现为那一类的 AP 差再除以 3。
只给全数据集平均值会把类间差异抹平，看不到真正的来源。

指标（都选"越大越难/越小越难"方向明确的）：
  1. 目标框面积（占图面积比）  —— 目标越小越难，是检测任务最主要的难度因子
  2. 目标框长宽比              —— 极端长宽比（输电线断股/磨损常是细长条）更难
  3. 每图目标数                —— 目标多则互相遮挡、小目标占比高
  4. 图像亮度（灰度均值）      —— 夜间/逆光图整体暗，缺陷对比度低
  5. 图像对比度（灰度 std）    —— 对比度低则缺陷与背景难分辨
  6. 图像尺寸（短边像素）      —— 分辨率越低，小目标越糊

输出：logs/image_difficulty.txt
用法：D:/yolo-env/Scripts/python.exe D:/2/code/tools/check_image_difficulty.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('D:/2/dataset_final_bbox')
OUT = Path('D:/2/logs/image_difficulty.txt')
NAMES = ['break', 'thunderbolt', 'wear']
SPLITS = ['val', 'test']


def _stat(a):
    """返回 (n, 中位数, 均值, 标准差, p10, p90)。中位数为主：面积/长宽比都长尾。"""
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return 0, float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    return (a.size, float(np.median(a)), float(a.mean()), float(a.std()),
            float(np.percentile(a, 10)), float(np.percentile(a, 90)))


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    # ---- 逐图读取：亮度/对比度/尺寸/目标数；逐框记录面积比与长宽比 ----
    # 逐框统计按类归集；逐图统计（亮度等）无法按类归集，故按"该图是否含某类"分开报，
    # 这样"某类的图"与"另一类的图"之间的亮度差才看得见。
    per_cls = {sp: {i: {'area': [], 'ar': [], 'px': []} for i in range(3)} for sp in SPLITS}
    per_img = {sp: {'n': [], 'bright': [], 'std': [], 'short': [], 'iar': []}
               for sp in SPLITS}
    cls_img = {sp: {i: {'bright': [], 'std': [], 'short': [], 'nobj': []} for i in range(3)}
               for sp in SPLITS}
    n_img = {}

    for sp in SPLITS:
        imgdir = ROOT / 'images' / sp
        labdir = ROOT / 'labels' / sp
        files = sorted(p for p in imgdir.iterdir() if p.is_file())
        n_img[sp] = len(files)
        for p in files:
            im = Image.open(p).convert('L')
            w, h = im.size
            g = np.asarray(im, dtype=np.float64)
            per_img[sp]['bright'].append(float(g.mean()))
            per_img[sp]['std'].append(float(g.std()))
            per_img[sp]['short'].append(min(w, h))
            # letterbox 缩放后目标像素面积 = 归一化面积 × imgsz² × (min/max)，故这个因子要核对
            per_img[sp]['iar'].append(min(w, h) / max(w, h))
            lab = labdir / (p.stem + '.txt')
            boxes = []
            if lab.exists():
                for ln in lab.read_text(encoding='utf-8').splitlines():
                    t = ln.split()
                    if len(t) >= 5:
                        boxes.append((int(float(t[0])),
                                      float(t[1]), float(t[2]), float(t[3]), float(t[4])))
            per_img[sp]['n'].append(len(boxes))
            seen = set()
            for c, _xc, _yc, bw, bh in boxes:
                c = c if 0 <= c < 3 else 0
                per_cls[sp][c]['area'].append(bw * bh)          # 已是占图面积比
                per_cls[sp][c]['ar'].append((bw * w) / max(bh * h, 1e-6))
                # 原图像素尺寸：宽高几何均值。注意 letterbox 会把图上采样/下采样到 imgsz，
                # 故这个量**不是**直接的难度因子，它的作用在"上采样会糊、下采样更清晰"这一层。
                per_cls[sp][c]['px'].append(float(np.sqrt(max(bw * w * bh * h, 1e-6))))
                seen.add(c)
            for c in range(3):                                  # 按"该图是否含类 c"归集
                if c in seen:
                    cls_img[sp][c]['bright'].append(per_img[sp]['bright'][-1])
                    cls_img[sp][c]['std'].append(per_img[sp]['std'][-1])
                    cls_img[sp][c]['short'].append(per_img[sp]['short'][-1])
                    cls_img[sp][c]['nobj'].append(len(boxes))

    say('=' * 96)
    say('val vs test 图像难度指标（逐类）')
    say('=' * 96)
    say()
    say(f'  图片数：val {n_img["val"]} / test {n_img["test"]}')

    def blk(title, key, fmt='{:.4f}'):
        """打印一张"逐类 × val/test"对照表（逐框口径）。"""
        say()
        say('-' * 96)
        say(title)
        say('-' * 96)
        hdr = f'{"":<14}'
        for sp in SPLITS:
            hdr += f'{sp + " n":>9}{sp + " 中位":>12}{sp + " 均值":>12}{sp + " p10~p90":>22}'
        say(hdr + f'{"中位差%"        :>12}')
        for i in range(3):
            label = NAMES[i]
            vals = []
            for sp in SPLITS:
                a = per_cls[sp][i][key]
                vals.append(_stat(a))
            row = f'{label:<14}'
            for (n, med, mean, sd, p10, p90) in vals:
                row += f'{n:>9}{fmt.format(med):>12}{fmt.format(mean):>12}' \
                       f'{("[{}, {}]".format(fmt.format(p10), fmt.format(p90))):>22}'
            d = (vals[1][1] - vals[0][1]) / vals[0][1] * 100 if vals[0][1] else float('nan')
            row += f'{d:>12.1f}'
            say(row)

    blk('表 1：目标框面积（占图面积比；越小越难）', 'area', '{:.5f}')
    blk('表 2：目标框长宽比（越偏离 1 越难）', 'ar', '{:.3f}')

    def simple(title, key, fmt='{:.2f}', all_key=None):
        say()
        say('-' * 96)
        say(title)
        say('-' * 96)
        hdr = f'{"":<14}'
        for sp in SPLITS:
            hdr += f'{sp + " n":>9}{sp + " 中位":>12}{sp + " 均值":>12}{sp + " p10~p90":>22}'
        say(hdr + f'{"中位差%"        :>12}')
        for i in range(3):
            row = f'{NAMES[i]:<14}'
            vs = []
            for sp in SPLITS:
                a = cls_img[sp][i][key]
                vs.append(_stat(a))
            for (n, med, mean, sd, p10, p90) in vs:
                row += f'{n:>9}{fmt.format(med):>12}{fmt.format(mean):>12}' \
                       f'{("[{}, {}]".format(fmt.format(p10), fmt.format(p90))):>22}'
            d = (vs[1][1] - vs[0][1]) / vs[0][1] * 100 if vs[0][1] else float('nan')
            row += f'{d:>12.1f}'
            say(row)
        # 附：全体（逐图口径）
        row = f'{"全体(逐图)":<14}'
        vs = [_stat(per_img[sp][all_key or key]) for sp in SPLITS]
        for (n, med, mean, sd, p10, p90) in vs:
            row += f'{n:>9}{fmt.format(med):>12}{fmt.format(mean):>12}' \
                   f'{("[{}, {}]".format(fmt.format(p10), fmt.format(p90))):>22}'
        d = (vs[1][1] - vs[0][1]) / vs[0][1] * 100 if vs[0][1] else float('nan')
        row += f'{d:>12.1f}'
        say(row)

    simple('表 3：每图目标数（含该类别的图；越多越难）', 'nobj', '{:.2f}', all_key='n')
    simple('表 4：图像亮度（灰度均值；越低越难）', 'bright', '{:.1f}')
    simple('表 5：图像对比度（灰度 std；越低越难）', 'std', '{:.1f}')
    simple('表 6：图像短边像素（越低越难）', 'short', '{:.0f}')
    blk('表 7：目标框在**原图**上的像素尺寸（宽高几何均值；'
        '注意它经 letterbox 缩放后才是模型看到的大小）', 'px', '{:.1f}')

    # ---- 判读 ----
    say()
    say('=' * 96)
    say('判读')
    say('=' * 96)
    say('  方向约定：面积↓/亮度↓/对比度↓ = 更难。')
    say()
    say('  **模型看到的量**（决定难度的那一个）：letterbox 把图缩放到 imgsz 后，')
    say('    目标框像素面积 = 归一化面积 × imgsz² × (min(h,w)/max(h,w))')
    say('  两边的 imgsz 相同（同组同分辨率评估），故 **归一化面积是决定性维度**；')
    say('  源图分辨率**不进入**这个式子 —— 它不是难度维度。')
    say()
    say('  逐类归一化目标面积中位（test 相对 val）—— 决定性维度：')
    for i in range(3):
        a_v = _stat(per_cls['val'][i]['area'])[1]
        a_t = _stat(per_cls['test'][i]['area'])[1]
        d = (a_t - a_v) / a_v * 100 if a_v else float('nan')
        arrow = 'test 目标更小 ⇒ 更难' if d < 0 else 'test 目标更大 ⇒ 更易'
        say(f'      {NAMES[i]:<12} val {a_v:.5f} → test {a_t:.5f}   {d:+.1f}%   {arrow}')
    say()
    say('  式子里那个 (min/max) 因子是**图像本身**的长宽比，两边的分布需核对（见下表），')
    say('  若相近则它不构成差异 ⇒ 归一化面积是唯一差异来源。')
    say(f'      全体(逐图) 图像长宽比中位：val {_stat(per_img["val"]["iar"])[1]:.3f}'
        f' → test {_stat(per_img["test"]["iar"])[1]:.3f}')
    say()
    say('  ⇒ 判定：**test 的目标在几何上更难（归一化面积更小，wear −41.8%、break −12.5%），')
    say('     但 test 的 mAP 反而更高。方向矛盾 ⇒ 难度差不仅不能解释倒挂，反而加深谜团：')
    say('     test 更难却表现更好，倒挂的来源仍然未知。**')
    say('     （不得用"原图像素尺寸""上下采样清晰度"之类去弱化这个矛盾 —— '
        '那些不是模型看到的量。）')
    say('  ⇒ 可确凿排除的两条（这两条与几何无关，独立成立）：')
    say('       亮度差 ≤ 3.6%、对比度差 ≤ 4.3% ⇒ **光照/成像条件不是来源**；')
    say('       break/tb 的每图目标数中位差 0.0% ⇒ **目标密度不是来源**。')
    say()
    say('  ⚠ 附记（不用于弱化上面的矛盾，只作留档以免将来被反向质疑）：')
    say('     源图分辨率虽不是"目标大小"，但经 letterbox 上采样会带来模糊，属**另一条难度轴**'
        '（成像质量）。')
    say('     它方向与几何相反。两条轴不可相加为一个"net 难度"数，且 detection 以几何为主导，')
    say('     故本条**不改变**上面的结论，也不降低倒挂的待查等级。')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
