# -*- coding: utf-8 -*-
"""两件事一起查：P0-B（CPU/GPU 设备差异）与 P1（val 泄漏是否影响早停选择）。

=============================================================================
P0-B 为什么必须查：CPU 评估与 GPU 评估不可混算
=============================================================================
P0 发现"leak/clean 两子集按图数加权还原"与"记录在案的完整读数"差 0.2~0.5 点：
    896 test：重建 0.9168 vs 记录 0.9148（差 +0.0020）
    640 test：重建 0.9032 vs 记录 0.8992（差 +0.0040）
两个可能来源，**结论完全不同**：
  (a) **设备差异**：完整读数在 GPU(device=0) 上跑的，子集读数在 CPU 上跑的。
      CPU/GPU 的浮点与 NMS 实现可能有微小差异 ⇒ 若属此类，则
      **"GPU 完整读数"与"CPU 子集读数"不可混算**，泄漏效应必须全部在同一设备上重推。
  (b) **子集构造有偏**：则要去查 `make_leak_subsets.py`（漏拷图/标签、或
      data.yaml 的 nc/names 不一致导致某类被丢弃）。
判法：**用 CPU 重跑一次完整 test（179 张）**：
  - CPU 完整 ≈ 0.9168（=重建值） ⇒ 是 (a) 设备差异
  - CPU 完整 ≈ 0.9148（=GPU 记录值） ⇒ 是 (b) 子集构造问题

=============================================================================
P1 为什么要查：best.pt 是按**完整 val** 选的，而 val 有 30% 泄漏
=============================================================================
若 best.pt 的优势主要来自"在 val 的泄漏图上表现好"，则
**所有报告数字偏乐观**，PDF 必须声明。
判法：比较 best.pt 与 last.pt 在 **val_clean** 上的读数：
  - best 在 val_clean 上仍占优 ⇒ 早停选择稳，泄漏不影响选择
  - best 在 val_clean 上不占优 ⇒ 早停选的是"在 val_leak 上好的模型"

⚠ **预先声明的功效限制（否则会把"测不出"误读成"没问题"）**：
val 子集的噪声基准高达 **9 点**（P0 实测：clean 对半切 h1 vs h2），
而要侦测的选择效应只有 ~1 点 ⇒ **本检验只能发现"很大"的选择偏置，
测不出 1~2 点的偏置**。故"不可分辨"≠"选择无偏"，必须如实这么写。

⚠ **另给一个不依赖子集样本量的判据（机制层面，比上面的检验更可靠）**：
早停是在**轮次之间**取 argmax。若泄漏带来的抬升在**各轮之间近似不变**
（因为"近复制品更容易"是图像的属性、与轮次无关），则它对所有轮次是同一个
常数偏移 ⇒ **不改变 argmax ⇒ 早停不受泄漏影响**。
这个前提可以直接测：比较 **best.pt 与 last.pt 的 leak−clean**，
两者接近 ⇒ 抬升与轮次无关 ⇒ 选择无偏。
（反之，若后者明显更大，说明泄漏抬升随训练增长 ⇒ 会把选择推向更晚的轮次。）

输出：logs/device_selection_check.txt
用法：D:/yolo-env/Scripts/python.exe D:/2/code/tools/eval/check_device_and_selection.py
"""
import sys
from pathlib import Path

sys.path.insert(0, 'D:/2/code/yolov11')
sys.path.insert(0, 'D:/2/code/yolov12')

from ultralytics import YOLO  # noqa: E402

FULL = 'D:/2/dataset_final_bbox/data.yaml'
SUB = Path('D:/2/dataset_leak_subsets')
OUT = Path('D:/2/logs/device_selection_check.txt')
NAMES = ['break', 'thunderbolt', 'wear']

MODELS = [('v11@896', 'D:/2/runs/wire_v11_896', 896),
          ('v11@640', 'D:/2/runs/wire_v11_640', 640)]

# 记录在案的 **GPU(device=0)** 完整读数，用于与本次 CPU 读数对照。
# 来源：logs/eval_896.log（test/val）；runs/wire_v11_640/results.csv 的 fitness-best 轮（val）。
GPU = {
    'v11@896': {'test': 0.9148, 'val': 0.8990},
    'v11@640': {'test': 0.8992, 'val': 0.8716},
}
# val 子集的噪声基准（P0 实测：clean 对半切 h1 vs h2），来源 logs/leak_eval.txt
NOISE_VAL = {'v11@896': 0.0906, 'v11@640': 0.0892}


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    R = {}

    def run(tag, w, imgsz, name, data, split):
        m = YOLO(w)
        r = m.val(data=data, split=split, imgsz=imgsz, batch=8, workers=0,
                  device='cpu', project='D:/2/runs/_leak_eval',
                  name=f'{name}_{tag}', exist_ok=True, verbose=False, plots=False)
        b = r.box
        per = {int(c): float(b.ap50[i]) for i, c in enumerate(b.ap_class_index)}
        say(f'  [done] {name:>16} {tag:<8} mAP50={b.map50:.4f}  '
            f'mAP50-95={b.map:.4f}  类=' + '/'.join(
                f'{NAMES[c]}{v:.3f}' for c, v in sorted(per.items())))
        return float(b.map50), float(b.map), per

    for tag, root, imgsz in MODELS:
        R[(tag, 'test_full')] = run(tag, f'{root}/weights/best.pt', imgsz,
                                    'cpufull_test', FULL, 'test')
        R[(tag, 'val_full', 'best')] = run(tag, f'{root}/weights/best.pt', imgsz,
                                           'valfull_best', FULL, 'val')
        R[(tag, 'val_full', 'last')] = run(tag, f'{root}/weights/last.pt', imgsz,
                                           'valfull_last', FULL, 'val')
        for who in ('best', 'last'):
            for sub in ('leak', 'clean'):
                R[(tag, f'val_{sub}', who)] = run(
                    tag, f'{root}/weights/{who}.pt', imgsz,
                    f'val{sub}_{who}', str(SUB / f'val_{sub}' / 'data.yaml'), 'val')

    # ================= P0-B：设备差异 =================
    say()
    say('=' * 104)
    say('P0-B：CPU vs GPU 设备差异（判"重建≠完整"是设备问题还是子集构造问题）')
    say('=' * 104)
    say(f'{"模型":<10}{"GPU完整test":>13}{"CPU完整test":>13}{"CPU−GPU":>11}'
        f'{"重建(P0,CPU)":>14}{"CPU−重建":>11}{"判定":>26}')
    RECON = {'v11@896': 0.9168, 'v11@640': 0.9032}   # 来自 logs/leak_eval.txt
    verd = {}
    for tag, _, _ in MODELS:
        cpu = R[(tag, 'test_full')][0]
        g, rec = GPU[tag]['test'], RECON[tag]
        d_dev, d_con = cpu - g, cpu - rec
        if abs(d_dev) > abs(d_con):
            v = '⇒ 设备差异 (a)'
        else:
            v = '⇒ 子集构造问题 (b)'
        say(f'{tag:<10}{g:>13.4f}{cpu:>13.4f}{d_dev:>+11.4f}{rec:>14.4f}{d_con:>+11.4f}{v:>26}')
        verd[tag] = (g, cpu, rec, d_dev, d_con)
    say()
    say('  读法：`CPU−GPU` 是设备差异的直接估计；`CPU−重建` 是子集构造偏差的直接估计。')
    say('        **哪个更小，就说明哪条路径能自洽**。')
    say()
    say('  另附 val 侧同样的对照（GPU val 取自各 run 的 fitness-best 轮）：')
    say(f'{"模型":<10}{"GPU完整val":>13}{"CPU完整val":>13}{"CPU−GPU":>11}')
    for tag, _, _ in MODELS:
        cpu = R[(tag, 'val_full', 'best')][0]
        g = GPU[tag]['val']
        say(f'{tag:<10}{g:>13.4f}{cpu:>13.4f}{cpu - g:>+11.4f}')
    say('  ⚠ 若两处 `CPU−GPU` 同号且量级接近 ⇒ 是**系统性设备偏移**（而非随机误差），')
    say('     则"GPU 完整读数"与"CPU 子集读数"**必须分别处理**，泄漏效应要在同一设备上重推。')

    # ================= P1：早停选择是否受泄漏影响 =================
    say()
    say('=' * 104)
    say('P1：val 泄漏是否影响早停选择（best.pt 是按完整 val 选的）')
    say('=' * 104)
    for tag, _, _ in MODELS:
        floor = NOISE_VAL[tag]
        say()
        say(f'--- {tag}（val 子集噪声基准 {floor:.4f}）---')
        say(f'{"权重":<8}{"val_full":>10}{"val_leak":>10}{"val_clean":>10}'
            f'{"leak−clean":>12}')
        for who in ('best', 'last'):
            f_ = R[(tag, 'val_full', who)][0]
            l_ = R[(tag, 'val_leak', who)][0]
            c_ = R[(tag, 'val_clean', who)][0]
            say(f'{who:<8}{f_:>10.4f}{l_:>10.4f}{c_:>10.4f}{l_ - c_:>+12.4f}')
        db = (R[(tag, 'val_clean', 'best')][0] - R[(tag, 'val_clean', 'last')][0])
        dl = (R[(tag, 'val_leak', 'best')][0] - R[(tag, 'val_leak', 'last')][0])
        df = (R[(tag, 'val_full', 'best')][0] - R[(tag, 'val_full', 'last')][0])
        gb = (R[(tag, 'val_leak', 'best')][0] - R[(tag, 'val_clean', 'best')][0])
        gl = (R[(tag, 'val_leak', 'last')][0] - R[(tag, 'val_clean', 'last')][0])
        say()
        say(f'  best − last：val_full {df:+.4f} / val_leak {dl:+.4f} / val_clean {db:+.4f}')
        say(f'  leak − clean：best {gb:+.4f} / last {gl:+.4f}'
            f'（差 {gb - gl:+.4f}）')
        if abs(db) <= 1.5 * floor and abs(dl) <= 1.5 * floor:
            say(f'  ⇒ Δ 全部 ≤ 1.5×噪声基准（{1.5 * floor:.4f}）⇒ **不可分辨**。')
            say('     ⚠ 这是**功效不足**，不是"选择无偏"。本检验测不出 1~2 点的选择偏置。')
        elif dl > 1.5 * floor and abs(db) <= 1.5 * floor:
            say('  ⇒ best 在 val_leak 上占优、在 val_clean 上不占优'
                ' ⇒ **早停选的可能是"在泄漏图上好的模型"**，必须声明。')
        elif db > 1.5 * floor:
            say('  ⇒ best 在 val_clean 上**仍占优** ⇒ 早停选择不依赖泄漏，选择稳。')
        else:
            say('  ⇒ 混合情形，按上表两个 Δ 如实写。')
        if abs(gb - gl) <= 1.5 * floor:
            say(f'  ⇒ 机制判据：best 与 last 的 leak−clean 接近（差 {gb - gl:+.4f} '
                f'≤ 1.5×噪声 {1.5 * floor:.4f}）')
            say('     ⇒ 泄漏抬升在各轮之间近似不变 ⇒ 对 argmax 是常数偏移 ⇒ '
                '**早停不受泄漏影响**（此判据不依赖子集样本量，比上面的检验更可靠）。')
        else:
            say(f'  ⇒ 机制判据：best 与 last 的 leak−clean 差 {gb - gl:+.4f} 已超噪声，')
            say('     说明泄漏抬升**随训练变化** ⇒ 可能把选择推向更晚/更早的轮次，需细查。')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
