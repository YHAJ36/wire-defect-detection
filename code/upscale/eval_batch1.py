# -*- coding: utf-8 -*-
"""第一批提升实验的评估（val 选型 + test 一次），产物落 exp_upscale/。

口径（**逐项照抄**项目既有约定，见 code/tools/eval/eval_test_896.py 与
eval_best_vs_last.py，不得自创）：
    YOLO(best.pt) 每次全新加载 → .val(data=dataset_final_bbox/data.yaml,
    split=..., imgsz=<该组的训练分辨率>, batch=8, workers=0, device=0,
    plots=False)               ← 评估 imgsz 必须等于训练 imgsz
    best.pt 由 ultralytics 在 **val** 上按 fitness 选出（fitness=0.1·mAP50+0.9·mAP50-95），
    这就是本项目的"val 选型"；此处只把它在 val 上的读数复核一遍，再在 test 上评一次。

基线（test，179 图 283 实例，2026-10-12 读数，来源 logs/eval_896.log）：
    wire_v11_896   mAP50 0.9148   mAP50-95 0.5362   P 0.9401   R 0.8212
      break 0.870/0.449   thunderbolt 0.962/0.620   wear 0.912/0.539

判读口径（**探索阶段**，用户 2026-10-14 钉死，见 WORKLOG A.9.7）：
    只报"相对基线的方向 + 幅度"，**不下显著/不显著结论**（单种子，未做显著性检验）。
    苗头阈值（用于决定"要不要补种子"这个**动作**，不是统计显著性阈值）：
        ΔmAP50 ≥ +0.5pt           → 苗头（值得补种子确认）
        −0.5pt < ΔmAP50 < +0.5pt  → 未观察到效应（不补种子）
        ΔmAP50 ≤ −0.5pt           → 反向（值得注意，可能是过拟合信号）

用法：
    D:/yolo-env/Scripts/python.exe D:/2/code/upscale/eval_batch1.py wire_v11_1024 1024
    ... 一次传多组：  ... wire_v11_1024 1024 wire_v11m_896_b4 896
"""
import sys
from pathlib import Path

if 'D:/2/code/yolov11' not in sys.path:
    sys.path.insert(0, 'D:/2/code/yolov11')      # Triplet 组反序列化需要

from ultralytics import YOLO  # noqa: E402

DATA = 'D:/2/dataset_final_bbox/data.yaml'
RUNS = Path('D:/2/exp_upscale/runs')
OUT = Path('D:/2/exp_upscale/logs/eval_batch1.txt')

BASELINE_NAME = 'wire_v11_896'
BASELINE = dict(map50=0.9148, map=0.5362, p=0.9401, r=0.8212)
THRESH = 0.005        # 0.5pt


def eval_one(run, imgsz, split):
    """一次全新加载 + 一次 val。返回 metrics dict。"""
    w = RUNS / run / 'weights' / 'best.pt'
    if not w.exists():
        return None
    m = YOLO(str(w))                      # 每次全新加载，避免复用串味
    r = m.val(data=DATA, split=split, imgsz=imgsz, batch=8, workers=0, device=0,
              project=str(RUNS), name=f'{run}_{split}', exist_ok=True,
              verbose=False, plots=False)
    b = r.box
    # ⚠ r.names 是 dict {0:'break',1:'thunderbolt',2:'wear'}，不是 list。
    #   曾经写成 enumerate(r.names) ⇒ 拿到的是**键 0/1/2 而非名字**，per-class 表全打「—」。
    #   ap50/ap 长度 == len(ap_class_index)，与 ap_class_index **按位置对齐**，
    #   故必须用 ap_class_index 去查名字，不能默认 0..nc-1。
    names = r.names
    per_class = {names[int(c)]: (float(b.ap50[j]), float(b.ap[j]))
                 for j, c in enumerate(b.ap_class_index)}
    return dict(map50=float(b.map50), map=float(b.map),
                p=float(b.mp), r=float(b.mr),
                per_class=per_class)


def fmt_verdict(d):
    if d >= THRESH:
        return '苗头（值得补种子确认）'
    if d <= -THRESH:
        return '反向（值得注意，可能是过拟合信号）'
    return '未观察到效应（不补种子）'


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    args = sys.argv[1:]
    if not args or len(args) % 2:
        print(__doc__)
        print('❌ 用法：eval_batch1.py <run名> <imgsz> [<run名> <imgsz> ...]')
        return
    jobs = [(args[i], int(args[i + 1])) for i in range(0, len(args), 2)]

    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    say('=' * 100)
    say('第一批提升实验评估 —— val 选型复核 + test 单次评估')
    say('口径：探索阶段，单种子，**只报方向与幅度，不下显著性结论**')
    say('=' * 100)
    say(f'{"run":<26}{"imgsz":>6}{"split":>7}{"mAP50":>10}{"mAP50-95":>11}'
        f'{"P":>9}{"R":>9}')
    say('-' * 100)
    say(f'{BASELINE_NAME + " (基线)":<26}{896:>6}{"test":>7}'
        f'{BASELINE["map50"]:>10.4f}{BASELINE["map"]:>11.4f}'
        f'{BASELINE["p"]:>9.4f}{BASELINE["r"]:>9.4f}')

    rows = []
    for run, imgsz in jobs:
        got = {}
        for split in ('val', 'test'):
            m = eval_one(run, imgsz, split)
            if m is None:
                say(f'{run:<26}{imgsz:>6}{split:>7}   ❌ 缺权重 best.pt')
                break
            got[split] = m
            say(f'{run:<26}{imgsz:>6}{split:>7}{m["map50"]:>10.4f}{m["map"]:>11.4f}'
                f'{m["p"]:>9.4f}{m["r"]:>9.4f}')
        if 'test' not in got:
            continue
        t = got['test']
        d50 = t['map50'] - BASELINE['map50']
        d95 = t['map'] - BASELINE['map']
        rows.append((run, imgsz, t, d50, d95, fmt_verdict(d50)))

    if rows:
        say('-' * 100)
        say('\n【相对基线的幅度（test，探索口径）】')
        say(f'{"run":<26}{"ΔmAP50(pt)":>13}{"ΔmAP50-95(pt)":>16}{"ΔP(pt)":>11}{"ΔR(pt)":>11}  判读')
        say('-' * 100)
        for run, imgsz, t, d50, d95, v in rows:
            say(f'{run:<26}{d50 * 100:>+13.2f}{d95 * 100:>+16.2f}'
                f'{(t["p"] - BASELINE["p"]) * 100:>+11.2f}'
                f'{(t["r"] - BASELINE["r"]) * 100:>+11.2f}  {v}')

        say('\n【per-class mAP50 / mAP50-95（test）】')
        say(f'{"run":<26}{"break":>20}{"thunderbolt":>20}{"wear":>20}')
        say('-' * 100)
        base_pc = {'break': (0.870, 0.449), 'thunderbolt': (0.962, 0.620),
                   'wear': (0.912, 0.539)}
        say(f'{"基线 " + BASELINE_NAME:<26}'
            + ''.join(f'{base_pc[c][0]:>10.4f}/{base_pc[c][1]:<9.4f}'
                      for c in ('break', 'thunderbolt', 'wear')))
        for run, imgsz, t, d50, d95, v in rows:
            cells = ''
            for c in ('break', 'thunderbolt', 'wear'):
                if c in t['per_class']:
                    a, b = t['per_class'][c]
                    cells += f'{a:>10.4f}/{b:<9.4f}'
                else:
                    cells += f'{"—":>20}'
            say(f'{run:<26}{cells}')
        say('\n注：wear 仅 18 图 / 64 实例，单类噪声大；break 为最大类（150 实例）。')

        say('\n【口径声明 —— 引用这些数字时必须一并带上】')
        say('  探索阶段，单种子（seed=42），未做多种子重复与显著性检验。')
        say('  只报方向与幅度；"苗头"是决定是否补种子的**动作阈值**（±0.5pt），')
        say('  不是统计显著性阈值。本批数字不得进入 v_final 或任何交付形态，')
        say('  补种子确认之前只能作为内部选型依据。')

    say('=' * 100)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n已追加写入 {OUT}')


if __name__ == '__main__':
    main()
