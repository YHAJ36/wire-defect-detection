# -*- coding: utf-8 -*-
"""注意力臂批次评估：噪声地板（baseline×3 种子）+ CBAM+BN×3 + Triplet×3。

口径（**逐项照抄**项目既有约定，见 code/tools/eval/eval_test_640.py 与
code/upscale/eval_batch1.py，不得自创）：
    YOLO(best.pt) 每次全新加载 → .val(data=dataset_final_bbox/data.yaml,
    split=<val|test>, imgsz=640, batch=8, workers=0, device=0, plots=False)
    评估 imgsz 必须等于训练 imgsz（本批全 640）。
    best.pt 由 ultralytics 在 **val** 上按 fitness（0.1·mAP50+0.9·mAP50-95）选出。

判读口径（用户 2026-10-15 拍板，见 WORKLOG A.9.16 第七节 / A.9.17 第六节）
    **噪声地板** = 3 个 baseline 种子的 test mAP50 区间 [min, max]。
    臂同样跑 3 个种子 → 得臂的区间。
        **两区间错开**（臂 min > 地板 max，或臂 max < 地板 min）→ 真实效应
        **两区间重叠**                                    → 不可分，诚实记负
    ⚠ 单种子时**不能判**（A.9.17 实测：两条臂各只在一个 split 越界、互为镜像 ⇒ 两次噪声）。
      故本脚本对 n<2 的臂只印数、不下判读。
    ⚠ **test 与 val 必须一起读**：只在一边错开、另一边重叠，不构成真实效应。

用法：
    D:/yolo-env/Scripts/python.exe D:/2/code/tools/eval/eval_att_640.py
    ... 指定 run： ... eval_att_640.py --out=D:/tmp/x.txt nf_v11_640_s43
"""
import sys
from pathlib import Path

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)          # CBAM/Triplet 权重反序列化需要

from ultralytics import YOLO

DATA = 'D:/2/dataset_final_bbox/data.yaml'
RUNS = Path('D:/2/exp_upscale/runs')
OUT = Path('D:/2/exp_upscale/logs/eval_att_640.txt')

FLOOR = ['nf_v11_640_s42', 'nf_v11_640_s43', 'nf_v11_640_s44']
ARMS = {
    'CBAM+BN@neck': ['att_v11_640_cbam_bn_s42', 'att_v11_640_cbam_bn_s43',
                     'att_v11_640_cbam_bn_s44'],
    'Triplet@neck': ['att_v11_640_triplet_s42', 'att_v11_640_triplet_s43',
                     'att_v11_640_triplet_s44'],
}
IMGSZ = 640

# 归档参照（**不参与判读**，只印出来看：那是 workers=4 的旧管线）
ARCHIVE = dict(name='runs/wire_v11_640 (归档, workers=4)', map50=0.8992,
               map=0.5286, p=0.8887, r=0.8264)
CLASSES = ('break', 'thunderbolt', 'wear')


def eval_one(run, split):
    w = RUNS / run / 'weights' / 'best.pt'
    if not w.exists():
        return None
    m = YOLO(str(w))                       # 每次全新加载，避免复用串味
    r = m.val(data=DATA, split=split, imgsz=IMGSZ, batch=8, workers=0, device=0,
              project=str(RUNS), name=f'{run}_{split}', exist_ok=True,
              verbose=False, plots=False)
    b = r.box
    names = r.names                       # dict {0:'break',...}，不是 list
    per_class = {names[int(c)]: (float(b.ap50[j]), float(b.ap[j]))
                 for j, c in enumerate(b.ap_class_index)}
    return dict(map50=float(b.map50), map=float(b.map),
                p=float(b.mp), r=float(b.mr), per_class=per_class)


def rng(vals):
    """区间统计。返回 (lo, hi, mean, n)。"""
    return min(vals), max(vals), sum(vals) / len(vals), len(vals)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    runs = sys.argv[1:]
    out_path = OUT
    if runs and runs[0].startswith('--out='):        # 冒烟用：换个输出文件
        out_path = Path(runs[0].split('=', 1)[1])
        runs = runs[1:]
    if not runs:
        runs = FLOOR + [r for lst in ARMS.values() for r in lst]
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    say('=' * 104)
    say('注意力臂批次评估 —— 噪声地板(3 种子) + CBAM+BN(3 种子) + Triplet(3 种子)  @640')
    say('口径：探索阶段，**只报方向与幅度，不下显著性结论**；test/val 必须一起读')
    say('=' * 104)
    say(f'{"run":<30}{"split":>6}{"mAP50":>9}{"mAP50-95":>10}{"P":>8}{"R":>8}')
    say('-' * 104)

    res = {}
    for run in runs:
        got = {}
        for split in ('val', 'test'):
            m = eval_one(run, split)
            if m is None:
                say(f'{run:<30}{split:>6}   ❌ 缺权重 best.pt')
                break
            got[split] = m
            say(f'{run:<30}{split:>6}{m["map50"]:>9.4f}{m["map"]:>10.4f}'
                f'{m["p"]:>8.4f}{m["r"]:>8.4f}')
        if 'test' in got:
            res[run] = got
    say('-' * 104)
    say(f'{ARCHIVE["name"]:<30}{"test":>6}{ARCHIVE["map50"]:>9.4f}'
        f'{ARCHIVE["map"]:>10.4f}{ARCHIVE["p"]:>8.4f}{ARCHIVE["r"]:>8.4f}   ← 仅参照，不参与判读')

    # ---------- 地板 ----------
    fl = [r for r in FLOOR if r in res]
    flo = frng = fval = None
    say()
    say('【噪声地板】640 baseline × 3 种子（同一脚本、workers=0、仅 seed 不同）')
    if len(fl) >= 2:
        tv = [res[r]['test']['map50'] for r in fl]
        vv = [res[r]['val']['map50'] for r in fl]
        flo = rng(tv)
        fval = rng(vv)
        for r in fl:
            say(f'  {r:<24} test {res[r]["test"]["map50"]:.4f}   '
                f'val {res[r]["val"]["map50"]:.4f}')
        say(f'  → test 区间 [{flo[0]:.4f}, {flo[1]:.4f}]  跨度 {(flo[1]-flo[0])*100:.2f} pt'
            f'   均值 {flo[2]:.4f}   n={flo[3]}')
        say(f'  → val  区间 [{fval[0]:.4f}, {fval[1]:.4f}]  跨度 {(fval[1]-fval[0])*100:.2f} pt'
            f'   均值 {fval[2]:.4f}   n={fval[3]}')
    else:
        say(f'  样本不足（{len(fl)}），无法估计地板。')

    # ---------- 臂 vs 地板 ----------
    say()
    say('【臂 vs 噪声地板】判据：臂区间与地板区间**错开**才算真实效应；重叠 ⇒ 不可分')
    if flo is not None:
        say(f'{"臂":<16}{"n":>3}{"test 区间":>24}{"Δ vs 地板均值":>16}'
            f'{"test 判读":>20}{"val 判读":>20}')
        say('-' * 104)
        for label, names in ARMS.items():
            have = [r for r in names if r in res]
            if not have:
                continue
            tv = [res[r]['test']['map50'] for r in have]
            vv = [res[r]['val']['map50'] for r in have]
            alo, ahi, amu, n = rng(tv)
            if n < 2:
                say(f'{label:<16}{n:>3}{"  仅单种子（%.4f）" % alo:>24}'
                    f'{(amu - flo[2]) * 100:>+16.2f}{"不可判（需≥2 种子）":>20}{"":>20}')
                continue
            verdict_t = ('★ 错开 → 真实效应' if ahi < flo[0] or alo > flo[1]
                         else '· 重叠 → 不可分')
            verdict_v = ('★ 错开 → 真实效应' if max(vv) < fval[0] or min(vv) > fval[1]
                         else '· 重叠 → 不可分')
            if verdict_t.startswith('★') and verdict_v.startswith('★'):
                verdict_t += '（两 split 一致）'
            say(f'{label:<16}{n:>3}{f"[{alo:.4f}, {ahi:.4f}]":>24}'
                f'{(amu - flo[2]) * 100:>+16.2f}{verdict_t:>20}{verdict_v:>20}')
        say()
        say('  ⚠ 只在一边错开、另一边重叠 ⇒ **不构成**真实效应（A.9.17 的镜像情形）。')
    else:
        say('  （地板未就绪，跳过）')

    # ---------- per-class ----------
    say()
    say('【per-class mAP50 / mAP50-95（test）】')
    say(f'{"run":<30}' + ''.join(f'{c:>20}' for c in CLASSES))
    say('-' * 104)
    for run in runs:
        if run not in res:
            continue
        pc = res[run]['test']['per_class']
        cells = ''.join(f'{pc[c][0]:>10.4f}/{pc[c][1]:<9.4f}' if c in pc
                        else f'{"—":>20}' for c in CLASSES)
        say(f'{run:<30}{cells}')
    say('注：wear 仅 18 图 / 64 实例 —— 光地板内部 wear 就晃 4.6 pt，单类读数不足以支撑结论。')

    say()
    say('【口径声明 —— 引用这些数字时必须一并带上】')
    say('  探索阶段：地板 3 种子、臂 3 种子；只报方向与幅度，未做显著性检验。')
    say('  判读只看**区间是否错开**；两 split 结论不一致时按"不可分"处理。')
    say('  地板是**本管线**（workers=0）的种子噪声，与归档 runs/wire_v11_640（workers=4）不可混用。')
    say('  本批数字不得进入 v_final 或任何交付形态。')
    say('=' * 104)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n已追加写入 {out_path}')


if __name__ == '__main__':
    main()
