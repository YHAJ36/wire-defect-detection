# -*- coding: utf-8 -*-
"""YOLO26s @896 评估 —— 与 `eval_att_batch2.py` **逐项同口径**，但必须用**新环境**解释器。

为什么另起一个脚本（不直接拿 eval_att_batch2.py 跑）
    `eval_att_batch2.py` 用 `D:/yolo-env` 的 ultralytics **8.3.180**，它**不认识 YOLO26**
    （yolo26 的 yaml 有 `end2end` / `reg_max=1`，8.3.180 无此代码路径），加载即失败。
    故 YOLO26 的评估只能用 `D:/yolo26-env` 的 **8.4.178**。

★★ 由此产生一条**必须随数字一起引用的口径事实**（判读时不可省略）
    YOLO26 的数字由 **ultralytics 8.4.178** 产出，v11 地板/臂的数字由 **8.3.180** 产出。
    **评测库版本不同** ⇒ 即使两者 mAP 数值接近，也不能断言"两代模型等价"——
    差值里混着一个无法分离的版本项（与 A.9.20 记的"续训改随机流 vs AMP/cuDNN 残余
    不确定性无法分离"是同一类诚实边界：能报差，不能报差的归因）。

    本条**只影响跨代比较**；YOLO26 三个种子之间（同一版本）的比较不受影响。

口径（逐项照抄 eval_att_batch2.py，不另立标准）
    YOLO(best.pt) 每次全新加载 → .val(data=dataset_final_bbox/data.yaml, split=<val|test>,
    imgsz=**896**（与训练同）, batch=8, workers=0, device=0, plots=False)
    best.pt 由 ultralytics 在 val 上按 fitness 选出（0.1·mAP50 + 0.9·mAP50-95）。
    判读：**test 为主判据，val 为一致性检查**（best.pt 在 val 上挑选 ⇒ val 带选择偏置）。

    ⚠ 本次 val/test 走的是 **one2many 头 + NMS**（`engine/validator.py:167` 置
      `model.end2end = (args.nms is False)`，`nms` 默认 None ⇒ False），
      **不是** YOLO26 的无 NMS 端到端模式。见 WORKLOG A.9.22 §二。
    ⚠ YOLO26 的 `dfl=1.5` 加权的是**按图像尺寸归一化的 L1 项**（`reg_max=1` ⇒
      `use_dfl=False`），与 v11 的 DFL 项**同名不同义**。同上。

用法：
    D:/yolo26-env/Scripts/python.exe D:/2/code/tools/eval/eval_yolo26_896.py nf_yolo26_896_s42 ...
    D:/yolo26-env/Scripts/python.exe D:/2/code/tools/eval/eval_yolo26_896.py      # 评全部 nf_yolo26_896_*
已评过的 run（val+test 齐全）自动跳过 ⇒ 可反复调用，不重复算。
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, 'D:/2/code/yolov11')

from ultralytics import YOLO  # noqa: E402

DATA = 'D:/2/dataset_final_bbox/data.yaml'
RUNS = Path('D:/2/exp_upscale/runs')
EVAL_DIR = Path('D:/2/exp_upscale/eval')
TSV = Path('D:/2/exp_upscale/logs/batch3_eval.tsv')
LOG = Path('D:/2/exp_upscale/logs/eval_yolo26_896.txt')
BATCH2_TSV = Path('D:/2/exp_upscale/logs/batch2_eval.tsv')   # 只读：地板数字的来源

IMGSZ = 896
EPOCHS_REQUIRED = 60
# v11 @896 地板（3 个 baseline 种子）—— 本脚本只从 batch2_eval.tsv **现算**，
# 下面这行是**预登记的期望值**，仅用于交叉核对；不一致就打印警告，不静默采用。
FLOOR_RUNS = ['nf_v11_896_s42', 'nf_v11_896_s43', 'nf_v11_896_s44']
FLOOR_EXPECT_MAP50 = (0.9044, 0.9122)

HEADER = ['run', 'split', 'imgsz', 'mAP50', 'mAP50-95', 'P', 'R']


def csv_best_val_map50(run):
    """run 自己的 results.csv 里 val mAP50 的**峰值**。

    用途：作 TSV 行的**交叉验证基准**。`best.pt` 是按 fitness(0.1·mAP50 + 0.9·mAP50-95)
    从 60 轮里挑的，故其 val mAP50 与"val mAP50 的峰值"通常差不到 1 pt。
    若 TSV 某行的 val mAP50 与此值相差**远超**这个量级，那行必是"训练中途的检查点"
    评出来的（中途模型没训完 ⇒ 必然低估），应判无效。
    """
    # ★ 局部变量名不可叫 csv：会**遮蔽**模块顶部的 `import csv`，
    #   于是下面 `csv.DictReader` 变成 Path.DictReader ⇒ AttributeError。
    #   这个坑只会在这条"踩过中途检查点"的路径上触发（即恰好是必须能跑通的那条路径）。
    p = RUNS / run / 'results.csv'
    if not p.exists():
        return None
    best = None
    with p.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            v = row.get('metrics/mAP50(B)')
            if v in (None, ''):
                continue
            try:
                v = float(v)
            except ValueError:
                continue
            best = v if best is None else max(best, v)
    return best


def row_is_stale(row):
    """TSV 行是否与 run 自己的 results.csv 矛盾 ⇒ 是中途检查点评出来的（无效）。

    为什么必须这样验（2026-10-16 事故，THE reason this function exists）
        第一版只用"run 是否跑满 60 轮"判有效性。但中途写进去的假行有个恶劣性质：
        **run 一旦跑满，那行就从"未跑满所以无效"变成"看起来有效"**，于是可重入判据
        认定"s42 已评过"而**跳过重评** —— 假数字 0.7472 被永久固化成 s42 的结果。
        教训：有效性不能用"外部状态"（跑满没跑满）推断，必须**用行自己声称的数值
        去对照 run 自己的训练记录**。AI 规则 8 的同一类：状态变了不等于内容有效。
    """
    if row['split'] != 'val':
        return False
    ref = csv_best_val_map50(row['run'])
    if ref is None:
        return False
    try:
        got = float(row['mAP50'])
    except (TypeError, ValueError):
        return True
    return abs(got - ref) > 0.05


def epochs_of(run):
    csv = RUNS / run / 'results.csv'
    if not csv.exists():
        return 0
    with csv.open(encoding='utf-8') as f:
        return max(0, sum(1 for _ in f) - 1)


def run_finished(run):
    """★ 硬闸门：只有跑满 EPOCHS_REQUIRED 轮的 run 才允许被评估。

    为什么必须挡（2026-10-16 实测踩到，本节即为事故记录）
        ultralytics 在**每一轮 val 变好时**就写 `weights/best.pt`，所以"best.pt 存在"
        **完全不能**证明训练结束 —— 一个跑到第 11 轮的 run 也有 best.pt。
        本脚本第一版只用"best.pt 存在"作闸门，结果自检时把一个**训练中途的检查点**
        评了，并把 0.7472 这种中途数字写进了 batch3_eval.tsv（真值约 0.90+）。
        中途检查点低估是必然的（模型还没训完），故这类行**是错的、必须忽略**。
        教训与 AI 规则 8 同类：**"文件存在"不等于"内容有效"**。
    """
    return epochs_of(run) >= EPOCHS_REQUIRED and (RUNS / run / 'weights' / 'best.pt').exists()



def load_tsv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def append_tsv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open('a', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        if new:
            w.writerow(header)
        w.writerows(rows)


def discover():
    """runs 下全部 nf_yolo26_896_*（排除探针壳）"""
    return sorted(p.name for p in RUNS.glob('nf_yolo26_896_*')
                  if p.is_dir() and 'probe' not in p.name)


def eval_one(run):
    w = RUNS / run / 'weights' / 'best.pt'
    if not w.exists():
        print(f'  !! {run}：缺 best.pt，跳过')
        return None
    rows = []
    for split in ('val', 'test'):
        m = YOLO(str(w))                    # 每次全新加载，避免复用串味
        n_par = sum(p.numel() for p in m.model.parameters())
        r = m.val(data=DATA, split=split, imgsz=IMGSZ, batch=8, workers=0, device=0,
                  project=str(EVAL_DIR), name=f'{run}_{split}', exist_ok=True,
                  verbose=False, plots=False)
        b = r.box
        rows.append([run, split, IMGSZ, f'{float(b.map50):.6f}', f'{float(b.map):.6f}',
                     f'{float(b.mp):.6f}', f'{float(b.mr):.6f}'])
        print(f'  {run:<26}{split:>5}  imgsz={IMGSZ}  params={n_par:,}  '
              f'mAP50={float(b.map50):.4f}  mAP50-95={float(b.map):.4f}  '
              f'P={float(b.mp):.4f}  R={float(b.mr):.4f}')
    return rows


def rng(v):
    return min(v), max(v), sum(v) / len(v), len(v)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    targets = [a for a in sys.argv[1:] if not a.startswith('--')] or discover()
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    # ---------- 评估（可重入 + 跑满才许评） ----------
    ready = [r for r in targets if run_finished(r)]
    pending = [r for r in targets if not run_finished(r)]
    for r in pending:
        say(f'>>> 跳过 {r}：训练未跑满（{epochs_of(r)}/{EPOCHS_REQUIRED} 轮）。'
            f'中途的 best.pt 只是检查点，评它得到的是低估的假数字。')

    # 有效性判定：先剔掉"与 run 自己的 results.csv 矛盾"的行（中途检查点评的）。
    # 按 **run** 剔除而非按 (run,split)：同一个中途检查点的 val/test 两行都不可信。
    rows_all = load_tsv(TSV)
    bad_runs = set()
    for r in rows_all:
        if row_is_stale(r):
            bad_runs.add(r['run'])
    if bad_runs:
        say('!! 下列 run 在 TSV 里的行与它自己的 results.csv 矛盾 ⇒ 判为"中途检查点"行，'
            '无效，**重新评估**：')
        for run in sorted(bad_runs):
            ref = csv_best_val_map50(run)
            got = [r['mAP50'] for r in rows_all if r['run'] == run and r['split'] == 'val']
            say(f'     {run}：TSV val mAP50={got[-1] if got else "?"}  '
                f'但 results.csv 的 val mAP50 峰值={ref:.4f}')

    done = {(r['run'], r['split']) for r in rows_all
            if run_finished(r['run']) and r['run'] not in bad_runs}
    todo = [r for r in ready if not {'val', 'test'} <= {s for (rr, s) in done if rr == r}]
    if todo:
        say(f'>>> 待评 {len(todo)} 个 run：{", ".join(todo)}')
        for run in todo:
            got = eval_one(run)
            if got:
                append_tsv(TSV, HEADER, got)
    else:
        say('>>> 无待评 run（TSV 已齐）')

    # 重复行提示：同一 (run, split) 有多行时，判读一律取**最后一行**（下面的 lut 后行覆盖前行）。
    _rows_all = load_tsv(TSV)
    seen, dup = set(), set()
    for r in _rows_all:
        k = (r['run'], r['split'])
        if k in seen:
            dup.add(k)
        seen.add(k)
    if dup:
        say(f'!! TSV 中以下 (run,split) 有重复行（先前的无效行未删，按纪律只追加）；'
            f'判读取**最后一行**：{", ".join(f"{a}/{b}" for a, b in sorted(dup))}')

    # ---------- 地板：从 batch2_eval.tsv 现算（不硬编码） ----------
    b2 = {(r['run'], r['split']): r for r in load_tsv(BATCH2_TSV)}
    fl50 = [float(b2[(r, 'test')]['mAP50']) for r in FLOOR_RUNS if (r, 'test') in b2]
    fl95 = [float(b2[(r, 'test')]['mAP50-95']) for r in FLOOR_RUNS if (r, 'test') in b2]
    if len(fl50) < 3:
        say(f'!! v11 896 地板在 {BATCH2_TSV} 里只找到 {len(fl50)} 个种子，无法比较')
        fl50 = fl95 = []
    lo50, hi50, mu50, n50 = rng(fl50) if fl50 else (0, 0, 0, 0)
    lo95, hi95, mu95, n95 = rng(fl95) if fl95 else (0, 0, 0, 0)

    # 后追加的行覆盖先前的行（dict 推导按文件顺序，后者胜）⇒
    # 同一 (run, split) 被评过两次时（如中途检查点那次），**以最后一次为准**。
    rows = load_tsv(TSV)
    lut = {(r['run'], r['split']): r for r in rows}

    say()
    say('=' * 104)
    say('YOLO26s @896 —— 与 v11s @896 地板对照')
    say('=' * 104)
    say(f'{"run":<26}{"split":>6}{"mAP50":>9}{"mAP50-95":>10}{"P":>8}{"R":>8}')
    say('-' * 104)
    for run in ready:
        for split in ('val', 'test'):
            r = lut.get((run, split))
            if not r:
                say(f'{run:<26}{split:>6}{"  缺":>9}')
                continue
            say(f'{run:<26}{split:>6}{float(r["mAP50"]):>9.4f}{float(r["mAP50-95"]):>10.4f}'
                f'{float(r["P"]):>8.4f}{float(r["R"]):>8.4f}')

    if pending:
        say()
        say(f'（下列 run **不判读**：训练未跑满 —— {", ".join(pending)}）')

    if fl50:
        say()
        say(f'【v11s @896 噪声地板】{len(fl50)} 种子（读出源：{BATCH2_TSV.name}）')
        for r in FLOOR_RUNS:
            if (r, 'test') in b2:
                say(f'  {r:<20} test mAP50 {float(b2[(r,"test")]["mAP50"]):.4f}   '
                    f'mAP50-95 {float(b2[(r,"test")]["mAP50-95"]):.4f}')
        say(f'  → mAP50    test 区间 [{lo50:.4f}, {hi50:.4f}]  跨度 {(hi50-lo50)*100:.2f} pt'
            f'   均值 {mu50:.4f}')
        say(f'  → mAP50-95 test 区间 [{lo95:.4f}, {hi95:.4f}]  跨度 {(hi95-lo95)*100:.2f} pt'
            f'   均值 {mu95:.4f}')
        if abs(lo50 - FLOOR_EXPECT_MAP50[0]) > 5e-4 or abs(hi50 - FLOOR_EXPECT_MAP50[1]) > 5e-4:
            say(f'  ★ 警告：现算地板 [{lo50:.4f}, {hi50:.4f}] 与预登记 '
                f'[{FLOOR_EXPECT_MAP50[0]:.4f}, {FLOOR_EXPECT_MAP50[1]:.4f}] 不一致，'
                f'判读前先查 {BATCH2_TSV.name}')
        else:
            say('  （与 A.9.20 预登记的地板一致）')

        say()
        say('【逐 run 判读：两个指标分开看，不合并】')
        say(f'  {"run":<26}{"test mAP50":>12}{"Δ50":>9}{"test mAP50-95":>14}{"Δ95":>9}'
            f'{"50 判读":>16}{"95 判读":>16}')
        for run in ready:
            r = lut.get((run, 'test'))
            if not r:
                continue
            t50, t95 = float(r['mAP50']), float(r['mAP50-95'])
            d50, d95 = (t50 - mu50) * 100, (t95 - mu95) * 100
            j50 = '★错开' if (t50 > hi50 or t50 < lo50) else '·重叠'
            j95 = '★错开' if (t95 > hi95 or t95 < lo95) else '·重叠'
            say(f'  {run:<26}{t50:>12.4f}{d50:>+9.2f}{t95:>14.4f}{d95:>+9.2f}'
                f'{j50:>16}{j95:>16}')
            if j50 != j95:
                say(f'    ★ {run}：mAP50 与 mAP50-95 **方向不一致**'
                    f'（50 {j50.strip("·★")} / 95 {j95.strip("·★")}）——'
                    f'即"找全"与"框准"在本 run 上不同步，判读必须分开报')
        say()
        say('  ⚠ 单种子不是判读（n<2）。YOLO26 s42 只给方向，等 s43/s44 到齐再按区间判。')
        say('  ⚠ 跨代比较混入"评测库版本项"（8.4.178 vs 8.3.180），差值可报、归因不可报。')

    say()
    say('【口径声明 —— 引用这些数字时必须一并带上】')
    say('  探索阶段，未做显著性检验；本批数字不得进入 v_final 或任何交付形态。')
    say('  YOLO26 数字由 ultralytics 8.4.178 产出；v11 地板/臂由 8.3.180 产出 ⇒ 跨代不可归因。')
    say('  val/test 走 one2many + NMS（非 YOLO26 的无 NMS 端到端模式）。')
    say('  YOLO26 的 dfl 权重作用于 L1 项（reg_max=1），与 v11 的 DFL 同名不同义。')
    say('  评估 imgsz = 训练 imgsz = 896；跨分辨率数字不可直接相比。')
    say('=' * 104)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n已追加写入 {LOG}')


if __name__ == '__main__':
    main()
