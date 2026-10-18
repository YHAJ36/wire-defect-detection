# -*- coding: utf-8 -*-
"""第二批（896 地板/臂 + 补齐的 640 种子）评估 —— 可重入、机器可读、按 run 名自识分辨率。

为什么另起一个脚本（不扩写 eval_att_640.py）
    `eval_att_640.py` 把 `IMGSZ = 640` 与 run 名清单**都写死**。第二批的主线在 896，
    若拿它评 896 的权重而 imgsz 传 640，会得到**分辨率不匹配的假数字**——
    这正是"跑完 13.5h 才发现数字不可用"那类事故。故此处按 run 名里的 `_640_` / `_896_`
    自动决定评估 imgsz，并落 TSV 便于中断后重入与事后判读。

口径（**逐项照抄**项目既有约定；见 code/tools/eval/eval_test_640.py、eval_att_640.py）
    YOLO(best.pt) 每次全新加载 → .val(data=dataset_final_bbox/data.yaml, split=<val|test>,
    imgsz=**训练时的 imgsz**, batch=8, workers=0, device=0, plots=False)
    best.pt 由 ultralytics 在 **val** 上按 fitness 选出（0.1·mAP50 + 0.9·mAP50-95）。

判读口径（沿用 A.9.17，并**在出数前预登记** val/test 不一致时的处理，2026-10-15）
    噪声地板 = 同分辨率 3 个 baseline 种子的 test mAP50 区间 [min, max]；
    臂 = 同分辨率 3 个种子的区间。**错开**（臂 min > 地板 max 或臂 max < 地板 min）⇒ 真实效应；
    **重叠** ⇒ 不可分，诚实记负。n<2 只印数、不下判读。

    ⚠ **val 与 test 不对等，不能各算一半**（关键，出数前定死）：
      · `best.pt` 是 ultralytics 在 **val** 上按 fitness 从 60 轮里挑出来的 ⇒
        val 是**选择集**，其 mAP 带向上的选择偏置，且偏置幅度随种子变化 ⇒
        val 的种子区间被**撑宽**、灵敏度低于 test。
      · **test 是留出集**，未参与任何选择 ⇒ **test 为主判据**，val 为一致性检查。
      · 故规则是：**test 区间错开 且 val 不反向矛盾** 才记真实效应；
        val 反向错开（臂在 val 明显更差、却在 test 更好）⇒ 判为噪声，不采信。
      · **两臂互相矛盾**（如 CBAM+BN 只在 test 错开、Triplet 只在 val 错开）⇒
        这不是"互为印证"，而是**两次独立噪声**；两臂**均记不可分**，不择一采信、
        不合并两臂来凑 n、不挑显示效应的那一边报。
    ⚠ 争议只能靠**加种子**（如 5 种子）解决，不能靠换分辨率或换 split 解决。

用法：
    D:/yolo-env/Scripts/python.exe D:/2/code/tools/eval/eval_att_batch2.py <run> [<run> ...]
    D:/yolo-env/Scripts/python.exe D:/2/code/tools/eval/eval_att_batch2.py --summary
    D:/yolo-env/Scripts/python.exe D:/2/code/tools/eval/eval_att_batch2.py --summary --tsv=D:/tmp/x.tsv
已评过的 run 若已在 TSV 里出现（val+test 齐全）则跳过 ⇒ 可反复调用，不重复算。
"""
import csv
import re
import sys
from pathlib import Path

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)          # CBAM / Triplet 权重反序列化需要

from ultralytics import YOLO  # noqa: E402

DATA = 'D:/2/dataset_final_bbox/data.yaml'
RUNS = Path('D:/2/exp_upscale/runs')
EVAL_DIR = Path('D:/2/exp_upscale/eval')      # 评估产物独立，不混进训练目录
TSV = Path('D:/2/exp_upscale/logs/batch2_eval.tsv')
# v2：per-class 行带 split 列（旧表只有 run/class，val/test 靠行序推断，见 A.9.19 §三）。
# 旧表保留不删；v2 由 regen_perclass_split.py 从旧表重建并逐组校验。
PC_TSV = Path('D:/2/exp_upscale/logs/batch2_eval_perclass_v2.tsv')
LOG = Path('D:/2/exp_upscale/logs/eval_att_batch2.txt')

CLASSES = ('break', 'thunderbolt', 'wear')

# 本批全量清单：地板 + 两条臂 × 两种分辨率 × 3 种子
FLOOR = ['nf_v11_640_s42', 'nf_v11_640_s43', 'nf_v11_640_s44',
         'nf_v11_896_s42', 'nf_v11_896_s43', 'nf_v11_896_s44']
ARMS = {
    'CBAM+BN@neck': ['att_v11_640_cbam_bn_s42', 'att_v11_640_cbam_bn_s43',
                     'att_v11_640_cbam_bn_s44',
                     'att_v11_896_cbam_bn_s42', 'att_v11_896_cbam_bn_s43',
                     'att_v11_896_cbam_bn_s44'],
    'Triplet@neck': ['att_v11_640_triplet_s42', 'att_v11_640_triplet_s43',
                     'att_v11_640_triplet_s44',
                     'att_v11_896_triplet_s42', 'att_v11_896_triplet_s43',
                     'att_v11_896_triplet_s44'],
}
ALL = FLOOR + [r for lst in ARMS.values() for r in lst]

HEADER = ['run', 'split', 'imgsz', 'mAP50', 'mAP50-95', 'P', 'R']
PC_HEADER = ['run', 'split', 'class', 'mAP50', 'mAP50-95']


def imgsz_of(run):
    """从 run 名里抠分辨率。抠不到 ⇒ 报错，绝不默认 640。"""
    m = re.search(r'_(640|896)_', run)
    if not m:
        raise SystemExit(f'❌ 无法从 run 名识别分辨率：{run}（名字里需含 _640_ 或 _896_）')
    return int(m.group(1))


# ================================================================
#            有效性判定（2026-10-17 补，修复 s42 型"陈旧行"缺陷）
# ================================================================
# 原实现直接把"TSV 里出现过 (run, split)"当作已评完，不做任何校验。缺陷在于：
#   ① ultralytics 在**每次 val 有提升**时就写 weights/best.pt，故"best.pt 存在"
#      不等于训练完成，中途评出来的行也能落进 TSV；
#   ② 更隐蔽的是：中途行在 run 跑完之后**会摇身一变"看起来有效"**（run 确实跑完了），
#      若只按外部状态（"跑完没有"）判有效性，重入时会永久跳过，坏数字被冻结。
# 正确做法：拿 TSV 行去对 **run 自己的 results.csv 里 val mAP50 的峰值**——
# 二者本应一致（同一 val、同一指标）；差 > 0.05 ⇒ 该行是中途检查点评的，无效。
# 阈值 0.05 的依据：本批 18 个已完成 run 的实测最大偏差仅 0.0145（复评 vs 峰值），
# 正常噪声够不到 0.05；而 s42 事故的偏差是 0.24。
EPOCHS_REQUIRED = 60


def epochs_of(run):
    """results.csv 的行数 − 1（ultralytics 每轮一行）。"""
    p = RUNS / run / 'results.csv'
    if not p.exists():
        return 0
    with p.open(encoding='utf-8', newline='') as f:
        # DictReader 本身已跳过表头 —— 别再 -1（+1 与 -1 差两轮，会让 60 轮的 run 报 59）。
        return max(0, sum(1 for _ in csv.DictReader(f)))


def csv_best_val_map50(run):
    """run 自己的 results.csv 里 val mAP50 的峰值。"""

    # ★ 局部变量名不可叫 csv：会遮蔽模块顶部的 `import csv`，
    #   于是下面 csv.DictReader 变成 Path.DictReader ⇒ AttributeError。
    p = RUNS / run / 'results.csv'
    if not p.exists():
        return None
    best = None
    with p.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            key = next((k for k in row if k and k.strip() == 'metrics/mAP50(B)'), None)
            if key is None:
                continue
            try:
                v = float(row[key])
            except (TypeError, ValueError):
                continue
            best = v if best is None else max(best, v)
    return best


def run_finished(run):
    return (epochs_of(run) >= EPOCHS_REQUIRED
            and (RUNS / run / 'weights' / 'best.pt').exists())


def row_is_stale(row):
    """TSV 行与该 run 自己的 results.csv 矛盾（差 > 0.05）⇒ 判为中途检查点行。"""
    if row.get('split') != 'val':
        return False
    ref = csv_best_val_map50(row['run'])
    try:
        got = float(row['mAP50'])
    except (TypeError, ValueError):
        return True
    return ref is not None and abs(got - ref) > 0.05


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


def eval_one(run):
    """评 val + test；返回 (主表行, 分类行) 或 None（缺权重）。"""
    w = RUNS / run / 'weights' / 'best.pt'
    if not w.exists():
        print(f'  ❌ {run}：缺 best.pt，跳过')
        return None
    imgsz = imgsz_of(run)
    rows, pc_rows = [], []
    for split in ('val', 'test'):
        m = YOLO(str(w))                    # 每次全新加载，避免复用串味
        r = m.val(data=DATA, split=split, imgsz=imgsz, batch=8, workers=0, device=0,
                  project=str(EVAL_DIR), name=f'{run}_{split}', exist_ok=True,
                  verbose=False, plots=False)
        b = r.box
        rows.append([run, split, imgsz, f'{float(b.map50):.6f}', f'{float(b.map):.6f}',
                     f'{float(b.mp):.6f}', f'{float(b.mr):.6f}'])
        names = r.names                      # dict {0:'break',...}
        for j, c in enumerate(b.ap_class_index):
            pc_rows.append([run, split, names[int(c)], f'{float(b.ap50[j]):.6f}',
                            f'{float(b.ap[j]):.6f}'])
        print(f'  {run:<32}{split:>5}  imgsz={imgsz}  mAP50={float(b.map50):.4f}  '
              f'mAP50-95={float(b.map):.4f}')
    return rows, pc_rows


def rng(vals):
    return min(vals), max(vals), sum(vals) / len(vals), len(vals)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    argv = sys.argv[1:]
    tsv = TSV
    for a in list(argv):
        if a.startswith('--tsv='):
            tsv = Path(a.split('=', 1)[1])
            argv.remove(a)
    # 带 split 列的新 schema（v2）；旧 *_perclass.tsv 只读留档，不再续写。
    pc_tsv = tsv.with_name(tsv.stem + '_perclass_v2' + tsv.suffix)
    summary_mode = '--summary' in argv
    # 不给 run 名 ⇒ 默认评**全清单**（可重入：已评过的自动跳过）。
    # 这样第一批已训好的 640 run 也会被补进 TSV，否则 640 判读会误报"样本不足"。
    targets = [a for a in argv if not a.startswith('--')] or ALL

    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    # ---------- 评估 ----------
    # 有效性判定：先按 run 剔除"与 run 自己的 results.csv 矛盾"的行（中途检查点评的）。
    # 按 run 而非按 (run,split) 剔除：同一个中途检查点的 val/test 两行都不可信。
    rows_so_far = load_tsv(tsv)
    bad_runs = {row['run'] for row in rows_so_far if row_is_stale(row)}
    if bad_runs:
        say('!! 下列 run 在 TSV 里的行与它自己的 results.csv 矛盾 ⇒ 判为"中途检查点"行，'
            '无效，重新评估：')
        for run in sorted(bad_runs):
            ref = csv_best_val_map50(run)
            ref_txt = f'{ref:.4f}' if ref is not None else '无 results.csv'
            got = [row['mAP50'] for row in rows_so_far
                   if row['run'] == run and row['split'] == 'val']
            say(f'     {run}：TSV val mAP50={got[-1] if got else "?"}  '
                f'results.csv 峰值={ref_txt}')
    done = {(row['run'], row['split']) for row in rows_so_far
            if run_finished(row['run']) and row['run'] not in bad_runs}
    if not summary_mode:
        todo = [r for r in targets
                if not {('val'), ('test')} <= {s for (rr, s) in done if rr == r}]
        if todo:
            say(f'>>> 待评 {len(todo)} 个 run：{", ".join(todo)}')
            for run in todo:
                got = eval_one(run)
                if got:
                    append_tsv(tsv, HEADER, got[0])
                    append_tsv(pc_tsv, PC_HEADER, got[1])
        else:
            say('>>> 无待评 run（TSV 已齐）')

    # ---------- 判读 ----------
    rows = load_tsv(tsv)
    lut = {(r['run'], r['split']): r for r in rows}
    pc = {}
    for r in load_tsv(pc_tsv):
        pc.setdefault((r['run'], r['split']), {})[r['class']] = (
            float(r['mAP50']), float(r['mAP50-95']))

    say()
    say('=' * 108)
    say('第二批评估 —— 噪声地板 vs 臂（按分辨率分开判读）')
    say('口径：探索阶段，只报方向与幅度；**区间错开才算真实效应**，两 split 必须一致')
    say('=' * 108)
    say(f'{"run":<32}{"split":>6}{"imgsz":>7}{"mAP50":>9}{"mAP50-95":>10}{"P":>8}{"R":>8}')
    say('-' * 108)
    for run in targets or sorted({r['run'] for r in rows}):
        for split in ('val', 'test'):
            r = lut.get((run, split))
            if not r:
                say(f'{run:<32}{split:>6}{"":>7}{"  缺":>9}')
                continue
            say(f'{run:<32}{split:>6}{r["imgsz"]:>7}{float(r["mAP50"]):>9.4f}'
                f'{float(r["mAP50-95"]):>10.4f}{float(r["P"]):>8.4f}{float(r["R"]):>8.4f}')

    for res in (896, 640):
        fl = [r for r in FLOOR if ('_%d_' % res) in r and (r, 'test') in lut]
        say()
        say(f'【{res} 噪声地板】baseline × {len(fl)} 种子（同脚本、workers=0、仅 seed 不同）')
        if len(fl) < 2:
            n = len(fl)
            say(f'  样本不足（{n}）⇒ 该分辨率**不可判**（需 ≥2 种子）。')
            say('  注：896 的种子噪声尺度未知时，896 臂的任何涨跌都无尺度可依。')
            continue
        tv = [float(lut[(r, "test")]['mAP50']) for r in fl]
        vv = [float(lut[(r, "val")]['mAP50']) for r in fl]
        flo, fval = rng(tv), rng(vv)
        for r in fl:
            say(f'  {r:<26} test {float(lut[(r,"test")]["mAP50"]):.4f}   '
                f'val {float(lut[(r,"val")]["mAP50"]):.4f}')
        say(f'  → test 区间 [{flo[0]:.4f}, {flo[1]:.4f}]  跨度 {(flo[1]-flo[0])*100:.2f} pt'
            f'   均值 {flo[2]:.4f}   n={flo[3]}')
        say(f'  → val  区间 [{fval[0]:.4f}, {fval[1]:.4f}]  跨度 {(fval[1]-fval[0])*100:.2f} pt'
            f'   均值 {fval[2]:.4f}   n={fval[3]}')
        say()
        say(f'  {"臂":<16}{"n":>3}{"test 区间":>24}{"Δ vs 地板均值":>16}{"test 判读":>22}{"val 判读":>22}')
        for label, names in ARMS.items():
            have = [r for r in names if ('_%d_' % res) in r and (r, 'test') in lut]
            if not have:
                continue
            at = [float(lut[(r, "test")]['mAP50']) for r in have]
            av = [float(lut[(r, "val")]['mAP50']) for r in have]
            alo, ahi, amu, n = rng(at)
            if n < 2:
                say(f'  {label:<16}{n:>3}{("  仅单种子（%.4f）" % alo):>24}'
                    f'{(amu - flo[2]) * 100:>+16.2f}{"不可判（需≥2 种子）":>22}{"":>22}')
                continue
            vt = '★ 错开 → 真实效应' if ahi < flo[0] or alo > flo[1] else '· 重叠 → 不可分'
            vv_ = ('★ 错开' if max(av) < fval[0] or min(av) > fval[1] else '· 重叠')
            if vt.startswith('★') and vv_.startswith('★'):
                vt += '（两 split 一致）'
            say(f'  {label:<16}{n:>3}{f"[{alo:.4f}, {ahi:.4f}]":>24}'
                f'{(amu - flo[2]) * 100:>+16.2f}{vt:>22}{vv_:>22}')
        # 【2026-10-17 文案更正】旧句"只在一边错开、另一边重叠 ⇒ 不构成真实效应"
        # 是 A.9.17 **单种子**镜像情形的规则，被误当成 n≥2 的判据（batch2 报告 §七
        # 据此把 640 CBAM+BN 的 test 错开降级，与本节开头预登记的
        # "test 错开 且 val 不反向矛盾" 相矛盾）。预登记只把 **val 反向错开**
        # （臂在 val 明显更差、却在 test 更好）当作否决条件；val 重叠不否决。
        say('  ⚠ 判据以本节开头预登记为准：test 错开 且 val **不反向**矛盾 即记效应；'
            'val 重叠不否决。A.9.17 的"单边错开即噪声"只适用于 n=1。')

    say()
    say('【per-class mAP50 / mAP50-95（test）】')
    say(f'{"run":<32}' + ''.join(f'{c:>20}' for c in CLASSES))
    say('-' * 108)
    for run in targets or sorted({r['run'] for r in rows}):
        if (run, 'test') not in lut:
            continue
        d = pc.get((run, 'test'), {})
        cells = ''.join(f'{d[c][0]:>10.4f}/{d[c][1]:<9.4f}' if c in d else f'{"—":>20}'
                        for c in CLASSES)
        say(f'{run:<32}{cells}')

    say()
    say('【口径声明 —— 引用这些数字时必须一并带上】')
    say('  探索阶段：地板 3 种子、臂 3 种子；只报方向与幅度，未做显著性检验。')
    say('  评估 imgsz = 训练 imgsz（由 run 名自动识别）；跨分辨率数字不可直接相比。')
    say('  832/896 与 640 是两条线，判读各自比各自的噪声地板。')
    say('  本批数字不得进入 v_final 或任何交付形态。')
    say('=' * 108)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n已追加写入 {LOG}')


if __name__ == '__main__':
    main()
