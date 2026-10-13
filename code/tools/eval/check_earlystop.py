# -*- coding: utf-8 -*-
"""Q6：逐 run 查 fitness-best 轮 vs mAP50-best 轮，量化"选择口径错位"。

fitness = 0.1·mAP50 + 0.9·mAP50-95（ultralytics 与 yolov5 同公式）。
best.pt 按 fitness 选；论文报告 mAP50。故两准则会选出不同轮。

⚠ 两个已踩的坑（本脚本已处理）：
  1. yolov5 的 results.csv 表头带前导空格（'               epoch'），须 strip。
  2. 部分 run（neck / backbone）被训练了两次，results.csv 里两段串联
     （epoch 中途重置）。只取**最后一段**（当前 best.pt/last.pt 来自最后一段）。

纯读 results.csv，CPU，无副作用。
输出：stdout + logs/check_earlystop.txt
用法：D:/yolo-env/Scripts/python.exe D:/2/code/tools/eval/check_earlystop.py
"""
import csv
import io
import sys
from pathlib import Path

REPORT = [
    ('wire_v11_640', 640),
    ('wire_v11_640_neck', 640),
    ('wire_v11_640_backbone', 640),
    ('wire_v11_640_both', 640),
    ('wire_v12_640', 640),
    ('wire_v12_640_cbam', 640),
    ('wire_v5_640', 640),
    ('wire_v11_896', 896),
    ('wire_v5_896', 896),
]


def read_last_segment(path):
    """读 results.csv，剥离串联段，返回 (rows, n_segments)。rows 为 dict 列表（键已 strip）。"""
    raw = path.read_text(encoding='utf-8')
    rdr = csv.reader(io.StringIO(raw))
    header = [c.strip() for c in next(rdr)]
    rows = []
    for r in rdr:
        if not r or not any(x.strip() for x in r):
            continue
        rows.append(dict(zip(header, r)))
    if not rows:
        return [], 0
    # epoch 索引
    ecol = None
    for i, h in enumerate(header):
        if h.lower() == 'epoch':
            ecol = i
            break
    if ecol is None:
        # 无 epoch 列：当单段处理
        return rows, 1
    def ep(r):
        try:
            return int(float(r[header[ecol]]))
        except Exception:
            return None
    # 检测 reset（epoch 下降）
    segs, cur = [], []
    prev = None
    for r in rows:
        e = ep(r)
        if e is None:
            cur.append(r)
            continue
        if prev is not None and e < prev:
            segs.append(cur)
            cur = []
        cur.append(r)
        prev = e
    segs.append(cur)
    return segs[-1], len(segs)


def find_col(header, which):
    for c in header:
        if 'map' not in c.lower():
            continue
        if which == '50':
            if '95' in c:
                continue
            if ('50' in c) or ('0.5' in c):
                return c
        else:
            if '95' in c:
                return c
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    say('=' * 104)
    say('Q6：fitness-best 轮 vs mAP50-best 轮（选择口径错位的逐 run 量化，仅取最后一段）')
    say('=' * 104)
    say(f'{"run":<24}{"段数":>4}{"fit最佳轮":>9}{"该轮mAP50":>11}{"mAP50最佳轮":>11}{"该轮mAP50":>11}{"Δ(mAP50)点":>11}')
    say('-' * 104)
    rows_out = []
    for run, _img in REPORT:
        p = Path(f'D:/2/runs/{run}/results.csv')
        if not p.exists():
            say(f'{run:<24}   （无 results.csv）')
            continue
        rows, nseg = read_last_segment(p)
        if not rows:
            continue
        header = list(rows[0].keys())
        c50 = find_col(header, '50')
        c95 = find_col(header, '95')
        ecol = None
        for i, h in enumerate(header):
            if h.lower() == 'epoch':
                ecol = i
                break
        if c50 is None or c95 is None:
            say(f'{run:<24}  列未识别（c50={c50} c95={c95}）')
            continue

        def val(r, c):
            try:
                return float(r[c])
            except Exception:
                return 0.0

        def fit(r):
            return 0.1 * val(r, c50) + 0.9 * val(r, c95)

        best_fit = max(rows, key=fit)
        best_50 = max(rows, key=lambda r: val(r, c50))
        d = val(best_50, c50) - val(best_fit, c50)
        ep_f = rows.index(best_fit)
        ep_50 = rows.index(best_50)
        seg_tag = f'  段数={nseg}' if nseg > 1 else ''
        say(f'{run:<24}{nseg:>4}{best_fit[header[ecol]]:>9}{val(best_fit, c50):>11.4f}'
            f'{best_50[header[ecol]]:>11}{val(best_50, c50):>11.4f}{d:>+11.4f}{seg_tag}')
        rows_out.append((run, d))
    say('-' * 104)
    if rows_out:
        ds = [d for _, d in rows_out]
        say(f'  共 {len(rows_out)} 个 run；Δ(mAP50) 范围 [{min(ds):+.4f}, {max(ds):+.4f}] 均值 {sum(ds)/len(ds):+.4f} 点')
        say('  Δ = mAP50-best 轮的 mAP50 − fitness-best 轮的 mAP50（点）。正 = 按论文指标选可更高。')

    OUT = Path('D:/2/logs/check_earlystop.txt')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
