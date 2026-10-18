# -*- coding: utf-8 -*-
"""把旧 per-class TSV 重建成带 split 列的 v2（离线，不加载模型、不评估）。

背景（WORKLOG A.9.19 §三 / A.9.20 §七）：
  `batch2_eval_perclass.tsv` 的旧 schema 是 run,class,mAP50,mAP50-95，
  **没有 split 列**；一个 run 的 val/test 只能靠“先写 val、后写 test”的行序区分。
  主判读表 batch2_eval.tsv 有 split 列，故主结论不受影响，但 per-class 记录易误读。

本脚本做的事：
  1. 按“每 run 连续 6 行 = val 三类 + test 三类”切组；
  2. 用主表 batch2_eval.tsv 的 mAP50 校验：每类三行的均值必须等于该 run/split 的 mAP50；
  3. 写出 `batch2_eval_perclass_v2.tsv`（run, split, class, mAP50, mAP50-95）。

原文件**不修改、不覆盖**；v2 已存在且内容不同则报错退出（避免静默覆盖）。

用法：
  D:/yolo-env/Scripts/python.exe D:/2/code/tools/eval/regen_perclass_split.py
"""
import csv
import sys
from pathlib import Path

LOGS = Path('D:/2/exp_upscale/logs')
OLD = LOGS / 'batch2_eval_perclass.tsv'
MAIN = LOGS / 'batch2_eval.tsv'
OUT = LOGS / 'batch2_eval_perclass_v2.tsv'
CLASSES = ['break', 'thunderbolt', 'wear']
TOL = 5e-4          # 主表与均值都保留 6 位小数，差值只应来自四舍五入


def load(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def main():
    old = load(OLD)
    main_rows = {(r['run'], r['split']): r for r in load(MAIN)}
    if len(old) % 6:
        raise SystemExit(f'旧表行数 {len(old)} 不是 6 的倍数，无法按 val/test 切组')

    out_rows = []
    for i in range(0, len(old), 6):
        chunk = old[i:i + 6]
        runs = {r['run'] for r in chunk}
        if len(runs) != 1:
            raise SystemExit(f'第 {i // 6 + 1} 组混入了多个 run：{runs}')
        run = runs.pop()
        for split, part in (('val', chunk[:3]), ('test', chunk[3:])):
            if sorted(r['class'] for r in part) != sorted(CLASSES):
                raise SystemExit(f'{run} {split} 的类别不完整：'
                                 f'{[r["class"] for r in part]}')
            mean50 = sum(float(r['mAP50']) for r in part) / 3
            ref = float(main_rows[(run, split)]['mAP50'])
            if abs(mean50 - ref) > TOL:
                raise SystemExit(f'{run} {split} 校验失败：三类均值 {mean50:.6f} '
                                 f'vs 主表 {ref:.6f}')
            for r in part:
                out_rows.append([run, split, r['class'], r['mAP50'], r['mAP50-95']])

    header = ['run', 'split', 'class', 'mAP50', 'mAP50-95']
    if OUT.exists():
        with OUT.open(encoding='utf-8', newline='') as f:
            existing = list(csv.reader(f, delimiter='\t'))
        want = [header] + out_rows
        if existing != want:
            raise SystemExit(f'{OUT} 已存在且内容不同；不覆盖，请人工处理')
        print(f'v2 已存在且一致：{OUT}')
        return

    with OUT.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(header)
        w.writerows(out_rows)
    print(f'OK  {OUT}')
    print(f'rows = {len(out_rows)}  ({len(out_rows) // 6} runs x 2 splits x 3 classes)')
    print('all groups verified against main TSV mAP50 (tol 5e-4)')


if __name__ == '__main__':
    sys.exit(main())
