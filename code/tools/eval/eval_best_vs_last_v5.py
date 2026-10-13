# -*- coding: utf-8 -*-
"""反2（yolov5 部分）：best.pt vs last.pt 在 test 上的 mAP50。

yolov5 用自己的 val.py（YOLOv5 v7.0）评估，口径与 eval_test_v5_640.py 完全一致
（--task test / --img / --batch-size 8 / --workers 0），仅 weights 换 last.pt、name 换 *_last_test。
本脚本用 subprocess 调 val.py，抓 stdout 里 "all" 汇总行的 mAP50。

用法：D:/2/envs/yolov5-env/Scripts/python.exe D:/2/code/tools/eval/eval_best_vs_last_v5.py
"""
import re
import subprocess
import sys
from pathlib import Path

PY = 'D:/2/envs/yolov5-env/Scripts/python.exe'
V5DIR = 'D:/2/code/baselines/yolov5'
REPORT = [
    ('wire_v5_640', 640),
    ('wire_v5_896', 896),
]
OUT = Path('D:/2/logs/eval_best_vs_last_v5.txt')

ANSI = re.compile(r'\x1b\[[0-9;]*m')


def run_val(run, img, tag):
    args = [PY, 'val.py', '--data', 'D:/2/dataset_final_bbox/data.yaml',
            '--weights', f'D:/2/runs/{run}/weights/{tag}.pt',
            '--img', str(img), '--batch-size', '8', '--task', 'test',
            '--workers', '0', '--project', 'D:/2/runs',
            '--name', f'{run}_{tag}_test', '--exist-ok']
    p = subprocess.run(args, cwd=V5DIR, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    for line in (p.stdout + '\n' + p.stderr).splitlines():
        s = ANSI.sub('', line).strip()
        if s.startswith('all ') or s.startswith('all\t'):
            parts = s.split()
            if len(parts) >= 7:
                # all  Images  Instances  P  R  mAP50  mAP50-95
                return float(parts[5]), float(parts[6])
    return None, None


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    lines = []

    def say(s=''):
        lines.append(s)
        print(s, flush=True)

    say('=' * 96)
    say('反2：best.pt vs last.pt 在 test 上的 mAP50（yolov5 模型）')
    say('=' * 96)
    say(f'{"run":<20}{"best mAP50":>13}{"last mAP50":>13}{"Δ(last−best)":>13}   判定')
    say('-' * 96)
    for run, img in REPORT:
        b = run_val(run, img, 'best')
        say(f'  [done] {run:<18} best mAP50={b[0] if b else None}')
        l = run_val(run, img, 'last')
        say(f'  [done] {run:<18} last mAP50={l[0] if l else None}')
        if b is None or l is None:
            say(f'{run:<20}  解析失败（best={b} last={l}）')
            continue
        d = l[0] - b[0]
        if b[0] > l[0]:
            verdict = '早停有效'
        elif abs(d) < 0.0005:
            verdict = '打平'
        else:
            verdict = '早停负收益'
        say(f'{run:<20}{b[0]:>13.4f}{l[0]:>13.4f}{d:>+13.4f}   {verdict}')
    say('-' * 96)

    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
