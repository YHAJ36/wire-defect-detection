# -*- coding: utf-8 -*-
"""反2（ultralytics 部分）：best.pt vs last.pt 在 test 上的 mAP50 —— 早停流程的净收益。

对每个 ultralytics report run，用同一口径（split=test / batch=8 / workers=0 /
device=0 / fp32，与 eval_test_640.py / eval_test_896.py 一致）分别评 best.pt 与 last.pt。

判定（用户钉死）：
  best > last ⇒ 早停有效（幅度 = 差）；
  last ≥ best ⇒ 用 val 早停在本数据集是负收益，整个 best.pt 报告流程要重新审视。

CBAM 模型反序列化需 yolov11/yolov12 的 cbam 可导入（两处 cbam.py 内容相同）。
每次 eval 全新加载 YOLO（避免复用串味）。
输出：logs/eval_best_vs_last.txt
用法：D:/yolo-env/Scripts/python.exe D:/2/code/tools/eval/eval_best_vs_last.py
"""
import sys
from pathlib import Path

for d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if d not in sys.path:
        sys.path.insert(0, d)

from ultralytics import YOLO  # noqa: E402

REPORT = [
    ('wire_v11_640', 640),
    ('wire_v11_640_neck', 640),
    ('wire_v11_640_backbone', 640),
    ('wire_v11_640_both', 640),
    ('wire_v12_640', 640),
    ('wire_v12_640_cbam', 640),
    ('wire_v11_896', 896),
]

OUT = Path('D:/2/logs/eval_best_vs_last.txt')


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
    say('反2：best.pt vs last.pt 在 test 上的 mAP50（ultralytics 模型）')
    say('=' * 96)
    say(f'{"run":<26}{"best mAP50":>13}{"last mAP50":>13}{"Δ(last−best)":>13}   判定')
    say('-' * 96)
    results = []
    for run, img in REPORT:
        row = {}
        for tag in ('best', 'last'):
            w = f'D:/2/runs/{run}/weights/{tag}.pt'
            m = YOLO(w)  # 每次全新加载
            r = m.val(data='D:/2/dataset_final_bbox/data.yaml', split='test',
                      imgsz=img, batch=8, workers=0, device=0,
                      project='D:/2/runs', name=f'{run}_{tag}_test', exist_ok=True,
                      verbose=False, plots=False)
            row[tag] = float(r.box.map50)
            say(f'  [done] {run:<24} {tag:<5} mAP50={row[tag]:.4f}')
        d = row['last'] - row['best']
        if row['best'] > row['last']:
            verdict = '早停有效'
        elif abs(d) < 0.0005:
            verdict = '打平'
        else:
            verdict = '早停负收益'
        say(f'{run:<26}{row["best"]:>13.4f}{row["last"]:>13.4f}{d:>+13.4f}   {verdict}')
        results.append((run, row['best'], row['last'], d, verdict))
    say('-' * 96)
    if results:
        diffs = [r[3] for r in results]
        n_pos = sum(1 for d in diffs if d > 0.0005)
        n_neg = sum(1 for d in diffs if d < -0.0005)
        n_tie = len(diffs) - n_pos - n_neg
        say(f'汇总：{len(results)} 个 run；best 占优 {n_pos}、last 占优 {n_neg}、打平 {n_tie}')
        say(f'  Δ(last−best) 范围 [{min(diffs):+.4f}, {max(diffs):+.4f}] 均值 {sum(diffs)/len(diffs):+.4f}')

    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n已写入 {OUT}')


if __name__ == '__main__':
    main()
