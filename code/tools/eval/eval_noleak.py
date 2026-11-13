# -*- coding: utf-8 -*-
"""在「段级干净」评估集上重跑推理，量化泄漏对指标的影响。

评估集（D:/2/dataset_noleak_clean/data.yaml，由 split_by_scene.py 的段划分导出）
    test = 100 张：原 test 中、所在视觉段不含任何原 train 图像的部分
                    —— 既没被训练过，也没参与 best.pt 选择
    val  = 92 张：原 val 中的同类图（**仅供参考**，best.pt 是按原 val 选的，有选择偏差）

口径与 eval_nms05.py 逐项一致：imgsz 由 run 名识别、batch 8、workers 0、
device 0、iou=0.5、agnostic_nms=False、conf 取 ultralytics val 默认（0.001）。

用法
    D:/yolo-env/Scripts/python.exe code/tools/eval/eval_noleak.py <run> [<run> ...]
    D:/yolo26-env/Scripts/python.exe code/tools/eval/eval_noleak.py nf_yolo26_896_s42
"""
import csv
import json
import re
import sys
from pathlib import Path

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from ultralytics import YOLO  # noqa: E402
import ultralytics  # noqa: E402

_ULTRA = tuple(int(x) for x in ultralytics.__version__.split('.')[:2])

import os as _os
DATA = _os.environ.get('NOLEAK_YAML', 'D:/2/dataset_noleak_clean/data.yaml')
SPLITS = _os.environ.get('NOLEAK_SPLITS', 'test,val').split(',')
TAG = _os.environ.get('NOLEAK_TAG', '')
RUNS = Path('D:/2/exp_upscale/runs')
EVAL_DIR = Path('D:/2/exp_upscale/eval_noleak')
TSV = Path('D:/2/exp_upscale/logs/eval_noleak.tsv')
NMS_IOU = 0.5


def imgsz_of(run):
    m = re.search(r'_(640|896|1024)(?:_|$)', run)
    if not m:
        raise SystemExit(f'无法从 run 名识别分辨率：{run}')
    return int(m.group(1))


def guard(runs):
    need = [r for r in runs if 'yolo26' in r.lower()]
    if need and _ULTRA < (8, 4):
        raise SystemExit(f'拒绝运行：{need} 需要 ultralytics>=8.4，当前 {ultralytics.__version__}')


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    runs = sys.argv[1:]
    if not runs:
        raise SystemExit('用法：eval_noleak.py <run> [<run> ...]')
    guard(runs)
    print(f'ultralytics {ultralytics.__version__}   data={DATA}')
    rows = []
    for run in runs:
        w = RUNS / run / 'weights' / 'best.pt'
        if not w.exists():
            print(f'缺 best.pt: {run}')
            continue
        sz = imgsz_of(run)
        for split in SPLITS:
            m = YOLO(str(w))
            r = m.val(data=DATA, split=split, imgsz=sz, batch=8, workers=0,
                      device=0, plots=False, verbose=False, iou=NMS_IOU,
                      agnostic_nms=False, project=str(EVAL_DIR),
                      name=f'{run}_{split}', exist_ok=True)
            b = r.box
            pc = {r.names[int(c)]: {'ap50': float(b.ap50[j]), 'ap': float(b.ap[j])}
                  for j, c in enumerate(b.ap_class_index)}
            rows.append({'run': run, 'split': split + TAG, 'imgsz': sz,
                         'mAP50': float(b.map50), 'mAP50-95': float(b.map),
                         'P': float(b.mp), 'R': float(b.mr), 'per_class': pc})
            print(f'  {run:<26}{split:>5} imgsz={sz}  mAP50={float(b.map50):.4f}  '
                  f'mAP50-95={float(b.map):.4f}  P={float(b.mp):.4f} R={float(b.mr):.4f}  '
                  + '  '.join(f'{k}={v["ap50"]:.3f}' for k, v in pc.items()))
    if rows:
        TSV.parent.mkdir(parents=True, exist_ok=True)
        old = []
        if TSV.exists():
            with TSV.open(encoding='utf-8', newline='') as f:
                old = [r for r in csv.DictReader(f, delimiter='\t')
                       if (r['run'], r['split']) not in {(x['run'], x['split']) for x in rows}]
        hdr = ['run', 'split', 'imgsz', 'mAP50', 'mAP50-95', 'P', 'R', 'per_class']
        with TSV.open('w', encoding='utf-8', newline='') as f:
            wr = csv.writer(f, delimiter='\t')
            wr.writerow(hdr)
            for r in old:
                wr.writerow([r[k] for k in hdr])
            for r in rows:
                wr.writerow([r['run'], r['split'], r['imgsz'], f'{r["mAP50"]:.6f}',
                             f'{r["mAP50-95"]:.6f}', f'{r["P"]:.6f}', f'{r["R"]:.6f}',
                             json.dumps(r['per_class'], ensure_ascii=False)])
        print(f'\n已写 {TSV}')


if __name__ == '__main__':
    main()
