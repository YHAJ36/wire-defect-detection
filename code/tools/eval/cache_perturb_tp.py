# -*- coding: utf-8 -*-
"""为「数据侧扰动」新跑的三组（A jitter / B drop / C multi_scale）生成
决策缓存，并顺便输出 203 图段级 test 上的 mAP50 / 每类 AP50。

缓存格式与 `regen_cache_orig.py` 产出的 `tp_cache_noleak/*.npz` **完全一致**
（__keys__ / tp50 / dconf / pcls / gcls），因此可以直接喂给
`bootstrap_ci_noleak.py` 里的 map50() 与 cluster bootstrap 逻辑。

口径与 v3 完全对齐：dataset_noleak（test=203）、imgsz=896、batch 8、
workers 0、device 0、iou=0.5、conf 默认 0.001、max_det 300。

只做前向推理，不训练、不改权重。

用法
    D:/yolo-env/Scripts/python.exe code/tools/eval/cache_perturb_tp.py \
        nf_v11_noleak_jit08_s42 nf_v11_noleak_jit08_s43 ...
"""
import csv
import sys
from pathlib import Path

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402

ROOT = Path('D:/2')
DATA = ROOT / 'dataset_noleak/data.yaml'
RUNS = ROOT / 'exp_upscale/runs'
OUT_TP = ROOT / 'exp_upscale/logs/tp_cache_noleak'
EVAL_DIR = ROOT / 'exp_upscale/eval_noleak'
TSV = ROOT / 'exp_upscale/logs/eval_perturb_noleak.tsv'

_ORIG = DetectionValidator.update_metrics
HOOK = {'c': None}


def _patched(self, preds, batch):
    if HOOK['c'] is not None:
        for si, p in enumerate(preds):
            stem = Path(batch['im_file'][si]).stem
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(p)
            tp = self._process_batch(predn, pbatch)['tp']
            tp50 = tp[:, 0].astype(np.float32) if getattr(tp, 'size', 0) else np.zeros(0, np.float32)
            HOOK['c'][stem] = {
                'tp50': tp50,
                'dconf': predn['conf'].detach().cpu().numpy().astype(np.float32),
                'pcls': predn['cls'].detach().cpu().numpy().astype(np.float32),
                'gcls': pbatch['cls'].detach().cpu().numpy().astype(np.float32),
            }
    return _ORIG(self, preds, batch)


DetectionValidator.update_metrics = _patched


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--imgsz', type=int, default=896)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--suffix', default='', help='缓存文件名后缀，如 _r1280')
    args = ap.parse_args()
    runs = args.runs
    if not runs:
        raise SystemExit('用法: cache_perturb_tp.py [--imgsz 1280] <run> [<run> ...]')
    OUT_TP.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in runs:
        w = RUNS / run / 'weights' / 'best.pt'
        if not w.exists():
            print(f'缺 {w}')
            continue
        HOOK['c'] = {}
        try:
            r = YOLO(str(w)).val(data=str(DATA), split='test', imgsz=args.imgsz, batch=args.batch,
                                 workers=0, device=0, plots=False, verbose=False,
                                 iou=0.5, agnostic_nms=False,
                                 project=str(EVAL_DIR), name=f'perturb_{run}', exist_ok=True)
        finally:
            o = HOOK['c'] or {}
            HOOK['c'] = None
        q = OUT_TP / f'{run}{args.suffix}.npz'
        np.savez_compressed(
            q,
            __keys__=np.array(list(o.keys())),
            tp50=np.array([o[k]['tp50'] for k in o], dtype=object),
            dconf=np.array([o[k]['dconf'] for k in o], dtype=object),
            pcls=np.array([o[k]['pcls'] for k in o], dtype=object),
            gcls=np.array([o[k]['gcls'] for k in o], dtype=object),
        )
        names = ['break', 'thunderbolt', 'wear']
        ap50 = [float(x) for x in np.asarray(r.box.ap50)]
        row = {'run': run + args.suffix, 'n_img': len(o), 'mAP50': float(r.box.map50),
               'mAP50-95': float(r.box.map), 'P': float(r.box.mp), 'R': float(r.box.mr),
               **{f'{n}_ap50': (ap50[i] if i < len(ap50) else float('nan'))
                  for i, n in enumerate(names)}}
        rows.append(row)
        print(f"{run:<32} n={len(o):<4} mAP50={row['mAP50']:.4f} "
              f"mAP50-95={row['mAP50-95']:.4f} "
              f"break={row['break_ap50']:.4f} thunder={row['thunderbolt_ap50']:.4f} "
              f"wear={row['wear_ap50']:.4f}  -> {q.name}")

    if rows:
        new = not TSV.exists()
        with TSV.open('a', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter='\t')
            if new:
                wr.writeheader()
            wr.writerows(rows)
        print(f'\n→ {TSV}')


if __name__ == '__main__':
    main()
