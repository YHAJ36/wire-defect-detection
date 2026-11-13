# -*- coding: utf-8 -*-
"""重推理一次，把预测坐标存成**原图归一化**（而不是 letterbox 归一化），供 bootstrap 使用。

为什么必须重推：`logs/wbf_cache_noleak/*.npz` 里的坐标是除以 letterbox 后的
(hs, ws)=(896,896) 得到的，而 GT 是原图归一化。两者直接比 IoU 会系统性偏低
（实测 mAP50 差 4~11 pt）。而 letterbox 的 pad 量取决于同 batch 内其他图的尺寸，
无法离线复原 —— 所以让框架自己用 `ops.scale_boxes` 转回原图，一次存干净。

只读：只调 val() 做前向，不训练、不写权重。
用法：D:/yolo-env/Scripts/python.exe code/tools/eval/regen_cache_orig.py
"""
import sys
from pathlib import Path

# Triplet/CBAM 权重的 pickle 里引用了自定义模块，反序列化前必须可见
for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops

ROOT = Path('D:/2')
DATA = ROOT / 'dataset_noleak/data.yaml'
RUNS = ROOT / 'exp_upscale/runs'
OUT = ROOT / 'exp_upscale/logs/wbf_cache_noleak_orig'
OUT_TP = ROOT / 'exp_upscale/logs/tp_cache_noleak'
EVAL_DIR = ROOT / 'exp_upscale/eval_noleak'

RUNS_LIST = [
    ('nf_v11_noleak_s42', 896), ('nf_v11_noleak_s43', 896), ('nf_v11_noleak_s44', 896),
    ('att_v11_noleak_triplet_s42', 896), ('att_v11_noleak_triplet_s43', 896),
    ('att_v11_noleak_triplet_s44', 896),
]

_ORIG = DetectionValidator.update_metrics
HOOK = {'c': None}


def _patched(self, preds, batch):
    if HOOK['c'] is not None:
        for si, p in enumerate(preds):
            stem = Path(batch['im_file'][si]).stem
            h0, w0 = batch['ori_shape'][si]
            # 决策量：直接取框架自己的匹配结果，避免自己实现 IoU 匹配时在
            # IoU≈0.5 的边界上翻转（实测会差 0.15~2.35 pt，方向不定）
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(p)
            tp = self._process_batch(predn, pbatch)['tp']
            tp50 = tp[:, 0].astype(np.float32) if getattr(tp, 'size', 0) else np.zeros(0, np.float32)
            conf = predn['conf'].detach().cpu().numpy().astype(np.float32)
            pcls = predn['cls'].detach().cpu().numpy().astype(np.float32)
            gcls = pbatch['cls'].detach().cpu().numpy().astype(np.float32)
            if len(p['cls']):
                b = ops.scale_boxes(batch['img'].shape[2:],
                                    p['bboxes'].detach().cpu().clone(), (h0, w0))
                b = b.numpy().astype(np.float32) / np.array([w0, h0, w0, h0], np.float32)
                rec = {'boxes': b,
                       'conf': p['conf'].detach().cpu().numpy().astype(np.float32),
                       'cls': p['cls'].detach().cpu().numpy().astype(np.int64)}
            else:
                rec = {'boxes': np.zeros((0, 4), np.float32),
                       'conf': np.zeros((0,), np.float32),
                       'cls': np.zeros((0,), np.int64)}
            rec['orig'] = np.array([h0, w0], np.int32)
            rec['tp50'], rec['dconf'], rec['pcls'], rec['gcls'] = tp50, conf, pcls, gcls
            HOOK['c'][stem] = rec
    return _ORIG(self, preds, batch)


DetectionValidator.update_metrics = _patched


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_TP.mkdir(parents=True, exist_ok=True)
    for run, imgsz in RUNS_LIST:
        w = RUNS / run / 'weights' / 'best.pt'
        if not w.exists():
            print(f'缺 {w}')
            continue
        HOOK['c'] = {}
        try:
            r = YOLO(str(w)).val(data=str(DATA), split='test', imgsz=imgsz, batch=8,
                                 workers=0, device=0, plots=False, verbose=False,
                                 iou=0.5, agnostic_nms=False, project=str(EVAL_DIR),
                                 name=f'regen_{run}', exist_ok=True)
        finally:
            data = HOOK['c']
            HOOK['c'] = None
        o = {k: v for k, v in data.items()}
        p = OUT / f'{run}.npz'
        np.savez_compressed(
            p,
            **{k: np.concatenate([v['boxes'], v['conf'][:, None],
                                  v['cls'][:, None].astype(np.float32)], 1)
               for k, v in o.items()},
            __keys__=np.array(list(o.keys())),
            __orig__=np.array([o[k]['orig'] for k in o]),
        )
        # 决策缓存：框架给出的 tp（IoU=0.5）+ 与之对齐的 conf/pred_cls/gt_cls
        q = OUT_TP / f'{run}.npz'
        np.savez_compressed(
            q,
            __keys__=np.array(list(o.keys())),
            tp50=np.array([o[k]['tp50'] for k in o], dtype=object),
            dconf=np.array([o[k]['dconf'] for k in o], dtype=object),
            pcls=np.array([o[k]['pcls'] for k in o], dtype=object),
            gcls=np.array([o[k]['gcls'] for k in o], dtype=object),
        )
        print(f'{run:<30} n={len(o):<4} mAP50={float(r.box.map50):.4f} '
              f'mAP50-95={float(r.box.map):.4f}  -> {p.name} / {q.name}')


if __name__ == '__main__':
    main()
