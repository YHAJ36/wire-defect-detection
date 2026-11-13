# -*- coding: utf-8 -*-
"""在「干净 test 100 张」上评估推理端外挂：TTA / WBF / WBF+TTA（基线族 + Triplet 族）。

为什么这样做
------------
干净集基线 89.98% 是 `model.val()` 跑出来的（ultralytics 8.3.180，iou=0.5、
conf=0.001、max_det=300、rect=False → 全部 letterbox 到 896×896）。
WBF 是框架外的东西，若自己写一套 mAP 代码，数字与 89.98 不可比。
所以这里不重写指标：**注入点选在 DetectionValidator.update_metrics**。
该函数的 preds 是 NMS 之后、仍在 letterbox 坐标系里的框，GT 也在同一坐标系
（_prepare_batch 把 xywh 乘了 imgsz），因此只要把 preds 换成融合结果，
后面的 _process_batch / DetMetrics 全部原样跑，口径与基线逐位一致。

自检：把 raw s42 的缓存原样喂回去，必须复现 0.899828（=89.98%）。

组别
----
族 base    = nf_v11_896_s42/43/44
族 triplet = att_v11_896_triplet_s42/43/44
每组都跑：
  单模型 raw / 单模型 TTA（augment=True，3 尺度 1/0.83/0.67 + lr flip）
  WBF(raw) / WBF(TTA)
WBF 采用 ZFTurbo ensemble-boxes 的加权框融合语义（等权）：
    簇内坐标 = Σ(score_i·box_i)/Σscore_i
    簇得分   = (Σscore_i/N) · min(|cluster|,N)/N      N = 模型数 = 3
主口径 iou_thr=0.55（预先固定）；另跑 0.50/0.60 作敏感性，不当主结果。

用法
    D:/yolo-env/Scripts/python.exe code/tools/eval/eval_wbf_tta.py
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402

DATA = os.environ.get('NOLEAK_YAML', 'D:/2/dataset_noleak_clean/data.yaml')
SPLIT = 'test'
IMGSZ = 896
NMS_IOU = 0.5           # 与 eval_noleak.tsv / 89.98% 同口径
CONF = 0.001            # val 默认
MAX_DET = 300
BATCH = 8

RUNS = Path(os.environ.get('NOLEAK_RUNS', 'D:/2/exp_upscale/runs'))
CACHE = Path(os.environ.get('NOLEAK_CACHE', 'D:/2/exp_upscale/logs/wbf_cache'))
EVAL_DIR = Path(os.environ.get('NOLEAK_EVALDIR', 'D:/2/exp_upscale/eval_wbf_tta'))
TSV = Path(os.environ.get('NOLEAK_TSV', 'D:/2/exp_upscale/logs/eval_wbf_tta_clean.tsv'))
JSON_OUT = Path(os.environ.get('NOLEAK_JSON', 'D:/2/exp_upscale/logs/eval_wbf_tta_clean.json'))
SKIP_TTA = os.environ.get('NOLEAK_SKIP_TTA') == '1'

FAMILIES = {
    'base':    ['nf_v11_896_s42', 'nf_v11_896_s43', 'nf_v11_896_s44'],
    'triplet': ['att_v11_896_triplet_s42', 'att_v11_896_triplet_s43', 'att_v11_896_triplet_s44'],
}
# 段级无泄漏重训（2026-10-26）：run 名不同，用环境变量注入，默认值保持不变
if os.environ.get('NOLEAK_FAMILIES'):
    FAMILIES = json.loads(os.environ['NOLEAK_FAMILIES'])
CLASSES = ('break', 'thunderbolt', 'wear')
WBF_IOU_MAIN = 0.55
WBF_IOU_SWEEP = (0.50, 0.60)

# --------------------------------------------------------------------------
# 注入钩子
# --------------------------------------------------------------------------
_ORIG_UPDATE = DetectionValidator.update_metrics
HOOK = {'collect': None, 'provider': None}


def _patched_update(self, preds, batch):
    if HOOK['collect'] is not None:
        hs, ws = batch['img'].shape[2], batch['img'].shape[3]
        for si, p in enumerate(preds):
            key = Path(batch['im_file'][si]).stem
            rec = {'file': str(batch['im_file'][si]), 'imgsz': [hs, ws]}
            if len(p['cls']):
                b = p['bboxes'].detach().cpu().numpy().astype(np.float32).copy()
                b /= np.array([ws, hs, ws, hs], np.float32)
                rec['boxes'] = b
                rec['conf'] = p['conf'].detach().cpu().numpy().astype(np.float32)
                rec['cls'] = p['cls'].detach().cpu().numpy().astype(np.int64)
            else:
                rec['boxes'] = np.zeros((0, 4), np.float32)
                rec['conf'] = np.zeros((0,), np.float32)
                rec['cls'] = np.zeros((0,), np.int64)
            HOOK['collect'][key] = rec
    if HOOK['provider'] is not None:
        preds = HOOK['provider'](preds, batch, self)
    return _ORIG_UPDATE(self, preds, batch)


DetectionValidator.update_metrics = _patched_update

# --------------------------------------------------------------------------
# WBF
# --------------------------------------------------------------------------


def _iou_one(a, b):
    """a: (4,)  b: (N,4)  归一化 xyxy"""
    if len(b) == 0:
        return np.zeros((0,), np.float32)
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(aa + ab - inter, 1e-9)


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, n_models, iou_thr=0.55):
    """ZFTurbo ensemble-boxes 的 wbf 语义（等权 weight=1）。输入/输出均为归一化 xyxy。"""
    all_b = np.concatenate(boxes_list, 0)
    all_s = np.concatenate(scores_list, 0)
    all_c = np.concatenate(labels_list, 0)
    if len(all_s) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)

    ob, os_, oc = [], [], []
    for c in np.unique(all_c):
        m = all_c == c
        b = all_b[m].astype(np.float32)
        s = all_s[m].astype(np.float32)
        order = np.argsort(-s, kind='stable')
        b, s = b[order], s[order]
        alive = np.ones(len(s), dtype=bool)
        while alive.any():
            idx = np.flatnonzero(alive)
            ious = _iou_one(b[idx[0]], b[idx])
            sel = idx[ious > iou_thr]          # 含参考框自身（IoU=1）
            w = s[sel]
            ob.append((b[sel] * w[:, None]).sum(0) / max(w.sum(), 1e-9))
            os_.append((w.sum() / n_models) * (min(len(sel), n_models) / n_models))
            oc.append(c)
            alive[sel] = False
    return (np.asarray(ob, np.float32), np.asarray(os_, np.float32),
            np.asarray(oc, np.int64))


# --------------------------------------------------------------------------
# 跑一次 val
# --------------------------------------------------------------------------


def run_val(run, augment, collect=None, provider=None, tag='x'):
    HOOK['collect'] = collect
    HOOK['provider'] = provider
    try:
        model = YOLO(str(RUNS / run / 'weights' / 'best.pt'))
        r = model.val(data=DATA, split=SPLIT, imgsz=IMGSZ, batch=BATCH, workers=0,
                      device=0, plots=False, verbose=False, iou=NMS_IOU, conf=CONF,
                      max_det=MAX_DET, agnostic_nms=False, augment=augment,
                      project=str(EVAL_DIR), name=tag, exist_ok=True)
    finally:
        HOOK['collect'] = None
        HOOK['provider'] = None
    b = r.box
    pc = {r.names[int(c)]: {'ap50': float(b.ap50[j]), 'ap': float(b.ap[j])}
          for j, c in enumerate(b.ap_class_index)}
    return {'mAP50': float(b.map50), 'mAP50-95': float(b.map),
            'P': float(b.mp), 'R': float(b.mr), 'per_class': pc}


def save_cache(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.concatenate(
        [v['boxes'], v['conf'][:, None], v['cls'][:, None].astype(np.float32)], 1)
        for k, v in data.items()},
        __keys__=np.array(list(data.keys())))


def make_provider(caches, iou_thr, n_models):
    def provider(preds, batch, validator):
        hs, ws = batch['img'].shape[2], batch['img'].shape[3]
        dev = batch['bboxes'].device
        out = []
        for si in range(len(batch['im_file'])):
            stem = Path(batch['im_file'][si]).stem
            bl = [c[stem]['boxes'] for c in caches]
            sl = [c[stem]['conf'] for c in caches]
            cl = [c[stem]['cls'] for c in caches]
            fb, fs, fc = weighted_boxes_fusion(bl, sl, cl, n_models, iou_thr)
            if len(fs):
                fb = fb * np.array([ws, hs, ws, hs], np.float32)
            out.append({'bboxes': torch.from_numpy(fb).to(dev),
                        'conf': torch.from_numpy(fs).to(dev),
                        'cls': torch.from_numpy(fc.astype(np.float32)).to(dev),
                        'extra': torch.zeros((len(fs), 0), device=dev)})
        return out
    return provider


def mean_of(mets):
    """三种子均值（AP50 / AP50-95 / P / R / 逐类）+ AP50 跨度。"""
    out = {'mAP50': float(np.mean([m['mAP50'] for m in mets])),
           'mAP50-95': float(np.mean([m['mAP50-95'] for m in mets])),
           'P': float(np.mean([m['P'] for m in mets])),
           'R': float(np.mean([m['R'] for m in mets])),
           'per_class': {}}
    for c in CLASSES:
        out['per_class'][c] = {
            'ap50': float(np.mean([m['per_class'].get(c, {'ap50': np.nan})['ap50'] for m in mets])),
            'ap': float(np.mean([m['per_class'].get(c, {'ap': np.nan})['ap'] for m in mets])),
        }
    out['span50'] = float(max(m['mAP50'] for m in mets) - min(m['mAP50'] for m in mets)) * 100
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    print(f'ultralytics {ultralytics.__version__}   torch {torch.__version__}')
    print(f'data={DATA}  split={SPLIT}  imgsz={IMGSZ}  iou={NMS_IOU}  conf={CONF}  max_det={MAX_DET}')
    rows = []

    def push(family, group, met, note='', nseed=1):
        pc = met['per_class']
        rows.append({'family': family, 'group': group, 'note': note, 'nseed': nseed,
                     'mAP50': met['mAP50'], 'mAP50-95': met['mAP50-95'],
                     'P': met['P'], 'R': met['R'],
                     **{f'{c}_ap50': pc.get(c, {}).get('ap50', float('nan')) for c in CLASSES},
                     **{f'{c}_ap': pc.get(c, {}).get('ap', float('nan')) for c in CLASSES}})

    for fam, seeds in FAMILIES.items():
        print(f'\n########## 族 {fam} ##########')
        cache, native = {}, {}
        modes = ('raw',) if SKIP_TTA else ('raw', 'tta')
        for run in seeds:
            for mode in modes:
                c, t0 = {}, time.time()
                met = run_val(run, augment=(mode == 'tta'), collect=c,
                              tag=f'collect_{run}_{mode}')
                cache[(run, mode)] = c
                native[(run, mode)] = met
                save_cache(CACHE / f'{run}_{mode}.npz', c)
                print(f'  [collect] {run:<26}{mode:<4} imgs={len(c):>3}  '
                      f'avg_box={np.mean([len(v["conf"]) for v in c.values()]):6.1f}  '
                      f'mAP50={met["mAP50"]:.6f}  mAP50-95={met["mAP50-95"]:.6f}  ({time.time()-t0:.0f}s)')

        raws = [native[(r, 'raw')] for r in seeds]
        for i, r in enumerate(seeds):
            push(fam, f'单模型 {r.split("_")[-1]}', raws[i], 'single raw')
        m_raw = mean_of(raws)
        push(fam, '单模型 raw 三种子均值', m_raw, f'span={m_raw["span50"]:.2f}pt', nseed=3)
        if not SKIP_TTA:
            ttas = [native[(r, 'tta')] for r in seeds]
            for i, r in enumerate(seeds):
                push(fam, f'单模型 {r.split("_")[-1]} +TTA', ttas[i], 'single TTA')
            m_tta = mean_of(ttas)
            push(fam, '单模型 TTA 三种子均值', m_tta, f'span={m_tta["span50"]:.2f}pt', nseed=3)

        fuse_modes = [('raw', [cache[(r, 'raw')] for r in seeds], 'WBF')]
        if not SKIP_TTA:
            fuse_modes.append(('tta', [cache[(r, 'tta')] for r in seeds], 'WBF+TTA'))
        for mode, caches, label in fuse_modes:
            for thr in (WBF_IOU_MAIN,) + WBF_IOU_SWEEP:
                star = '' if thr == WBF_IOU_MAIN else '*'
                met = run_val(seeds[0], augment=False,
                              provider=make_provider(caches, thr, len(seeds)),
                              tag=f'{fam}_{mode}_iou{thr:.2f}')
                push(fam, f'{label}{star} 三种子融合', met, f'WBF iou={thr}', nseed=3)
                print(f'  [fuse] {fam:<8}{label:<8}iou={thr:.2f}  '
                      f'mAP50={met["mAP50"]:.6f}  mAP50-95={met["mAP50-95"]:.6f}')

    # ---- 输出 ----
    hdr = ['family', 'group', 'note', 'nseed', 'mAP50', 'mAP50-95', 'P', 'R',
           'break_ap50', 'thunderbolt_ap50', 'wear_ap50', 'break_ap', 'thunderbolt_ap', 'wear_ap']
    TSV.parent.mkdir(parents=True, exist_ok=True)
    with TSV.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(hdr)
        for r in rows:
            line = []
            for k in hdr:
                v = r[k]
                if isinstance(v, str):
                    line.append(v)
                elif isinstance(v, int):
                    line.append(str(v))
                else:
                    line.append(f'{v:.4f}')
            w.writerow(line)
    JSON_OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

    for fam in FAMILIES:
        sub = [r for r in rows if r['family'] == fam]
        base = next(r for r in sub if r['group'] == '单模型 raw 三种子均值')
        base95 = base['mAP50-95']
        print('\n' + '=' * 116)
        print(f'族 {fam}   (Δ 相对该族「单模型 raw 三种子均值」= {base["mAP50"]*100:.2f})')
        print(f'{"组":<28}{"mAP50":>9}{"Δ50":>8}{"mAP50-95":>11}{"Δ95":>8}'
              f'{"break":>9}{"thunder":>9}{"wear":>9}{"P":>8}{"R":>8}')
        print('-' * 116)
        for r in sub:
            print(f'{r["group"]:<28}{r["mAP50"]*100:>9.2f}{(r["mAP50"]-base["mAP50"])*100:>+8.2f}'
                  f'{r["mAP50-95"]*100:>11.2f}{(r["mAP50-95"]-base95)*100:>+8.2f}'
                  f'{r["break_ap50"]:>9.4f}{r["thunderbolt_ap50"]:>9.4f}{r["wear_ap50"]:>9.4f}'
                  f'{r["P"]:>8.4f}{r["R"]:>8.4f}')
        print('=' * 116)

    print(f'\n已写 {TSV}\n已写 {JSON_OUT}\n缓存 {CACHE}')


if __name__ == '__main__':
    main()
