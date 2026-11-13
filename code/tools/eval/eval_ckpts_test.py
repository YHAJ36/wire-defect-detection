# -*- coding: utf-8 -*-
"""对若干检查点在 test 上按交付口径（NMS 0.5）评一次，供 60 vs 150 预算对照。

为什么单独写
    A.9.46 量到：`DetectionValidator(model=)` 与 `YOLO.val()` 在同一权重的 test 上
    差 1.41 pt。任何横向对比**必须同路径**。本脚本与 `reselect_nms05.py` 一致，
    统一走 `YOLO(...).val()`，因此它的数字可以和 reselect 的栏位直接并列。

主对比口径（无挑点偏差）
    两个 run 的 **last.pt** 都是"cosine 走到 0 的终点模型"，规则固定、没有"K 个里挑最好"
    的选择偏差，是 60 vs 150 最干净的对照。best.pt 受训练期 iou=0.7 影响、且 150 轮
    可挑的检查点更多（偏差幅度不同），只能作参考。

用法
    D:/yolo-env/Scripts/python.exe code/tools/eval/eval_ckpts_test.py \
        --ckpt runs/nf_v11_896_s42/weights/last.pt \
        --ckpt runs/nf_v11_896_e150_s42/weights/last.pt \
        --imgsz 896 --tag lastpt_60v150
"""
import argparse
import json
import sys
from pathlib import Path

for _d in ('D:/2/code/yolov11', 'D:/2/code/yolov12'):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from ultralytics import YOLO  # noqa: E402

DATA = 'D:/2/dataset_final_bbox/data.yaml'
OUT = Path('D:/2/report/_p2_diag')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True,
                    help='可重复；给 .pt 路径或 run 名/weights/best.pt 相对 RUNS 的路径')
    ap.add_argument('--imgsz', type=int, default=896)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--split', default='test')
    ap.add_argument('--device', default='0')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--data', default=DATA,
                    help='数据集 yaml；默认 test 集。合并 val+test 终评时传 '
                         'D:/2/dataset_final_bbox/_valtest/data.yaml')
    args = ap.parse_args()

    rows = []
    for spec in args.ckpt:
        p = Path(spec)
        if not p.is_absolute():
            p = Path('D:/2/exp_upscale/runs') / spec
        if not p.exists():
            raise SystemExit(f'❌ 不存在：{p}')
        r = YOLO(str(p)).val(data=args.data, split=args.split, imgsz=args.imgsz,
                             batch=args.batch, workers=0, device=args.device,
                             plots=False, verbose=False, iou=args.iou,
                             agnostic_nms=False)
        b = r.box
        row = dict(ckpt=str(p), name=f'{p.parent.parent.name}/{p.name}',
                   mAP50=float(b.map50), mAP95=float(b.map),
                   P=float(b.mp), R=float(b.mr))
        # 逐类 AP50，用来核对是否只是某一类在动
        try:
            names = r.names
            per = {names[i]: float(v) for i, v in enumerate(b.ap50)}
            row['ap50_per_class'] = per
        except Exception:
            row['ap50_per_class'] = None
        rows.append(row)
        print(f'  {row["name"]:<44} mAP50={row["mAP50"]:.4f}  '
              f'mAP50-95={row["mAP95"]:.4f}  P={row["P"]:.4f}  R={row["R"]:.4f}')
        if row['ap50_per_class']:
            print('      per-class AP50: ' +
                  '  '.join(f'{k}={v:.4f}' for k, v in row['ap50_per_class'].items()))

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f'eval_{args.tag}.json'
    out.write_text(json.dumps(dict(data=args.data, split=args.split, imgsz=args.imgsz,
                                   iou=args.iou, rows=rows),
                              ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n→ {out}')

    if len(rows) >= 2:
        a, b = rows[0], rows[1]
        print(f'\n=== {a["name"]}  vs  {b["name"]} ===')
        print(f'  Δ mAP50    {(b["mAP50"] - a["mAP50"]) * 100:+.2f} pt')
        print(f'  Δ mAP50-95 {(b["mAP95"] - a["mAP95"]) * 100:+.2f} pt')
        print('  门槛（A.9.47 三档）：≥+3.7 真瓶颈 / 0~3.7 不可判 / ≤0 无收益')


if __name__ == '__main__':
    main()
